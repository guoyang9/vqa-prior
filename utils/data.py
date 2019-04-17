import json
import os, sys
import os.path
import re

from PIL import Image
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import utils.config as config
import utils.utils as utils

def get_loader(train=False, val=False, test=False, need_test=False, version='v1'):
    """ Returns a data loader for the desired split """
    assert train + val + test == 1, 'need to set exactly one of {train, val, test} to True'
    vocab_path = getattr(config, 'vocabulary_path_{}'.format(version))
    if not need_test:
        split = VQATrainOrVal(
            utils.path_for(train=train, val=val, question=True, version=version),
            utils.path_for(train=train, val=val, answer=True, version=version),
            config.preprocessed_path,
            vocab_path = vocab_path,
            answerable_only=train,
        )
    else:
        if not test:
            split_train = VQATrainOrVal(
                utils.path_for(train=True, question=True, version=version),
                utils.path_for(train=True, answer=True, version=version),
                config.preprocessed_path,
                vocab_path = vocab_path,
                answerable_only=train,
            )
            split_val = VQATrainOrVal(
                utils.path_for(val=True, question=True, version=version),
                utils.path_for(val=True, answer=True, version=version),
                config.preprocessed_path,
                vocab_path = vocab_path,
                answerable_only=val,
            )
            splits = [split_train, split_val]
            split = Composite(*splits)
        else:
            split = VQATest(
                utils.path_for(test=True, question=True, version=version),
                config.preprocessed_path,
                vocab_path = vocab_path,
            )

    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config.batch_size,
        shuffle=train,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
        collate_fn=collate_fn,
    )
    return loader


def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)


class VQATrainOrVal(data.Dataset):
    """ VQA train dataset, open-ended """
    def __init__(self, questions_path, 
                       answers_path, 
                       image_features_path, 
                       vocab_path,
                       answerable_only=False):
        super(VQATrainOrVal, self).__init__()
        with open(questions_path, 'r') as fd:
            questions_json = json.load(fd)
        with open(answers_path, 'r') as fd:
            answers_json = json.load(fd)
        with open(vocab_path, 'r') as fd:
            vocab_json = json.load(fd)
        self._check_integrity(questions_json, answers_json)

        # vocab
        self.vocab = vocab_json
        self.token_to_index = self.vocab['question']
        self.answer_to_index = self.vocab['answer']

        self._max_length = 14

        # q and a
        self.questions = list(prepare_questions(questions_json))
        self.answers = list(prepare_answers(answers_json))
        self.questions = [self._encode_question(q) for q in self.questions]
        self.answers = [self._encode_answers(a) for a in self.answers] # each answer has the length of all answers
        self.question_ids = [int(q['question_id']) for q in questions_json['questions']]

        # v
        self.image_features_path = image_features_path
        self.coco_id_to_index = self._create_coco_id_to_index()
        self.coco_ids = [q['image_id'] for q in questions_json['questions']]

        # only use questions that have at least one answer, because of the removal of low frequency answers
        self.answerable_only = answerable_only
        if self.answerable_only:
            self.answerable = self._find_answerable()

    @property
    def num_tokens(self):
        return len(self.token_to_index) + 1  # add 1 for <unknown> token at index 0

    def _create_coco_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def _check_integrity(self, questions, answers):
        """ Verify that we are using the correct data """
        qa_pairs = list(zip(questions['questions'], answers['annotations']))
        assert all(q['question_id'] == a['question_id'] for q, a in qa_pairs), 'Questions not aligned with answers'
        assert all(q['image_id'] == a['image_id'] for q, a in qa_pairs), 'Image id of question and answer don\'t match'
        assert questions['data_type'] == answers['data_type'], 'Mismatched data types'
        assert questions['data_subtype'] == answers['data_subtype'], 'Mismatched data subtypes'

    def _find_answerable(self):
        """ Create a list of indices into questions that will have at least one answer that is in the vocab """
        answerable = []
        for i, answers in enumerate(self.answers):
            answer_has_index = len(answers.nonzero()) > 0 # return indices
            # store the indices of anything that is answerable
            if answer_has_index:
                answerable.append(i)
        return answerable

    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.zeros(self._max_length).long()
        for i, token in enumerate(question):
            index = self.token_to_index.get(token, 0)
            vec[i] = index
            if i == self._max_length - 1:
                break
        return vec, i+1

    def _encode_answers(self, answers):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        answer_vec = torch.zeros(len(self.answer_to_index))
        for answer in answers:
            index = self.answer_to_index.get(answer)
            if index is not None:
                answer_vec[index] += 1 # can directly divide 10
        return answer_vec

    def _load_image(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path, 'r')
        index = self.coco_id_to_index[image_id]
        dataset = self.features_file['features']
        img = dataset[index].astype('float32')
        return torch.from_numpy(img)

    def __getitem__(self, item):
        if self.answerable_only:
            # change of indices to only address answerable questions
            item = self.answerable[item]

        q, q_length = self.questions[item]
        q_id = self.question_ids[item]
        a = self.answers[item]
        image_id = self.coco_ids[item]
        v = self._load_image(image_id)
        # since batches are re-ordered for PackedSequence's, the original question order is lost
        # we return `item` so that the order of (v, q, a) triples can be restored if desired
        # without shuffling in the dataloader, these will be in the order that they appear in the q and a json's.
        return v, q, a, q_id, q_length

    def __len__(self):
        if self.answerable_only:
            return len(self.answerable)
        else:
            return len(self.questions)


class VQATest(data.Dataset):
    """ VQA test dataset, open-ended """
    def __init__(self, questions_path, 
                       image_features_path,
                       vocab_path
                ):
        super(VQATest, self).__init__()
        with open(questions_path, 'r') as fd:
            questions_json = json.load(fd)
        with open(vocab_path, 'r') as fd:
            vocab_json = json.load(fd)

        self._max_length = 14
        self.vocab = vocab_json
        self.token_to_index = self.vocab['question']

        self.questions = list(prepare_questions(questions_json))
        self.questions = [self._encode_question(q) for q in self.questions]
        self.question_ids = [int(q['question_id']) for q in questions_json['questions']]

        self.image_features_path = image_features_path
        self.coco_id_to_index = self._create_coco_id_to_index()
        self.coco_ids = [q['image_id'] for q in questions_json['questions']]


    @property
    def num_tokens(self):
        return len(self.token_to_index) + 1  # add 1 for <unknown> token at index 0

    def _create_coco_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.zeros(self._max_length).long()
        for i, token in enumerate(question):
            index = self.token_to_index.get(token, 0)
            vec[i] = index
            if i == self._max_length - 1:
                break
        return vec, i+1

    def _load_image(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path, 'r')
        index = self.coco_id_to_index[image_id]
        dataset = self.features_file['features']
        img = dataset[index].astype('float32')
        return torch.from_numpy(img)

    def __getitem__(self, item):
        q, q_length = self.questions[item]
        q_id = self.question_ids[item]
        image_id = self.coco_ids[item]
        v = self._load_image(image_id)

        return v, q, q_id, q_length

    def __len__(self):
        return len(self.questions)


# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)') # match . when before and after are not numerical (i.e., decimal)
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars))) 
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))

contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                "youll": "you'll", "youre": "you're", "youve": "you've"}
manualMap = { 'none': '0',
                'zero': '0',
                'one': '1',
                'two': '2',
                'three': '3',
                'four': '4',
                'five': '5',
                'six': '6',
                'seven': '7',
                'eight': '8',
                'nine': '9',
                'ten': '10'
            }
articles = ['a', 'an', 'the']


def prepare_questions(questions_json):
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    questions = [q['question'] for q in questions_json['questions']]
    for question in questions:
        question = question.lower()
        question = question.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        yield question.split()


def process_punctuation(s):
    # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
    # this version should be faster since we use re instead of repeated operations on str's
    if _punctuation.search(s) is None:
        return s
    s = _punctuation_with_a_space.sub('', s)
    if re.search(_comma_strip, s) is not None:
        s = s.replace(',', '') # In case of comma, no space is inserted if it occurs between digits
    s = _punctuation.sub(' ', s) # Replacing all punctuation (except apostrophe and colon) with a space character
    s = _period_strip.sub('', s) # Removing periods except if it occurs as decimal
    return s.strip()
    

def processDigitArticle(inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = manualMap.setdefault(word, word)
            if word not in articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in contractions: 
                outText[wordId] = contractions[word]
        outText = ' '.join(outText)
        return outText


def prepare_answers(answers_json):
    """ Normalize answers from a given answer json in the usual VQA format. """
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]
    # The only normalization that is applied to both machine generated answers as well as
    # ground truth answers is replacing most punctuation with space (see [0] and [1]).
    # Since potential machine generated answers are just taken from most common answers, applying the other
    # normalizations is not needed, assuming that the human answers are already normalized.
    # [0]: http://visualqa.org/evaluation.html
    # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96

    for answer_list in answers:
        yield list(map(processDigitArticle, map(process_punctuation, answer_list)))


def prepare_multiple_answers(answers_json):
    """ Normalize answers from a given answer json in the usual VQA format. """
    answers = [ans_dict['multiple_choice_answer'] for ans_dict in answers_json['annotations']]
    
    return list(map(processDigitArticle, map(process_punctuation, answers)))


class CocoImages(data.Dataset):
    """ Dataset for MSCOCO images located in a folder on the filesystem """
    def __init__(self, path, transform=None):
        super(CocoImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform # transform each image

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)


class Composite(data.Dataset):
    """ Dataset that is a composite of several Dataset objects. Useful for combining splits of a dataset. """
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        current = self.datasets[0]
        for d in self.datasets:
            if item < len(d):
                return d[item]
            item -= len(d) # the item should be in the next dataset
        else:
            raise IndexError('Index too large for composite dataset')

    @property
    def num_tokens(self):
        """ Exclusively for embedding questions when using train and val sets for training."""
        return self.datasets[0].num_tokens

    def __len__(self):
        return sum(map(len, self.datasets))
