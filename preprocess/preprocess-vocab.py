import os
import sys
import json
sys.path.append(os.getcwd())

import itertools
from collections import Counter

import utils.config as config
import utils.data as data
import utils.utils as utils


def _get_file_(train=False, val=False, test=False, question=False, answer=False):
    """ Get the correct question or answer file."""
    _file = utils.path_for(train=train, val=val, test=test, 
                            question=question, answer=answer)
    with open(_file, 'r') as fd:
        _object = json.load(fd)
    return _object


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = iterable if top_k else itertools.chain.from_iterable(iterable) 
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


def main():
    questions = _get_file_(train=True, question=True)
    answers = _get_file_(train=True, answer=True)

    questions = list(data.prepare_questions(questions))
    answers = list(data.prepare_mul_answers(answers))

    if config.train_set == 'train+val':
        questions_val = _get_file_(val=True, question=True)
        answers_val = _get_file_(val=True, answer=True)

        questions_val = list(data.prepare_questions(questions_val))
        answers_val = list(data.prepare_mul_answers(answers_val))

        questions = itertools.chain(questions, questions_val)
        answers = answers + answers_val
  
    question_vocab = extract_vocab(questions, start=1) # leave 0 for non-found words
    answer_vocab = extract_vocab(answers, top_k=config.max_answers)
    question_idx_vocab = {question_vocab[k]: k for k in question_vocab}
    answer_idx_vocab = {answer_vocab[k]: k for k in answer_vocab}

    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
        'question_idx': question_idx_vocab,
        'answer_idx': answer_idx_vocab
    }
    with open(config.vocabulary_path, 'w') as fd:
        json.dump(vocabs, fd)


if __name__ == '__main__':
    main()
