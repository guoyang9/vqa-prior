import json, sys, os
sys.path.append(os.getcwd())

from collections import Counter
import itertools
import argparse

import utils.config as config
import utils.data as data
import utils.utils as utils


parser = argparse.ArgumentParser()
parser.add_argument('--train_set', default='train',
                    choices=['train', 'train+val'],
                    help='training set (train | train+val)')
parser.add_argument('--version', default='v1',
                    choices=['v1', 'v2'],
                    help='dataset version type (v1 | v2)')


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = iterable if top_k else itertools.chain.from_iterable(iterable) 
    counter = Counter(all_tokens)
    del counter['']
    if top_k:
        most_common = counter.most_common(top_k)

        test = [i[1] for i in most_common]
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


def extract_question(train=False, val=False):
    questions = utils.path_for(question=True, 
                               train=train,
                               val=val,
                               version=args.version)
    with open(questions, 'r') as fd:
        questions = json.load(fd)
    questions = data.prepare_questions(questions)

    return questions


def extract_answer(train=False, val=False):
    answers = utils.path_for(answer=True,
                             train=train,
                             val=val,
                             version=args.version)
    with open(answers, 'r') as fd:
        answers = json.load(fd)
    answers = data.prepare_multiple_answers(answers)

    return answers


def main():
    global args
    args = parser.parse_args()

    if args.train_set == 'train':
        questions = extract_question(train=True)
        answers = extract_answer(train=True)
    else: # train+val
        questions_train = extract_question(train=True)
        questions_val = extract_question(val=True)
        questions = itertools.chain(questions_train, questions_val)

        answers_train = extract_answer(train=True)
        answers_val = extract_answer(val=True)
        answers = answers_train + answers_val
  
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
    with open(getattr(config, 'vocabulary_path_{}'.format(args.version)), 'w') as fd:
        json.dump(vocabs, fd)


if __name__ == '__main__':
    main()