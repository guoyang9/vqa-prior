import json, sys, os
sys.path.append(os.getcwd())

import argparse
from collections import Counter
from collections import defaultdict
from itertools import chain

import utils.config as config
import utils.utils as utils
import utils.data as data


parser = argparse.ArgumentParser()
parser.add_argument('--train_set', default='train',
					choices=['train', 'train+val'],
					help='training set (train | train+val)')
parser.add_argument('--version', default='v1',
					choices=['v1', 'v2'],
					help='dataset version type (v1 | v2)')


def extract_answer_question_type(train=False, val=False, answer_vocab=None):
	question_type_dict = {}
	answer_idx_dict = {}
	answers = utils.path_for(answer=True,
							 train=train,
							 val=val,
							 version=args.version)
	with open(answers, 'r') as fd:
		answers = json.load(fd)

	bias = len(answer_vocab) # for unseeing answers
	for answer_dict in answers['annotations']:
		question_type = answer_dict['question_type']
		if question_type not in question_type_dict:
			question_type_dict[question_type] = []

		answ_idxs = []
		for answs in answer_dict['answers']:
			answ = answs['answer']
			answ = data.processDigitArticle(data.process_punctuation(answ))
			answ_idx = answer_vocab.get(answ, bias)
			if answ_idx not in answer_idx_dict:
				answer_idx_dict[answ_idx] = []
			answer_idx_dict[answ_idx].append(question_type) # answer_idx: question_type
			if not answ_idx == 3000:
				answ_idxs.append(answ)
			# answ_idxs.append(answ_idx)
		question_type_dict[question_type].extend(answ_idxs) # question_type: answer_idx (needs counting later)

	# Count the answers for each question_type
	for qt in question_type_dict:
		qt_answs = question_type_dict[qt]
		question_type_dict[qt] = {'answers': Counter(qt_answs),
								'total_answers': len(qt_answs),
								'answer_type_num': len(set(qt_answs))}
	answer_idx_dict = {ai: Counter(answer_idx_dict[ai]) for ai in answer_idx_dict}
	return question_type_dict, answer_idx_dict


def merge_dict(dict1, dict2):
	""" For merging train and val datasets dict together. """
	dict_merge = defaultdict(list)
	for k, v in chain(dict1.items(), dict2.items()):
		dict_merge[k].append(v)
	dict_merge = {k: v[0] if len(v)==1 else v[0]+v[1] for k, v in dict_merge.items()}

	return dict_merge


def main():
	global args
	args = parser.parse_args()

	answer_path = getattr(config, 'vocabulary_path_{}'.format(args.version))
	with open(answer_path, 'r') as fd:
		vocab_json = json.load(fd)
	answer_vocab = vocab_json['answer']

	if args.train_set == 'train':
		qta, aid = extract_answer_question_type(
										train=True, 
										answer_vocab=answer_vocab)
	else:
		qtd_train, aid_train = extract_answer_question_type(
										train=True, 
										answer_vocab=answer_vocab)
		qtd_val, aid_val = extract_answer_question_type(
										val=True, 
										answer_vocab=answer_vocab)

	qta = merge_dict(qtd_train, qtd_val)
	aid = merge_dict(aid_train, aid_val)

	qta = dict(sorted(qta.items()))
	aid = dict(sorted(aid.items()))
	qa_dict = {
		'qta': qta, # remember to remove the unseeing answer (len(answer_vocab))!
		'aid': aid
	}

	with open(getattr(config, 'question_type_answer_{}'.format(args.version)), 'w') as fd:
		json.dump(qa_dict, fd)


if __name__ == '__main__':
	main()