import os
import sys
import json
sys.path.append(os.getcwd())

from itertools import chain
from collections import Counter

import utils.config as config
import utils.utils as utils
import utils.data as data


def extract_type(train=False, val=False, answer_vocab=None):
	""" Counting answers under each question type. """
	question_type_dict = {}
	answer_idx_dict = {}
	answers = utils.path_for(answer=True, train=train, val=val)
	with open(answers, 'r') as fd:
		answers = json.load(fd)

	bias = len(answer_vocab) # for unseeing answers
	for answer_dict in answers['annotations']:
		question_type = answer_dict['question_type']
		if question_type not in question_type_dict:
			question_type_dict[question_type] = []

		answ_idxs = []
		for answs in answer_dict['answers']:
			answ = data.process_answers(answs['answer'])
			answ_idx = answer_vocab.get(answ, bias)
			if answ_idx not in answer_idx_dict:
				answer_idx_dict[answ_idx] = []
			answer_idx_dict[answ_idx].append(question_type) # answer_idx: question_type
			if not answ_idx == bias:
				answ_idxs.append(answ)
		question_type_dict[question_type].extend(answ_idxs) # question_type: answer_idx (needs counting later)

	# Count the answers for each question_type
	for qt in question_type_dict:
		qt_answs = question_type_dict[qt]
		question_type_dict[qt] = {'answers': Counter(qt_answs),
								'total_answers': len(qt_answs),
								'answer_type_num': len(set(qt_answs))}
	answer_idx_dict = {ai: Counter(answer_idx_dict[ai]) for ai in answer_idx_dict}
	return question_type_dict, answer_idx_dict


def main():
	with open(config.vocabulary_path, 'r') as fd:
		vocab_json = json.load(fd)
	answer_vocab = vocab_json['answer']

	qta, aid = extract_type(train=True, answer_vocab=answer_vocab)
	qa_dict = {
		'qta': dict(sorted(qta.items())),
		'aid': dict(sorted(aid.items()))
	}
	with open(config.question_type_path, 'w') as fd:
		json.dump(qa_dict, fd)


if __name__ == '__main__':
	main()
