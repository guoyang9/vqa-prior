import os
import sys
import json
import math

import utils.data as data
import utils.utils as utils
import utils.config as config


def eval(result, annotations, qta):
	'''
		result: predicted answer file: question_id: answer
		annotations: the validation annotation file
		qta: answer distribution per question type
	'''
	annotation = []
	result = [{'question_id': qa_pair['question_id'], 'answer':
				data.process_answers(qa_pair['answer'])} for qa_pair in result]
	
	for anntt in annotations['annotations']:
		anntt['multiple_choice_answer'] = data.process_answers(
							anntt['multiple_choice_answer'])
		annotation.append(anntt)

	answ_predict = cluster_answer(result, annotation, qta)
	LP = calculate_LP(answ_predict, qta)
	return LP


def cluster_answer(result, annotation, qta):
	''' Clustering all the predicted answers according to the following creterion:
		|--question_type 
		|  |--answer
		|	  |--t: correct answer number.
		|	  |--f: incorrect answer number.
	'''
	answ_predict = {}
	# sort result and annotation so that the question ids are corresponding
	result = sorted(result, key=lambda k: k['question_id'])
	annotation = sorted(annotation, key=lambda k: k['question_id'])
	assert all([rq['question_id']==aq['question_id'] for (rq, aq) in zip(result, annotation)])

	for (res, anntt) in zip(result, annotation):
		ques_res = res['question_id']
		answ_res = res['answer']

		ques_anntt = anntt['question_id']
		answ_anntt = anntt['multiple_choice_answer']
		question_type = anntt['question_type']

		answs_qt = list(qta[question_type]['answers'].keys())

		if question_type == 'none of the above':
			continue
		if question_type not in answ_predict:
			answ_predict[question_type] = {}
		if answ_res not in answs_qt:
			# print('Not found answer {0} in question type {1}.'.format(
			# 										answ_res, question_type))
			continue # hard to quantify, let accuracy determine 

		if answ_res not in answ_predict[question_type]:
			answ_predict[question_type][answ_res] = {'t':0, 'f':0}
		if answ_res == answ_anntt:
			answ_predict[question_type][answ_res]['t'] += 1
		else:
			answ_predict[question_type][answ_res]['f'] += 1

		if not answ_predict[question_type]:
			answ_predict.pop(question_type)
	return answ_predict


def calculate_LP(answ_predict, qta):
	''' Compute language prior (LP rate) according to:
		LPij = (1-Pij) * (nij/Aj).
	'''
	LP = []
	for qt in answ_predict:
		LPj = []
		answs_res = answ_predict[qt]
		answs_qt = qta[qt]['answers']
		Aj = qta[qt]['total_answers']

		for answ in answs_res:
			Pij = answs_res[answ]['t'] / (answs_res[answ]['t'] + answs_res[answ]['f'])
			nij = answs_qt[answ]
			LPij = (1-Pij) * sigmoid(nij/Aj)
			LPj.append(LPij)
		LP.append(sum(LPj)/len(LPj))
	return sum(LP)/len(LP)


def sigmoid(x):
	return 1 / (1+math.exp(-x))


def main():
	results_path = sys.argv[1] # the result file
	annotations_path = utils.path_for(val=True, answer=True)
	qta_path = config.question_type_path

	with open(results_path, 'r') as fd:
		results = json.load(fd)
	with open(annotations_path, 'r') as fd:
		annotations = json.load(fd)
	with open(qta_path, 'r') as fd:
		qta = json.load(fd)['qta']

	lp_best = 1.0
	for k in results:
		lp = eval(results[k], annotations, qta)
		if lp < lp_best:
			lp_best = lp
	print("The best LP score is {:.4f}".format(lp_best))


if __name__ == '__main__':
	main()
