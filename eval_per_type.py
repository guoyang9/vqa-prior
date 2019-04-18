import os, sys
import utils.utils as utils

from vqa_eval.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
from vqa_eval.PythonHelperTools.vqaTools.vqa import VQA


quesFile = utils.path_for(val=True, question=True)
annFile = utils.path_for(val=True, answer=True)
resFile = sys.argv[1] 

# create vqa object and vqaRes object
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

# create vqaEval object by taking vqa and vqaRes
vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

# evaluate results
vqaEval.evaluate() 

# print accuracies
print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
print("Per Question Type Accuracy is the following:")
for quesType in vqaEval.accuracy['perQuestionType']:
	print("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
print("\n")
print("Per Answer Type Accuracy is the following:")
for ansType in vqaEval.accuracy['perAnswerType']:
	print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))

