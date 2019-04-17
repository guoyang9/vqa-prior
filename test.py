import sys
import os.path
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import utils.config as config
import utils.data as data
import model.model as model
import utils.utils as utils

def test(net, loader, epoch, answer_idx_vocab):
	""" Test the model on test_standard and test_develop datasets. """
	net.eval()

	question_ids = [] # the current question id
	answer_idxs = [] # the answer id, which will be useful for retrieving answer
	results = []

	tq = tqdm(loader, desc='Test after training {:03d} epoches'.format(epoch), ncols=0)

	for v, q, q_id, q_len in tq:
		v = v.cuda(async=True)
		q = q.cuda(async=True)
		q_len = q_len.cuda(async=True)

		out = net(v, q, q_len) # out and a are the same size
		_, a_idx = out.cpu().max(dim=1)
		answer_idxs.extend(a_idx.view(-1).numpy())
		question_ids.extend(q_id.view(-1).numpy())

	answers = [answer_idx_vocab[a] for a in answer_idxs] # true answers

	for q, a in zip(question_ids, answers):
		results.append({'question_id': int(q), 'answer': a})

	return results