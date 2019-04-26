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
		answ, accs, answer_idxs, question_ids, results_epoch = ([] for _ in range(5))	

	embedding = embedding.cuda()
	loader = tqdm(loader, desc='{} Epoch {:03d}'.format(prefix, epoch), ncols=0)

	loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
	acc_qv_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
	acc_q_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

	for v, q, a, q_id, q_len in loader:
		v = v.cuda(async=True)
		q = q.cuda(async=True)
		a = a.cuda(async=True)
		q_len = q_len.cuda(async=True)
		embed_a = (embedding(a.long()) /10).sum(dim=1)

		answer_vq, answer_q, score_loss = net(v, q, q_len, embed_a) 
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
		return answ, accs, results_epoch


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
				'acc_val_best': acc_val_best,
				'name': name,
				'model_state': net.state_dict(),
				'optim_state': optimizer.state_dict(),
				'scheduler_state': scheduler.state_dict(),
				'eval': {
					'answers': r[0],
					'accuracies': r[1],
				},
			}
			result_dict[i] = r[-1]

			if config.train_set == 'train' and r[1].mean() > acc_val_best:
				acc_val_best = r[1].mean()
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
