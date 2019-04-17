import sys
import os.path
import math, argparse
import json
from datetime import datetime
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import utils.config as config
import utils.data as data
import model.model as model
import utils.utils as utils
import utils.word2vec as word2vec
from test import test


parser = argparse.ArgumentParser()
parser.add_argument('--need_test', default=False, action='store_true',
					help='decide whether to test or not, i.e., train on train set or on train+val sets (choose this argument means True)')
parser.add_argument('--test_times', default=3, type=int,
					help='test the model on the last test_times epoches')
parser.add_argument('--version', default='v1',
					choices=['v1', 'v2'],
					help='dataset version type (v1 | v2)')
parser.add_argument('--lambd', default=1,
					type=float,
					help='trade-off hyperparameters between two types of losses.')


def update_learning_rate(optimizer, iteration):
	lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


total_iterations = 0


def run(net, loader, optimizer, tracker, train=False, prefix='', embedding=None, epoch=0, answer_idx_vocab=None):
	""" Run an epoch over the given loader """
	if train:
		net.train()
		tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
	else:
		net.eval()
		tracker_class, tracker_params = tracker.MeanMonitor, {}
		answ = []
		accs = []
		results_epoch = []
        answer_idxs = []
        question_ids = []

	embedding = embedding.cuda()
	tq = tqdm(loader, desc='{} Epoch {:03d}'.format(prefix, epoch), ncols=0)

	loss_qv_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
	loss_q_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
	loss_score_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
	acc_qv_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
	acc_q_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

	log_softmax = nn.LogSoftmax(dim=-1).cuda() 
	for v, q, a, q_len in tq:
		v = v.cuda(async=True)
		q = q.cuda(async=True)
		a = a.cuda(async=True)
		q_len = q_len.cuda(async=True)
		embed_a = (embedding(a.long()) /10).sum(dim=1)

		answer_vq, answer_q, score_loss = net(v, q, q_len, embed_a) 
		nll_qv = -log_softmax(answer_vq)
		nll_q = -log_softmax(answer_q)
		loss_qv = (nll_qv * a / 10).sum(dim=1).mean() # only for correct answers
		loss_q = (nll_q * a / 10).sum(dim=1).mean() # only for correct answers
		acc_qv = utils.batch_accuracy(answer_vq, a).cpu()
		acc_q = utils.batch_accuracy(answer_q, a).cpu()

		if train:
			global total_iterations
			# update_learning_rate(optimizer, total_iterations)

			optimizer.zero_grad()
			loss = loss_qv + loss_q + args.lambd * score_loss
			# loss = args.lambd*(loss_qv + loss_q) + score_loss
			loss.backward()
			optimizer.step()

			total_iterations += 1
		else:
			# store information about evaluation of this minibatch
			_, answer = answer_vq.cpu().max(dim=1)
			answ.append(answer.view(-1))
			accs.append(acc_qv.view(-1))
			answer_idxs.extend(answer.view(-1).numpy())
            question_ids.extend(q_id.view(-1).numpy())

		loss_qv_tracker.append(loss_qv.item())
		loss_q_tracker.append(loss_q.item())
		loss_score_tracker.append(score_loss.item())
		acc_qv_tracker.append(acc_qv.mean())
		acc_q_tracker.append(acc_q.mean())

		fmt = '{:.4f}'.format
		tq.set_postfix(loss_qv=fmt(loss_qv_tracker.mean.value),
						loss_q=fmt(loss_q_tracker.mean.value), 
						loss_score=fmt(loss_score_tracker.mean.value), 
						acc_qv=fmt(acc_qv_tracker.mean.value), 
						acc_q=fmt(acc_q_tracker.mean.value))

	if not train:
		answ = list(torch.cat(answ, dim=0))
		accs = list(torch.cat(accs, dim=0))

		answers = [answer_idx_vocab[a] for a in answer_idxs] # true answers

        for q, a in zip(question_ids, answers):
            results_epoch.append({'question_id': int(q), 'answer': a})

        return answ, accs, results_epoch


def main():
	global args
	args = parser.parse_args()
		
	name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
	target_name = os.path.join('logs', '{}.pth'.format(name))
	print('will save to {}'.format(target_name))

	os.environ["CUDA_VISIBLE_DEVICES"] = config.opt_gpu
	cudnn.benchmark = True

	if not args.need_test:
		train_loader = data.get_loader(train=True, version=args.version)
		val_loader = data.get_loader(val=True, version=args.version)
	else:
		# train loader includes both train and val sets
		train_loader = data.get_loader(train=True, need_test=True, version=args.version) 
		test_loader = data.get_loader(test=True, need_test=True, version=args.version)

	# extract the answer idx with true answer, for testing prediction
	answer_idx_path = getattr(config, 'vocabulary_path_{}'.format(args.version))
	with open(answer_idx_path, 'r') as fd:
		vocab_json = json.load(fd)
	answer_idx_vocab = vocab_json['answer_idx']
	answer_idx_vocab = {int(a_idx): a for a_idx, a in answer_idx_vocab.items()}

	# load pre-trained word embedding (glove) for embedding answers
	answer_vocab = vocab_json['answer']
	embedding, num, dim = word2vec.filter_glove_embedding(answer_vocab)
	embedding = word2vec.LoadWordEmbedding(embedding, num, dim)

	vqa_model = model.Net(train_loader.dataset.num_tokens)
	net = nn.DataParallel(vqa_model).cuda()
	optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
	# optimizer = optim.Adam([
	# 	{'params': chain(net.module.text.parameters(),
	# 					net.module.attention.parameters(),
	# 					net.module.transform_qv.parameters(),
	# 					net.module.transform_q.parameters(),
	# 					net.module.classifier.parameters())
	# 	},
	# 	{'params': net.module.score.parameters(), 'lr': 1e-5}],
	# 	lr=1e-3)

	tracker = utils.Tracker()
	config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
	result_dict = {}

	for i in range(config.epochs):
		_ = run(net, train_loader, optimizer, tracker, train=True, prefix='train', embedding=embedding, epoch=i)
		if not args.need_test:
			r = run(net, val_loader, optimizer, tracker, train=False, prefix='val', embedding=embedding, epoch=i, answer_idx_vocab=answer_idx_vocab)
            result_dict[i] = results_epoch
		else:
			if i in range(config.epochs-args.test_times, config.epochs):
				results_test = test(net, test_loader, i, answer_idx_vocab)

				with open(os.path.join(
					getattr(config, 'result_{}'.format(args.version)),
					'vqa_{0}_{1}_test-dev2015_{2}_results.json'.format(
						config.task,
						config.dataset,
						i)), 'w') as fd:
					json.dump(results_test, fd)

		# results = {
		# 	'name': name,
		# 	'tracker': tracker.to_dict(),
		# 	'config': config_as_dict,
		# 	'weights': net.state_dict(),
		# }
		# torch.save(results, target_name)
	with open('./results.json', 'w') as fd:
        json.dump(result_dict, fd)


if __name__ == '__main__':
	main()