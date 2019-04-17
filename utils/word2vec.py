import os, sys, time
import json
import numpy as np

import torch
import torch.nn as nn

import utils.config as config


def load_glove_weights(glove_file):
	f = open(glove_file, 'r')
	weights = {}
	for line in f:
		split = line.split()
		word = split[0]
		embedding = np.array([float(val) for val in split[1:]])
		weights[word] = embedding
	return weights


def filter_embedding(weights, keys):
	""" Filtering the number of embeddings to accelarate. 
		weights: Original loaded word weights dict.
		keys: Index words dict, the keys in this dict should be ranked correctly in ascending order.
	"""
	embedding = []
	dim = len(list(weights.values())[0])
	for (idx, answ) in enumerate(keys):
		embed_k = [weights.get(k, np.zeros(dim, dtype=np.float32)) for k in answ.split()]
		embedding.append(np.asarray(embed_k).mean(axis=0))

	return np.asarray(embedding), len(embedding), dim


def filter_glove_embedding(keys):
	"""
		keys: Index words dict, the keys in this dict should be ranked correctly in ascending order.
	"""
	weights = load_glove_weights(os.path.join(
							config.word_embedding_path, 
							'glove/glove.6B.300d.txt'))
	embedding, num, dim = filter_embedding(weights, keys)
	return embedding, num, dim


class LoadWordEmbedding(nn.Module):
	def __init__(self, weight_matrix, num_embed, embed_dim, need_finetune=False):
		super(LoadWordEmbedding, self).__init__()
		weight_matrix = torch.tensor(weight_matrix)
		self.embedding = nn.Embedding(num_embed, embed_dim)
		self.embedding.load_state_dict({'weight': weight_matrix})
		if not need_finetune:
			self.embedding.weight.require_grad = False

	def forward(self, idx):
		return self.embedding(idx)	