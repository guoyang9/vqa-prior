import os
import torch
import torch.nn as nn
import utils.config as config


def load_glove_weights(glove_file):
	glove_weights = {}
	with open(glove_file, 'r') as fd:
		for line in fd:
			split = line.split()
			word = split[0]
			embedding = torch.tensor([float(val) for val in split[1:]])
			glove_weights[word] = embedding
	return glove_weights


def filter_embedding(weights, keys):
	""" Filtering the number of embeddings to accelarate. 
		weights: Original loaded word weights dict.
		keys: Index words dict, the keys in this dict should be ranked correctly in ascending order.
	"""
	embeddings = []
	dim = len(list(weights.values())[0])

	for answ in keys:
		embed_k = [weights.get(word, torch.zeros(dim)) for word in answ.split()]
		if embed_k:
			embeddings.append(torch.stack(embed_k, dim=0).mean(dim=0))
		else:
			embeddings.append(torch.zeros(dim))

	return torch.stack(embeddings, dim=0), len(embeddings), dim


def filter_glove_embedding(keys):
	"""
		keys: Index words dict, the keys in this dict should be ranked correctly in ascending order.
	"""
	weights = load_glove_weights(os.path.join(
							config.word_embedding_path, 
							'glove/glove.6B.300d.txt'))
	embedding, num, dim = filter_embedding(weights, keys)
	return embedding, num, dim
