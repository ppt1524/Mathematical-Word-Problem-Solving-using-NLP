from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


###########################################
####### start - code ######################
###########################################

import numpy as np
import os
import sys
import pdb
import torch.nn as nn
import torch
import math
import collections
import logging
import json
import pandas as pd

from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from sympy import Eq, solve
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp

from glob import glob
from torch.autograd import Variable

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import argparse

import re
from torch.utils.data import Dataset
import unicodedata
from collections import OrderedDict

import random
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
try:
	import cPickle as pickle
except ImportError:
	import pickle

    
from time import time
from torch import optim
import torch.nn.functional as F
from transformers import AdamW
# from pytorch_pretrained_bert.optimization import BertAdam
# from tensorboardX import SummaryWriter
from gensim import models


########################################################
# contextual_embeddings.py #
########################################################

class BertEncoder(nn.Module):
	def __init__(self, bert_model = 'bert-base-uncased',device = 'cuda:0 ', freeze_bert = False):
		super(BertEncoder, self).__init__()
		self.bert_layer = BertModel.from_pretrained(bert_model)
		self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
		self.device = device
		
		if freeze_bert:
			for p in self.bert_layer.parameters():
				p.requires_grad = False
		
	def bertify_input(self, sentences):
		'''
		Preprocess the input sentences using bert tokenizer and converts them to a torch tensor containing token ids
		
		Args:
			sentences (list): source sentences
		Returns:
			token_ids (tensor): tokenized sentences | size: [BS x S]
			attn_masks (tensor): masks padded indices | size: [BS x S]
			input_lengths (list): lengths of sentences | size: [BS]
		'''

		# Tokenize the input sentences for feeding into BERT
		all_tokens  = [['[CLS]'] + self.bert_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]
		
		# Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		# Convert tokens to token ids
		token_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		# Obtain attention masks
		pad_token = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
		
		Args:
			sentences (list): source sentences
		Returns:
			cont_reps (tensor): BERT Embeddings | size: [BS x S x d_model]
			token_ids (tensor): tokenized sentences | size: [BS x S]
		'''

		# Preprocess sentences
		token_ids, attn_masks, input_lengths = self.bertify_input(sentences)

		# Feed through bert
		cont_reps, _ = self.bert_layer(token_ids, attention_mask = attn_masks)

		return cont_reps, token_ids

class RobertaEncoder(nn.Module):
	def __init__(self, roberta_model = 'roberta-base', device = 'cuda:0 ', freeze_roberta = False):
		super(RobertaEncoder, self).__init__()
		self.roberta_layer = RobertaModel.from_pretrained(roberta_model)
		self.roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
		self.device = device
		
		if freeze_roberta:
			for p in self.roberta_layer.parameters():
				p.requires_grad = False
		
	def robertify_input(self, sentences):
		'''
		Preprocess the input sentences using roberta tokenizer and converts them to a torch tensor containing token ids
		
		Args:
			sentences (list): source sentences
		Returns:
			token_ids (tensor): tokenized sentences | size: [BS x S]
			attn_masks (tensor): masks padded indices | size: [BS x S]
			input_lengths (list): lengths of sentences | size: [BS]
		'''

		# Tokenize the input sentences for feeding into RoBERTa
		all_tokens  = [['<s>'] + self.roberta_tokenizer.tokenize(sentence) + ['</s>'] for sentence in sentences]
		
		# Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['<pad>' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		# Convert tokens to token ids
		token_ids = torch.tensor([self.roberta_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		# Obtain attention masks
		pad_token = self.roberta_tokenizer.convert_tokens_to_ids('<pad>')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a RoBERTa encoder to obtain contextualized representations of each token
		'''
		# Preprocess sentences
		token_ids, attn_masks, input_lengths = self.robertify_input(sentences)

		# Feed through RoBERTa
		output = self.roberta_layer(token_ids, attention_mask = attn_masks)
        
		cont_reps = output.last_hidden_state

		return cont_reps, token_ids
	
########################################################
# bleu.py #
########################################################

def _get_ngrams(segment, max_order):
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
 
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                           (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                             possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        if ratio > 1E-1:
            bp = math.exp(1 - 1. / ratio)
        else:
            bp = 1E-2

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)

########################################################
# evaluate.py #
########################################################

def format_eq(eq):
	fin_eq = ""
	ls = ['0','1','2','3','4','5','6','7','8','9','.']
	temp_num = ""
	flag = 0
	for i in eq:
		if flag > 0:
			fin_eq = fin_eq + i
			flag = flag-1
		elif i == 'n':
			flag = 6
			if fin_eq == "":
				fin_eq = fin_eq + i
			else:
				fin_eq = fin_eq + ' ' + i
		elif i in ls:
			temp_num = temp_num + i
		elif i == ' ':
			if temp_num == "":
				continue
			else:
				if fin_eq == "":
					fin_eq = fin_eq + temp_num
				else:
					fin_eq = fin_eq + ' ' + temp_num
			temp_num = ""
		else:
			if fin_eq == "":
				if temp_num == "":
					fin_eq = fin_eq + i
				else:
					fin_eq = fin_eq + temp_num + ' ' + i
			else:
				if temp_num == "":
					fin_eq = fin_eq + ' ' + i
				else:
					fin_eq = fin_eq + ' ' + temp_num + ' ' + i
			temp_num = ""
	if temp_num != "":
		fin_eq = fin_eq + ' ' + temp_num
	return fin_eq

def prefix_to_infix(prefix):
	operators = ['+', '-', '*', '/']
	stack = []
	elements = format_eq(prefix).split()
	for i in range(len(elements)-1, -1, -1):
		if elements[i] in operators and len(stack)>1:
			op1 = stack.pop(-1)
			op2 = stack.pop(-1)
			fin_operand = '(' + ' ' + op1 + ' ' + elements[i] + ' ' + op2 + ' ' + ')'
			stack.append(fin_operand)
		else:
			stack.append(elements[i])
	try:
		return stack[0]
	except:
		return ''

def stack_to_string(stack):
	op = ""
	for i in stack:
		if op == "":
			op = op + i
		else:
			op = op + ' ' + i
	return op

def back_align(eq, list_num):
	elements = eq.split()
	for i in range(len(elements)):
		if elements[i][0] == 'n':
			index = int(elements[i][6])
			try:
				number = str(list_num[index])
			except:
				return '-1000.112'
			elements[i] = number
	return stack_to_string(elements)    

def ans_evaluator(eq, list_num):
	#pdb.set_trace()
	infix = prefix_to_infix(eq)
	aligned = back_align(infix, list_num)
	try:
		final_ans = parse_expr(aligned, evaluate = True)
	except:
		final_ans = -1000.112
	return final_ans

def cal_score(outputs, num):
	for i in range(len(outputs)):
		op = stack_to_string(outputs[i])

		pred = ans_evaluator(op, num)

	return pred


def get_infix_eq(outputs, nums):
	eqs = []
	for i in range(len(outputs)):
		op = stack_to_string(outputs[i])
		num = nums[i].split()
		num = [float(nu) for nu in num]

		infix = prefix_to_infix(op)
		eqs.append(infix)

	return eqs


########################################################
# helper.py #
########################################################

def gpu_init_pytorch(gpu_num):
	'''
		Initialize GPU

		Args:
			gpu_num (int): Which GPU to use
		Returns:
			device (torch.device): GPU device
	'''

	torch.cuda.set_device(int(gpu_num))
	device = torch.device("cuda:{}".format(
		gpu_num) if torch.cuda.is_available() else "cpu")
	return device

def create_save_directories(path):
	if not os.path.exists(path):
		os.makedirs(path)

def save_checkpoint(state, epoch, logger, model_path, ckpt):
	'''
		Saves the model state along with epoch number. The name format is important for 
		the load functions. Don't mess with it.

		Args:
			state (dict): model state
			epoch (int): current epoch
			logger (logger): logger variable to log messages
			model_path (string): directory to save models
			ckpt (string): checkpoint name
	'''

	ckpt_path = os.path.join(model_path, '{}_{}.pt'.format(ckpt, epoch))
	# logger.info('Saving Checkpoint at : {}'.format(ckpt_path))
	torch.save(state, ckpt_path)

def get_latest_checkpoint(model_path, logger):
	'''
		Looks for the checkpoint with highest epoch number in the directory "model_path" 

		Args:
			model_path (string): directory where model is saved
			logger (logger): logger variable to log messages
		Returns:
			ckpt_path: checkpoint path to the latest checkpoint 
	'''

	ckpts = glob('{}/*.pt'.format(model_path))
	ckpts = sorted(ckpts)

	if len(ckpts) == 0:
		# logger.warning('No Checkpoints Found')

		return None
	else:
		latest_epoch = max([int(x.split('_')[-1].split('.')[0]) for x in ckpts])
		ckpts = sorted(ckpts, key= lambda x: int(x.split('_')[-1].split('.')[0]) , reverse=True )
		ckpt_path = ckpts[0]
		# logger.info('Checkpoint found with epoch number : {}'.format(latest_epoch))
		# logger.debug('Checkpoint found at : {}'.format(ckpt_path))

		return ckpt_path

def load_checkpoint(model, mode, ckpt_path, logger, device):
	'''
		Load the model at checkpoint

		Args:
			model (object of class TransformerModel): model
			mode (string): train or test mode
			ckpt_path: checkpoint path to the latest checkpoint 
			logger (logger): logger variable to log messages
			device (torch.device): GPU device
		Returns:
			start_epoch (int): epoch from which to start
			min_train_loss (float): minimum train loss
			min_val_loss (float): minimum validation loss
			max_train_acc (float): maximum train accuracy
			max_val_acc (float): maximum validation accuracy score
			max_val_bleu (float): maximum valiadtion bleu score
			best_epoch (int): epoch with highest validation accuracy
			voc1 (object of class Voc1): vocabulary of source
			voc2 (object of class Voc2): vocabulary of target
	'''

	checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	start_epoch = checkpoint['epoch']
	min_train_loss  =checkpoint['min_train_loss']
	min_val_loss = checkpoint['min_val_loss']
	voc1 = checkpoint['voc1']
	voc2 = checkpoint['voc2']
	max_train_acc = checkpoint['max_train_acc']
	max_val_acc = checkpoint['max_val_acc']
	max_val_bleu = checkpoint['max_val_bleu']
	best_epoch = checkpoint['best_epoch']

	model.to(device)

	if mode == 'train':
		model.train()
	else:
		model.eval()

	# logger.info('Successfully Loaded Checkpoint from {}, with epoch number: {} for {}'.format(ckpt_path, start_epoch, mode))

	return start_epoch, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2

class Voc1:
	def __init__(self):
		self.trimmed = False
		self.frequented = False
		self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2}
		self.id2w = {0: '<s>', 1: '</s>', 2: 'unk'}
		self.w2c = {}
		self.nwords = 3

	def add_word(self, word):
		if word not in self.w2id:
			self.w2id[word] = self.nwords
			self.id2w[self.nwords] = word
			self.w2c[word] = 1
			self.nwords += 1
		else:
			self.w2c[word] += 1

	def add_sent(self, sent):
		for word in sent.split():
			self.add_word(word)

	def most_frequent(self, topk):
		# if self.frequented == True:
		# 	return
		# self.frequented = True

		keep_words = []
		count = 3
		sort_by_value = sorted(
			self.w2c.items(), key=lambda kv: kv[1], reverse=True)
		for word, freq in sort_by_value:
			keep_words += [word]*freq
			count += 1
			if count == topk:
				break

		self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2}
		self.id2w = {0: '<s>', 1: '</s>', 2: 'unk'}
		self.w2c = {}
		self.nwords = 3

		for word in keep_words:
			self.add_word(word)

	def trim(self, mincount):
		if self.trimmed == True:
			return
		self.trimmed = True

		keep_words = []
		for k, v in self.w2c.items():
			if v >= mincount:
				keep_words += [k]*v

		self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2}
		self.id2w = {0: '<s>', 1: '</s>', 2: 'unk'}
		self.w2c = {}
		self.nwords = 3
		for word in keep_words:
			self.addWord(word)

	def get_id(self, idx):
		return self.w2id[idx]

	def get_word(self, idx):
		return self.id2w[idx]

	def create_vocab_dict(self, args, train_dataloader):
		for data in train_dataloader:
			for sent in data['ques']:
				self.add_sent(sent)

		self.most_frequent(args.vocab_size)
		assert len(self.w2id) == self.nwords
		assert len(self.id2w) == self.nwords

	def add_to_vocab_dict(self, args, dataloader):
		for data in dataloader:
			for sent in data['ques']:
				self.add_sent(sent)

		self.most_frequent(args.vocab_size)
		assert len(self.w2id) == self.nwords
		assert len(self.id2w) == self.nwords

class Voc2:
	def __init__(self, config):
		self.frequented = False
		if config.mawps_vocab:
			# '0.25', '8.0', '0.05', '60.0', '7.0', '5.0', '2.0', '4.0', '1.0', '12.0', '100.0', '25.0', '0.1', '3.0', '0.01', '0.5', '10.0'
			self.w2id = {'<s>': 11, '</s>': 1, '+': 2, '-': 3, '*': 4, '/': 5, 'number0': 6, 'number1': 7, 'number2': 8, 'number3': 9, 'number4': 10, '0.25': 0, '8.0': 12, '0.05': 13, '60.0': 14, '7.0': 15, '5.0': 16, '2.0': 17, '4.0': 18, '1.0': 19, '12.0': 20, '100.0': 21, '25.0': 22, '0.1': 23, '3.0': 24, '0.01': 25, '0.5': 26, '10.0': 27, 'unk': 28}
			self.id2w = {11: '<s>', 1: '</s>', 2: '+', 3: '-', 4: '*', 5: '/', 6: 'number0', 7: 'number1', 8: 'number2', 9: 'number3', 10: 'number4', 0: '0.25', 12: '8.0', 13: '0.05', 14: '60.0', 15: '7.0', 16: '5.0', 17: '2.0', 18: '4.0', 19: '1.0', 20: '12.0', 21: '100.0', 22: '25.0', 23: '0.1', 24: '3.0', 25: '0.01', 26: '0.5', 27: '10.0', 28: 'unk'}
			self.w2c = {'+': 0, '-': 0, '*': 0, '/': 0, 'number0': 0, 'number1': 0, 'number2': 0, 'number3': 0, 'number4': 0, '0.25': 0, '8.0': 0, '0.05': 0, '60.0': 0, '7.0': 0, '5.0': 0, '2.0': 0, '4.0': 0, '1.0': 0, '12.0': 0, '100.0': 0, '25.0': 0, '0.1': 0, '3.0': 0, '0.01': 0, '0.5': 0, '10.0': 0, 'unk': 0}
			self.nwords = 29
		else:
			self.w2id = {'<s>': 11, '</s>': 1, '+': 2, '-': 3, '*': 4, '/': 5, 'number0': 6, 'number1': 7, 'number2': 8, 'number3': 9, 'number4': 10, 'number5': 0, 'unk': 12}
			self.id2w = {11: '<s>', 1: '</s>', 2: '+', 3: '-', 4: '*', 5: '/', 6: 'number0', 7: 'number1', 8: 'number2', 9: 'number3', 10: 'number4', 0: 'number5', 12: 'unk'}
			self.w2c = {'+': 0, '-': 0, '*': 0, '/': 0, 'number0': 0, 'number1': 0, 'number2': 0, 'number3': 0, 'number4': 0, 'number5': 0, 'unk': 0}
			self.nwords = 13 # For some reason, model outputs NANs if I keep <s> as token 0 - That's because it was masking out the <s> token in make_len_mask in model

	def add_word(self, word):
		if word not in self.w2id: # IT SHOULD NEVER GO HERE!!
			self.w2id[word] = self.nwords
			self.id2w[self.nwords] = word
			self.w2c[word] = 1
			self.nwords += 1
		else:
			self.w2c[word] += 1

	def add_sent(self, sent):
		for word in sent.split():
			self.add_word(word)

	def get_id(self, idx):
		return self.w2id[idx]

	def get_word(self, idx):
		return self.id2w[idx]

	def create_vocab_dict(self, args, train_dataloader):
		for data in train_dataloader:
			for sent in data['eqn']:
				self.add_sent(sent)

		assert len(self.w2id) == self.nwords
		assert len(self.id2w) == self.nwords

	def add_to_vocab_dict(self, args, dataloader):
		for data in dataloader:
			for sent in data['eqn']:
				self.add_sent(sent)

		assert len(self.w2id) == self.nwords
		assert len(self.id2w) == self.nwords
        
def bleu_scorer(ref, hyp, script='default'):
	'''
		Bleu Scorer (Send list of list of references, and a list of hypothesis)
	'''
	
	refsend = []
	for i in range(len(ref)):
		refsi = []
		for j in range(len(ref[i])):
			refsi.append(ref[i][j].split())
		refsend.append(refsi)

	gensend = []
	for i in range(len(hyp)):
		gensend.append(hyp[i].split())

	if script == 'nltk':
		metrics = corpus_bleu(refsend, gensend)
		return [metrics]

	metrics = compute_bleu(refsend, gensend)
	return metrics

########################################################
# logger.py #
########################################################

# def get_logger(name, log_file_path='./logs/temp.log', logging_level=logging.INFO, 
# 				log_format='%(asctime)s | %(levelname)s | %(filename)s: %(lineno)s : %(funcName)s() ::\t %(message)s'):
# 	# logger = logging.getLogger(name)
# 	# logger.setLevel(logging_level)
# 	formatter = logging.Formatter(log_format)

# 	file_handler = logging.FileHandler(log_file_path, mode='w') # Sends logging output to a disk file
# 	file_handler.setLevel(logging_level)
# 	file_handler.setFormatter(formatter)

# 	stream_handler = logging.StreamHandler() # Sends logging output to stdout
# 	stream_handler.setLevel(logging_level)
# 	stream_handler.setFormatter(formatter)

# 	# logger.addHandler(file_handler)
# 	# logger.addHandler(stream_handler)

# 	# logger.addFilter(ContextFilter(expt_name))

# 	return logger


# def print_log(logger, dict):
# 	string = ''
# 	for key, value in dict.items():
# 		string += '\n {}: {}\t'.format(key.replace('_', ' '), value)
	# string = string.strip()
	# logger.info(string)



def store_results(config, max_val_bleu, max_val_acc, min_val_loss, max_train_acc, min_train_loss, best_epoch):
	try:
		with open(config.result_path) as f:
			res_data =json.load(f)
	except:
		res_data = {}
	try:
		min_train_loss = min_train_loss.item()
	except:
		pass
	try:
		min_val_loss = min_val_loss.item()
	except:
		pass
	try:

		data= {'run_name' : str(config.run_name)
		, 'max val acc': str(max_val_acc)
		, 'max train acc': str(max_train_acc)
		, 'max val bleu' : str(max_val_bleu)
		, 'min val loss' : str(min_val_loss)
		, 'min train loss': str(min_train_loss)
		, 'best epoch': str(best_epoch)
		, 'epochs' : config.epochs
		, 'dataset' : config.dataset
		, 'embedding': config.embedding
		, 'embedding_lr': config.emb_lr
		, 'freeze_emb': config.freeze_emb
		, 'i/p and o/p embedding size' : config.d_model
		, 'encoder_layers' : config.encoder_layers
		, 'decoder_layers' : config.decoder_layers
		, 'heads' : config.heads
		, 'FFN size' : config.d_ff
		, 'lr' : config.lr
		, 'batch_size' : config.batch_size
		, 'dropout' : config.dropout
		, 'opt' : config.opt
		}
		res_data[str(config.run_name)] = data

		with open(config.result_path, 'w', encoding='utf-8') as f:
			json.dump(res_data, f, ensure_ascii= False, indent= 4)
	except:
		pdb.set_trace()

def store_val_results(config, acc_score, folds_scores):
	try:
		with open(config.val_result_path) as f:
			res_data = json.load(f)
	except:
		res_data = {}
	try:
		data= {'run_name' : str(config.run_name)
		, '5-fold avg acc score' : str(acc_score)
		, 'Fold0 acc' : folds_scores[0]
		, 'Fold1 acc' : folds_scores[1]
		, 'Fold2 acc' : folds_scores[2]
		, 'Fold3 acc' : folds_scores[3]
		, 'Fold4 acc' : folds_scores[4]
		, 'dataset' : config.dataset
		, 'embedding': config.embedding
		, 'embedding_lr': config.emb_lr
		, 'freeze_emb': config.freeze_emb
		, 'i/p and o/p embedding size' : config.d_model
		, 'encoder_layers' : config.encoder_layers
		, 'decoder_layers' : config.decoder_layers
		, 'heads' : config.heads
		, 'FFN size' : config.d_ff
		, 'lr' : config.lr
		, 'batch_size' : config.batch_size
		, 'dropout' : config.dropout
		, 'opt' : config.opt
		}
		# res_data.update(data)
		res_data[str(config.run_name)] = data

		with open(config.val_result_path, 'w', encoding='utf-8') as f:
			json.dump(res_data, f, ensure_ascii= False, indent= 4)
	except:
		pdb.set_trace()
        
        
########################################################
# sentence_processing.py #
########################################################

def sent_to_idx(voc, sent, max_length, flag = 0):
	if flag == 0:
		idx_vec = []
	else:
		idx_vec = [voc.get_id('<s>')]
	for w in sent.split(' '):
		try:
			idx = voc.get_id(w)
			idx_vec.append(idx)
		except:
			idx_vec.append(voc.get_id('unk'))
	# idx_vec.append(voc.get_id('</s>'))
	if flag == 1 and len(idx_vec) < max_length-1:
		idx_vec.append(voc.get_id('</s>'))
	return idx_vec

def sents_to_idx(voc, sents, max_length, flag = 0):
	all_indexes = []
	for sent in sents:
		all_indexes.append(sent_to_idx(voc, sent, max_length, flag))
	return all_indexes

def sent_to_tensor(voc, sentence, device, max_length):
	indexes = sent_to_idx(voc, sentence, max_length)
	return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def batch_to_tensor(voc, sents, device, max_length):
	batch_sent = []
	# batch_label = []
	for sent in sents:
		sent_id = sent_to_tensor(voc, sent, device, max_length)
		batch_sent.append(sent_id)

	return batch_sent

def idx_to_sent(voc, tensor, no_eos=False):
	sent_word_list = []
	for idx in tensor:
		word = voc.get_word(idx.item())
		if no_eos:
			if word != '</s>':
				sent_word_list.append(word)
			# else:
			# 	break
		else:
			sent_word_list.append(word)
	return sent_word_list

def idx_to_sents(voc, tensors, no_eos=False):
	tensors = tensors.transpose(0, 1)
	batch_word_list = []
	for tensor in tensors:
		batch_word_list.append(idx_to_sent(voc, tensor, no_eos))

	return batch_word_list

def pad_seq(seq, max_length, voc):
	seq += [voc.get_id('</s>') for i in range(max_length - len(seq))]
	return seq

def sort_by_len(seqs, input_len, device=None, dim=1):
	orig_idx = list(range(seqs.size(dim)))

	# Index by which sorting needs to be done
	sorted_idx = sorted(orig_idx, key=lambda k: input_len[k], reverse=True)
	sorted_idx= torch.LongTensor(sorted_idx)
	if device:
		sorted_idx = sorted_idx.to(device)

	sorted_seqs = seqs.index_select(1, sorted_idx)
	sorted_lens=  [input_len[i] for i in sorted_idx]

	# For restoring original order
	orig_idx = sorted(orig_idx, key=lambda k: sorted_idx[k])
	orig_idx = torch.LongTensor(orig_idx)
	if device:
		orig_idx = orig_idx.to(device)
	return sorted_seqs, sorted_lens, orig_idx

def restore_order(seqs, input_len, orig_idx):
	orig_seqs= [seqs[i] for i in orig_idx]
	orig_lens= [input_len[i] for i in orig_idx]
	return orig_seqs, orig_lens

def process_batch(sent1s, sent2s, voc1, voc2, device):
	input_len1 = [len(s) for s in sent1s]
	input_len2 = [len(s) for s in sent2s]
	max_length_1 = max(input_len1)
	max_length_2 = max(input_len2)

	sent1s_padded = [pad_seq(s, max_length_1, voc1) for s in sent1s]
	sent2s_padded = [pad_seq(s, max_length_2, voc2) for s in sent2s]

	# Convert to [Max_len X Batch]
	sent1_var = Variable(torch.LongTensor(sent1s_padded)).transpose(0, 1)
	sent2_var = Variable(torch.LongTensor(sent2s_padded)).transpose(0, 1)

	sent1_var = sent1_var.to(device)
	sent2_var = sent2_var.to(device)

	return sent1_var, sent2_var, input_len1, input_len2

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Run Single sequence model')

	# Mode specifications
	parser.add_argument('-mode', type=str, default='train', choices=['train', 'test'], help='Modes: train, test')
	parser.add_argument('-debug', dest='debug', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-debug', dest='debug', action='store_false', help='Operate in normal mode')
	parser.set_defaults(debug=False)
	
	# Run Config
	parser.add_argument('-run_name', type=str, default='debug', help='run name for logs')
	parser.add_argument('-dataset', type=str, default='asdiv-a_fold0_final', help='Dataset')
	parser.add_argument('-display_freq', type=int, default= 10000, help='number of batches after which to display samples')
	parser.add_argument('-outputs', dest='outputs', action='store_true', help='Show full validation outputs')
	parser.add_argument('-no-outputs', dest='outputs', action='store_false', help='Do not show full validation outputs')
	parser.set_defaults(outputs=True)
	parser.add_argument('-results', dest='results', action='store_true', help='Store results')
	parser.add_argument('-no-results', dest='results', action='store_false', help='Do not store results')
	parser.set_defaults(results=True)

	# Meta Attributes
	parser.add_argument('-vocab_size', type=int, default=30000, help='Vocabulary size to consider')
	parser.add_argument('-histogram', dest='histogram', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-histogram', dest='histogram', action='store_false', help='Operate in normal mode')
	parser.set_defaults(histogram=False)
	parser.add_argument('-save_writer', dest='save_writer',action='store_true', help='To write tensorboard')
	parser.add_argument('-no-save_writer', dest='save_writer', action='store_false', help='Dont write tensorboard')
	parser.set_defaults(save_writer=False)

	# Device Configuration
	parser.add_argument('-gpu', type=int, default=2, help='Specify the gpu to use')
	parser.add_argument('-early_stopping', type=int, default=500, help='Early Stopping after n epoch')
	parser.add_argument('-seed', type=int, default=6174, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	parser.add_argument('-ckpt', type=str, default='model', help='Checkpoint file name')
	parser.add_argument('-save_model', dest='save_model',action='store_true', help='To save the model')
	parser.add_argument('-no-save_model', dest='save_model', action='store_false', help='Dont save the model')
	parser.set_defaults(save_model=True)

	# Transformer parameters
	parser.add_argument('-heads', type=int, default=8, help='Number of Attention Heads')
	parser.add_argument('-encoder_layers', type=int, default=6, help='Number of layers in encoder')
	parser.add_argument('-decoder_layers', type=int, default=6, help='Number of layers in decoder')
	parser.add_argument('-d_model', type=int, default=300, help='the number of expected features in the encoder inputs') #768? features of BERT? HAS TO BE 300 if using word2Vec
	parser.add_argument('-d_ff', type=int, default=1200, help='Embedding dimensions of intermediate FFN Layer (refer Vaswani et. al)')
	parser.add_argument('-lr', type=float, default=0.001, help='Learning rate')
	parser.add_argument('-dropout', type=float, default=0.1, help= 'Dropout probability for input/output/state units (0.0: no dropout)')
	parser.add_argument('-warmup', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for')
	parser.add_argument('-max_grad_norm', type=float, default=0.25, help='Clip gradients to this norm')
	parser.add_argument('-batch_size', type=int, default=16, help='Batch size')

	parser.add_argument('-max_length', type=int, default=80, help='Specify max decode steps: Max length string to output')
	parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')
	
	parser.add_argument('-embedding', type=str, default='word2vec', choices=['bert', 'roberta', 'word2vec', 'random'], help='Embeddings')
	parser.add_argument('-word2vec_bin', type=str, default='/datadrive/satwik/global_data/GoogleNews-vectors-negative300.bin', help='Binary file of word2vec')
	parser.add_argument('-emb_name', type=str, default='roberta-base', choices=['bert-base-uncased', 'roberta-base'], help='Which pre-trained model')
	parser.add_argument('-emb_lr', type=float, default=1e-5, help='Larning rate to train embeddings')
	parser.add_argument('-freeze_emb', dest='freeze_emb', action='store_true', help='Freeze embedding weights')
	parser.add_argument('-no-freeze_emb', dest='freeze_emb', action='store_false', help='Train embedding weights')
	parser.set_defaults(freeze_emb=False)

	parser.add_argument('-epochs', type=int, default=10, help='Maximum # of training epochs')
	parser.add_argument('-opt', type=str, default='adamw', choices=['adam', 'adamw', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')

	parser.add_argument('-grade_disp', dest='grade_disp', action='store_true', help='Display grade information in validation outputs')
	parser.add_argument('-no-grade_disp', dest='grade_disp', action='store_false', help='Don\'t display grade information')
	parser.set_defaults(grade_disp=False)
	parser.add_argument('-type_disp', dest='type_disp', action='store_true', help='Display Type information in validation outputs')
	parser.add_argument('-no-type_disp', dest='type_disp', action='store_false', help='Don\'t display Type information')
	parser.set_defaults(type_disp=False)
	parser.add_argument('-challenge_disp', dest='challenge_disp', action='store_true', help='Display information in validation outputs')
	parser.add_argument('-no-challenge_disp', dest='challenge_disp', action='store_false', help='Don\'t display information')
	parser.set_defaults(challenge_disp=False)
	parser.add_argument('-nums_disp', dest='nums_disp', action='store_true', help='Display number of numbers information in validation outputs')
	parser.add_argument('-no-nums_disp', dest='nums_disp', action='store_false', help='Don\'t display number of numbers information')
	parser.set_defaults(nums_disp=True)
	parser.add_argument('-more_nums', dest='more_nums', action='store_true', help='More numbers in Voc2')
	parser.add_argument('-no-more_nums', dest='more_nums', action='store_false', help='Usual numbers in Voc2')
	parser.set_defaults(more_nums=False)
	parser.add_argument('-mawps_vocab', dest='mawps_vocab', action='store_true', help='Custom Numbers in Voc2')
	parser.add_argument('-no-mawps_vocab', dest='mawps_vocab', action='store_false', help='No Custom Numbers in Voc2')
	parser.set_defaults(mawps_vocab=False)

	parser.add_argument('-show_train_acc', dest='show_train_acc', action='store_true', help='Calculate the train accuracy')
	parser.add_argument('-no-show_train_acc', dest='show_train_acc', action='store_false', help='Don\'t calculate the train accuracy')
	parser.set_defaults(show_train_acc=True)

	parser.add_argument('-full_cv', dest='full_cv', action='store_true', help='5-fold CV')
	parser.add_argument('-no-full_cv', dest='full_cv', action='store_false', help='No 5-fold CV')
	parser.set_defaults(full_cv=False)

	return parser

def parse_arguments(arg_dict=None):
    parser = build_parser()
    if arg_dict:
        # Override default values with provided dictionary values
        args = parser.parse_args([])
        for key, value in arg_dict.items():
            setattr(args, key, value)
        return args
    else:
        return parser.parse_args()  # If no dictionary is provided, use default command line arguments
	
class TextDataset(Dataset):
	'''
		Expecting csv files with columns ['Question', 'Equation', 'Numbers', 'Answer']

		Args:
						data_path: Root folder Containing all the data
						dataset: Specific Folder ==> data_path/dataset/	(Should contain train.csv and dev.csv)
						max_length: Self Explanatory
						is_debug: Load a subset of data for faster testing
						is_train: 

	'''

	def __init__(self, data_path='/kaggle/input/svamp-data/data', dataset='mawps', datatype='train', max_length=30, is_debug=False, is_train=False, grade_info=False, type_info=False, challenge_info=False):
		if datatype=='train':
			file_path = os.path.join(data_path, dataset, 'train.csv')
		elif datatype=='dev':
			file_path = os.path.join(data_path, dataset, 'dev.csv')
		else:
			file_path = os.path.join(data_path, dataset, 'test.csv')

		if grade_info:
			self.grade_info = True
		else:
			self.grade_info = False

		if type_info:
			self.type_info = True
		else:
			self.type_info = False

		if challenge_info:
			self.challenge_info = True
		else:
			self.challenge_info = False

		file_df= pd.read_csv(file_path)

		self.ques = file_df['Question'].values # np ndarray of size (#examples,)
		self.eqn = file_df['Equation'].values
		self.nums = file_df['Numbers'].values
		self.ans = file_df['Answer'].values

		if grade_info:
			self.grade = file_df['Grade'].values

		if type_info:
			self.type = file_df['Type'].values

		if challenge_info:
			self.type = file_df['Type'].values
			self.var_type = file_df['Variation Type'].values
			self.annotator = file_df['Annotator'].values
			self.alternate = file_df['Alternate'].values

		if is_debug:
			self.ques = self.ques[:5000:500]
			self.eqn = self.eqn[:5000:500]

		self.max_length = max_length

		if grade_info and type_info:
			all_sents = zip(self.ques, self.eqn, self.nums, self.ans, self.grade, self.type)
		elif grade_info and not type_info:
			all_sents = zip(self.ques, self.eqn, self.nums, self.ans, self.grade)
		elif type_info and not grade_info:
			all_sents = zip(self.ques, self.eqn, self.nums, self.ans, self.type)
		elif challenge_info:
			all_sents = zip(self.ques, self.eqn, self.nums, self.ans, self.type, self.var_type, self.annotator, self.alternate)
		else:
			all_sents = zip(self.ques, self.eqn, self.nums, self.ans)

		if is_train:
			all_sents = sorted(all_sents, key = lambda x : len(x[0].split()))

		if grade_info and type_info:
			self.ques, self.eqn, self.nums, self.ans, self.grade, self.type = zip(*all_sents)
		elif grade_info and not type_info:
			self.ques, self.eqn, self.nums, self.ans, self.grade = zip(*all_sents)
		elif type_info and not grade_info:
			self.ques, self.eqn, self.nums, self.ans, self.type = zip(*all_sents)
		elif challenge_info:
			self.ques, self.eqn, self.nums, self.ans, self.type, self.var_type, self.annotator, self.alternate = zip(*all_sents)
		else:
			self.ques, self.eqn, self.nums, self.ans = zip(*all_sents)

	def __len__(self):
		return len(self.ques)

	def __getitem__(self, idx):
		ques = self.process_string(str(self.ques[idx]))
		eqn = self.process_string(str(self.eqn[idx]))
		nums = self.nums[idx]
		ans = self.ans[idx]
		
		if self.grade_info and self.type_info:
			grade = self.grade[idx]
			type1 = self.type[idx]
			return {'ques': self.curb_to_length(ques), 'eqn': self.curb_to_length(eqn), 'nums': nums, 'ans': ans, 'grade': grade, 'type': type1}
		elif self.grade_info and not self.type_info:
			grade = self.grade[idx]
			return {'ques': self.curb_to_length(ques), 'eqn': self.curb_to_length(eqn), 'nums': nums, 'ans': ans, 'grade': grade}
		elif self.type_info and not self.grade_info:
			type1 = self.type[idx]
			return {'ques': self.curb_to_length(ques), 'eqn': self.curb_to_length(eqn), 'nums': nums, 'ans': ans, 'type': type1}
		elif self.challenge_info:
			type1 = self.type[idx]
			var_type = self.var_type[idx]
			annotator = self.annotator[idx]
			alternate = self.alternate[idx]
			return {'ques': self.curb_to_length(ques), 'eqn': self.curb_to_length(eqn), 'nums': nums, 'ans': ans, 'type': type1, 
					'var_type': var_type, 'annotator': annotator, 'alternate': alternate}

		return {'ques': self.curb_to_length(ques), 'eqn': self.curb_to_length(eqn), 'nums': nums, 'ans': ans}

	def curb_to_length(self, string):
		return ' '.join(string.strip().split()[:self.max_length])

	def process_string(self, string):
		#string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
		string = re.sub(r"\'s", " 's", string)
		string = re.sub(r"\'ve", " 've", string)
		string = re.sub(r"n\'t", " n't", string)
		string = re.sub(r"\'re", " 're", string)
		string = re.sub(r"\'d", " 'd", string)
		string = re.sub(r"\'ll", " 'll", string)
		#string = re.sub(r",", " , ", string)
		#string = re.sub(r"!", " ! ", string)
		#string = re.sub(r"\(", " ( ", string)
		#string = re.sub(r"\)", " ) ", string)
		#string = re.sub(r"\?", " ? ", string)
		#string = re.sub(r"\s{2,}", " ", string)
		return string   
	
class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.scale = nn.Parameter(torch.ones(1)) # nn.Parameter causes the tensor to appear in the model.parameters()

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # max_len x 1
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # torch.arange(0, d_model, 2) gives 2i
		pe[:, 0::2] = torch.sin(position * div_term) # all alternate columns 0 onwards
		pe[:, 1::2] = torch.cos(position * div_term) # all alternate columns 1 onwards
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		'''
			Args:
				x (tensor): embeddings | size : [max_len x batch_size x d_model]
			Returns:
				z (tensor) : embeddings with positional encoding | size : [max_len x batch_size x d_model]
		'''
		
		x = x + self.scale * self.pe[:x.size(0), :]
		z = self.dropout(x)
		return z

class TransformerModel(nn.Module):
	def __init__(self, config, voc1, voc2, device, logger, EOS_tag = '</s>', SOS_tag = '<s>'):
		super(TransformerModel, self).__init__()
		self.config = config
		self.device = device
		self.voc1 = voc1
		self.voc2 = voc2
		self.EOS_tag = EOS_tag
		self.SOS_tag = SOS_tag
		self.EOS_token = voc2.get_id(EOS_tag)
		self.SOS_token = voc2.get_id(SOS_tag)
		self.logger = logger

		# self.logger.debug('Initialising Embeddings.....')

		if self.config.embedding == 'bert':
			config.d_model = 768
			self.embedding1 = BertEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'roberta':
			config.d_model = 768
			self.embedding1 = RobertaEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'word2vec':
			config.d_model = 300
			self.embedding1  = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), 
								freeze = self.config.freeze_emb)
		else:
			self.embedding1  = nn.Embedding(self.voc1.nwords, self.config.d_model)
			nn.init.uniform_(self.embedding1.weight, -1 * self.config.init_range, self.config.init_range)

		self.pos_embedding1 = PositionalEncoding(self.config.d_model, self.config.dropout)

		self.embedding2  = nn.Embedding(self.voc2.nwords, self.config.d_model)
		nn.init.uniform_(self.embedding2.weight, -1 * self.config.init_range, self.config.init_range)
		
		self.pos_embedding2 = PositionalEncoding(self.config.d_model, self.config.dropout)

		# self.logger.debug('Embeddings initialised.....')
		# self.logger.debug('Building Transformer Model.....')

		self.transformer = nn.Transformer(d_model=self.config.d_model, nhead=self.config.heads, 
											num_encoder_layers=self.config.encoder_layers, num_decoder_layers=self.config.decoder_layers, 
											dim_feedforward=self.config.d_ff, dropout=self.config.dropout)
		
		self.fc_out = nn.Linear(self.config.d_model, self.voc2.nwords)

		# self.logger.debug('Transformer Model Built.....')

		self.src_mask = None
		self.trg_mask = None
		self.memory_mask = None

		# self.logger.debug('Initalizing Optimizer and Criterion...')

		self._initialize_optimizer()

		self.criterion = nn.CrossEntropyLoss() # nn.CrossEntropyLoss() does both F.log_softmax() and nn.NLLLoss() 

		# self.logger.info('All Model Components Initialized...')

	def _form_embeddings(self, file_path):
		'''
			Args:
				file_path (string): path of file with word2vec weights
			Returns:
				weight_req (tensor) : embedding matrix | size : [voc1.nwords x d_model]
		'''

		weights_all = models.KeyedVectors.load_word2vec_format(file_path, limit=200000, binary=True)
		weight_req  = torch.randn(self.voc1.nwords, self.config.d_model)
		for key, value in self.voc1.id2w.items():
			if value in weights_all:
				weight_req[key] = torch.FloatTensor(weights_all[value])

		return weight_req

	def _initialize_optimizer(self):
		self.params = list(self.embedding1.parameters()) + list(self.transformer.parameters()) + list(self.fc_out.parameters()) + \
						list(self.embedding2.parameters()) + list(self.pos_embedding1.parameters()) + list(self.pos_embedding2.parameters())
		self.non_emb_params = list(self.transformer.parameters()) + list(self.fc_out.parameters()) + list(self.embedding2.parameters()) + \
								list(self.pos_embedding1.parameters()) + list(self.pos_embedding2.parameters())

		if self.config.opt == 'adam':
			self.optimizer = optim.Adam(
				[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
				{"params": self.non_emb_params, "lr": self.config.lr}]
			)
		elif self.config.opt == 'adamw':
			self.optimizer = optim.AdamW(
				[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
				{"params": self.non_emb_params, "lr": self.config.lr}]
			)
		elif self.config.opt == 'adadelta':
			self.optimizer = optim.Adadelta(
				[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
				{"params": self.non_emb_params, "lr": self.config.lr}]
			)
		elif self.config.opt == 'asgd':
			self.optimizer = optim.ASGD(
				[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
				{"params": self.non_emb_params, "lr": self.config.lr}]
			)
		else:
			self.optimizer = optim.SGD(
				[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
				{"params": self.non_emb_params, "lr": self.config.lr}]
			)

	def generate_square_subsequent_mask(self, sz):
		'''
			Args:
				sz (integer): max_len of sequence in target without EOS i.e. (T-1)
			Returns:
				mask (tensor) : square mask | size : [T-1 x T-1]
		'''

		mask = torch.triu(torch.ones(sz, sz), 1)
		mask = mask.masked_fill(mask==1, float('-inf'))
		return mask

	def make_len_mask(self, inp):
		'''
			Args:
				inp (tensor): input indices | size : [S x BS]
			Returns:
				mask (tensor) : pad mask | size : [BS x S]
		'''

		mask = (inp == -1).transpose(0, 1)
		return mask
		# return (inp == self.EOS_token).transpose(0, 1)

	def forward(self, ques, src, trg):
		'''
			Args:
				ques (list): raw source input | size : [BS]
				src (tensor): source indices | size : [S x BS]
				trg (tensor): target indices | size : [T x BS]
			Returns:
				output (tensor) : Network output | size : [T-1 x BS x voc2.nwords]
		'''

		if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
			self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

		# trg_mask when T-1 = 4: [When decoding for position i, only indexes with 0 in the ith row are attended over]
		# tensor([[0., -inf, -inf, -inf],
		# 		[0., 0., -inf, -inf],
		# 		[0., 0., 0., -inf],
		# 		[0., 0., 0., 0.],

		if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
			src, src_tokens = self.embedding1(ques)
			src = src.transpose(0,1)
			if isinstance(src_tokens, list):
				src_tokens = torch.tensor(src_tokens).to(device)  # Convert to tensor and move to device if needed
			# src: Tensor [S x BS x d_model]
# 			print(src_tokens, src.shape)
			src_pad_mask = self.make_len_mask(src_tokens.transpose(0,1))
			src = self.pos_embedding1(src)
		else:
			src_pad_mask = self.make_len_mask(src)
			src = self.embedding1(src)
			src = self.pos_embedding1(src)

		trg_pad_mask = self.make_len_mask(trg)
		trg = self.embedding2(trg)
		trg = self.pos_embedding2(trg)

		output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
								  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
		
		output = self.fc_out(output)

		return output

	def trainer(self, ques, input_seq1, input_seq2, config, device=None ,logger=None):
		'''
			Args:
				ques (list): raw source input | size : [BS]
				input_seq1 (tensor): source indices | size : [S x BS]
				input_seq2 (tensor): target indices | size : [T x BS]
			Returns:
				fin_loss (float) : Train Loss
		'''

		self.optimizer.zero_grad() # zero out gradients from previous backprop computations

		output = self.forward(ques, input_seq1, input_seq2[:-1,:])
		# output: (T-1) x BS x voc2.nwords [T-1 because it predicts after start symbol]
        
		output_dim = output.shape[-1]
		self.loss = self.criterion(output.reshape(-1, output_dim), input_seq2[1:,:].reshape(-1))

		self.loss.backward()
		if self.config.max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(self.params, self.config.max_grad_norm)
		self.optimizer.step()

		fin_loss = self.loss.item()

		return fin_loss

	def greedy_decode(self, ques=None, input_seq1=None, input_seq2=None, input_len2 = None, validation=False):
		'''
			Args:
				ques (list): raw source input | size : [BS]
				input_seq1 (tensor): source indices | size : [S x BS]
				input_seq2 (tensor): target indices | size : [T x BS]
				input_len2 (list): lengths of targets | size: [BS]
				validation (bool): whether validate
			Returns:
				if validation:
					validation loss (float): Validation loss
					decoded_words (list): predicted equations | size : [BS x target_len]
				else:
					decoded_words (list): predicted equations | size : [BS x target_len]
		'''

		with torch.no_grad():
			loss = 0.0

			if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
				src, _ = self.embedding1(ques)
				src = src.transpose(0,1)
				# src: Tensor [S x BS x emb1_size]
				memory = self.transformer.encoder(self.pos_embedding1(src))
			else: 
				memory = self.transformer.encoder(self.pos_embedding1(self.embedding1(input_seq1)))
			# memory: S x BS x d_model

			input_list = [[self.SOS_token for i in range(input_seq1.size(1))]]

			decoded_words = [[] for i in range(input_seq1.size(1))]

			if validation:
				target_len = max(input_len2)
			else:
				target_len = self.config.max_length

			for step in range(target_len):
				decoder_input = torch.LongTensor(input_list).to(self.device) # seq_len x bs

				decoder_output = self.fc_out(self.transformer.decoder(self.pos_embedding2(self.embedding2(decoder_input)), memory)) # seq_len x bs x voc2.nwords

				if validation:
					loss += self.criterion(decoder_output[-1,:,:], input_seq2[step])

				out_tokens = decoder_output.argmax(2)[-1,:] # bs

				for i in range(input_seq1.size(1)):
					if out_tokens[i].item() == self.EOS_token:
						continue
					decoded_words[i].append(self.voc2.get_word(out_tokens[i].item()))
				
				input_list.append(out_tokens.detach().tolist())

			if validation:
					return loss/target_len, decoded_words
			else:
				return decoded_words

def build_model(config, voc1, voc2, device, logger):
	'''
		Args:
			config (dict): command line arguments
			voc1 (object of class Voc1): vocabulary of source
			voc2 (object of class Voc2): vocabulary of target
			device (torch.device): GPU device
			logger (logger): logger variable to log messages
		Returns:
			model (object of class TransformerModel): model 
	'''

	model = TransformerModel(config, voc1, voc2, device, logger)
	model = model.to(device)

	return model

def train_model(model, train_dataloader, val_dataloader, voc1, voc2, device, config, logger, epoch_offset= 0, min_val_loss=float('inf'), 
				max_val_bleu=0.0, max_val_acc = 0.0, min_train_loss=float('inf'), max_train_acc = 0.0, best_epoch = 0, writer= None):
	'''
		Args:
			model (object of class TransformerModel): model
			train_dataloader (object of class Dataloader): dataloader for train set
			val_dataloader (object of class Dataloader): dataloader for dev set
			voc1 (object of class Voc1): vocabulary of source
			voc2 (object of class Voc2): vocabulary of target
			device (torch.device): GPU device
			config (dict): command line arguments
			logger (logger): logger variable to log messages
			epoch_offset (int): How many epochs of training already done
			min_val_loss (float): minimum validation loss
			max_val_bleu (float): maximum valiadtion bleu score
			max_val_acc (float): maximum validation accuracy score
			min_train_loss (float): minimum train loss
			max_train_acc (float): maximum train accuracy
			best_epoch (int): epoch with highest validation accuracy
			writer (object of class SummaryWriter): writer for Tensorboard
		Returns:
			max_val_acc (float): maximum validation accuracy score
	'''

	if config.histogram and config.save_writer and writer:
		for name, param in model.named_parameters():
			writer.add_histogram(name, param, epoch_offset)
	
	estop_count=0
	
	for epoch in range(1, config.epochs + 1):
		od = OrderedDict()
		od['Epoch'] = epoch + epoch_offset
		print_log(logger, od)

		batch_num = 1
		train_loss_epoch = 0.0
		train_acc_epoch = 0.0
		train_acc_epoch_cnt = 0.0
		train_acc_epoch_tot = 0.0
		val_loss_epoch = 0.0

		start_time= time()
		total_batches = len(train_dataloader)

		for data in train_dataloader:
			ques = data['ques']

			sent1s = sents_to_idx(voc1, data['ques'], config.max_length, flag=0)
			sent2s = sents_to_idx(voc2, data['eqn'], config.max_length, flag=1)
			sent1_var, sent2_var, input_len1, input_len2  = process_batch(sent1s, sent2s, voc1, voc2, device)

			nums = data['nums']
			ans = data['ans']

			model.train()

			loss = model.trainer(ques, sent1_var, sent2_var, config, device, logger)
			train_loss_epoch += loss

			if config.show_train_acc:
				model.eval()

				_, decoder_output = model.greedy_decode(ques, sent1_var, sent2_var, input_len2, validation=True)
				temp_acc_cnt, temp_acc_tot, _ = cal_score(decoder_output, nums, ans)
				train_acc_epoch_cnt += temp_acc_cnt
				train_acc_epoch_tot += temp_acc_tot

			print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)
			batch_num+=1

		train_loss_epoch = train_loss_epoch / len(train_dataloader)
		if config.show_train_acc:
			train_acc_epoch = train_acc_epoch_cnt/train_acc_epoch_tot
		else:
			train_acc_epoch = 0.0

		time_taken = (time() - start_time)/60.0

		if config.save_writer and writer:
			writer.add_scalar('loss/train_loss', train_loss_epoch, epoch + epoch_offset)

		# logger.debug('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_taken))
		# logger.debug('Starting Validation')

		val_bleu_epoch, val_loss_epoch, val_acc_epoch = run_validation(config=config, model=model, val_dataloader=val_dataloader, 
																	voc1=voc1, voc2=voc2, device=device, logger=logger, epoch_num = epoch)

		if train_loss_epoch < min_train_loss:
			min_train_loss = train_loss_epoch

		if train_acc_epoch > max_train_acc:
			max_train_acc = train_acc_epoch

		if val_bleu_epoch[0] > max_val_bleu:
			max_val_bleu = val_bleu_epoch[0]

		if val_loss_epoch < min_val_loss:
			min_val_loss = val_loss_epoch

		if val_acc_epoch > max_val_acc:
			max_val_acc = val_acc_epoch
			best_epoch = epoch + epoch_offset

			state = {
				'epoch' : epoch + epoch_offset,
				'best_epoch': best_epoch,
				'model_state_dict': model.state_dict(),
				'voc1': model.voc1,
				'voc2': model.voc2,
				'optimizer_state_dict': model.optimizer.state_dict(),
				'train_loss_epoch' : train_loss_epoch,
				'min_train_loss' : min_train_loss,
				'train_acc_epoch' : train_acc_epoch,
				'max_train_acc' : max_train_acc,
				'val_loss_epoch' : val_loss_epoch,
				'min_val_loss' : min_val_loss,
				'val_acc_epoch' : val_acc_epoch,
				'max_val_acc' : max_val_acc,
				'val_bleu_epoch': val_bleu_epoch[0],
				'max_val_bleu': max_val_bleu
			}
			# logger.debug('Validation Bleu: {}'.format(val_bleu_epoch[0]))

			if config.save_model:
				save_checkpoint(state, epoch + epoch_offset, logger, config.model_path, config.ckpt)
			estop_count = 0
		else:
			estop_count+=1

		# if config.save_writer and writer:
		# 	writer.add_scalar('loss/val_loss', val_loss_epoch, epoch + epoch_offset)
		# 	writer.add_scalar('acc/val_score', val_score_epoch[0], epoch + epoch_offset)

		od = OrderedDict()
		od['Epoch'] = epoch + epoch_offset
		od['best_epoch'] = best_epoch
		od['train_loss_epoch'] = train_loss_epoch
		od['min_train_loss'] = min_train_loss
		od['val_loss_epoch']= val_loss_epoch
		od['min_val_loss']= min_val_loss
		od['train_acc_epoch'] = train_acc_epoch
		od['max_train_acc'] = max_train_acc
		od['val_acc_epoch'] = val_acc_epoch
		od['max_val_acc'] = max_val_acc
		od['val_bleu_epoch'] = val_bleu_epoch
		od['max_val_bleu'] = max_val_bleu
		# print_log(logger, od)

		if config.histogram and config.save_writer and writer:
			for name, param in model.named_parameters():
				writer.add_histogram(name, param, epoch + epoch_offset)

		if estop_count >config.early_stopping:
			# logger.debug('Early Stopping at Epoch: {} after no improvement in {} epochs'.format(epoch, estop_count))
			break

	if config.save_writer:
		writer.export_scalars_to_json(os.path.join(config.board_path, 'all_scalars.json'))
		writer.close()

	# logger.info('Training Completed for {} epochs'.format(config.epochs))

	if config.results:
		store_results(config, max_val_bleu, max_val_acc, min_val_loss, max_train_acc, min_train_loss, best_epoch)
		# logger.info('Scores saved at {}'.format(config.result_path))

	return max_val_acc

def run_validation(config, model, val_dataloader, voc1, voc2, device, logger, epoch_num, validation = True):
	'''
		Args:
			config (dict): command line arguments
			model (object of class TransformerModel): model
			val_dataloader (object of class Dataloader): dataloader for dev set
			voc1 (object of class Voc1): vocabulary of source
			voc2 (object of class Voc2): vocabulary of target
			device (torch.device): GPU device
			logger (logger): logger variable to log messages
			epoch_num (int): Ongoing epoch number
			validation (bool): whether validating
		Returns:
			if config.mode == 'test':
				max_test_acc (float): maximum test accuracy obtained
			else:
				val_bleu_epoch (float): validation bleu score for this epoch
				val_loss_epoch (float): va;iadtion loss for this epoch
				val_acc (float): validation accuracy score for this epoch
	'''

	batch_num = 1
	val_loss_epoch = 0.0
	val_bleu_epoch = 0.0
	val_acc_epoch = 0.0
	val_acc_epoch_cnt = 0.0
	val_acc_epoch_tot = 0.0

	model.eval() # Set specific layers such as dropout to evaluation mode

	refs= []
	hyps= []

	if config.mode == 'test':
		questions, gen_eqns, act_eqns, scores = [], [], [], []

	display_n = config.batch_size

	with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
		f_out.write('---------------------------------------\n')
		f_out.write('Epoch: ' + str(epoch_num) + '\n')
		f_out.write('---------------------------------------\n')
	total_batches = len(val_dataloader)
	for data in val_dataloader:
		sent1s = sents_to_idx(voc1, data['ques'], config.max_length, flag = 0)
		sent2s = sents_to_idx(voc2, data['eqn'], config.max_length, flag = 0)
		nums = data['nums']
		ans = data['ans']
		if config.grade_disp:
			grade = data['grade']
		if config.type_disp:
			type1 = data['type']
		if config.challenge_disp:
			type1 = data['type']
			var_type = data['var_type']
			annotator = data['annotator']
			alternate = data['alternate']

		ques = data['ques']

		sent1_var, sent2_var, input_len1, input_len2 = process_batch(sent1s, sent2s, voc1, voc2, device)

		val_loss, decoder_output = model.greedy_decode(ques, sent1_var, sent2_var, input_len2, validation=True)

		temp_acc_cnt, temp_acc_tot, disp_corr = cal_score(decoder_output, nums, ans)
		val_acc_epoch_cnt += temp_acc_cnt
		val_acc_epoch_tot += temp_acc_tot

		sent1s = idx_to_sents(voc1, sent1_var, no_eos= True)
		sent2s = idx_to_sents(voc2, sent2_var, no_eos= True)

		refs += [[' '.join(sent2s[i])] for i in range(sent2_var.size(1))]
		hyps += [' '.join(decoder_output[i]) for i in range(sent1_var.size(1))]

		if config.mode == 'test':
			questions+= data['ques']
			gen_eqns += [' '.join(decoder_output[i]) for i in range(sent1_var.size(1))]
			act_eqns += [' '.join(sent2s[i]) for i in range(sent2_var.size(1))]
			scores   += [cal_score([decoder_output[i]], [nums[i]], [ans[i]], [data['eqn'][i]])[0] for i in range(sent1_var.size(1))]

		with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
			f_out.write('Batch: ' + str(batch_num) + '\n')
			f_out.write('---------------------------------------\n')
			for i in range(len(sent1s[:display_n])):
				try:
					f_out.write('Example: ' + str(i) + '\n')
					if config.grade_disp:
						f_out.write('Grade: ' + str(grade[i].item()) + '\n')
					if config.type_disp:
						f_out.write('Type: ' + str(type1[i]) + '\n')
					f_out.write('Source: ' + stack_to_string(sent1s[i]) + '\n')
					f_out.write('Target: ' + stack_to_string(sent2s[i]) + '\n')
					f_out.write('Generated: ' + stack_to_string(decoder_output[i]) + '\n')
					if config.challenge_disp:
						f_out.write('Type: ' + str(type1[i]) + '\n')
						f_out.write('Variation Type: ' + str(var_type[i]) + '\n')
						f_out.write('Annotator: ' + str(annotator[i]) + '\n')
						f_out.write('Alternate: ' + str(alternate[i].item()) + '\n')
					if config.nums_disp:
						src_nums = 0
						tgt_nums = 0
						pred_nums = 0
						for k in range(len(sent1s[i])):
							if sent1s[i][k][:6] == 'number':
								src_nums += 1
						for k in range(len(sent2s[i])):
							if sent2s[i][k][:6] == 'number':
								tgt_nums += 1
						for k in range(len(decoder_output[i])):
							if decoder_output[i][k][:6] == 'number':
								pred_nums += 1
						f_out.write('Numbers in question: ' + str(src_nums) + '\n')
						f_out.write('Numbers in Target Equation: ' + str(tgt_nums) + '\n')
						f_out.write('Numbers in Predicted Equation: ' + str(pred_nums) + '\n')
					f_out.write('Result: ' + str(disp_corr[i]) + '\n' + '\n')
				except:
					# logger.warning('Exception: Failed to generate')
					pdb.set_trace()
					break
			f_out.write('---------------------------------------\n')
			f_out.close()

		if batch_num % config.display_freq == 0:
			for i in range(len(sent1s[:display_n])):
				try:
					od = OrderedDict()
					# logger.info('-------------------------------------')
					od['Source'] = ' '.join(sent1s[i])

					od['Target'] = ' '.join(sent2s[i])

					od['Generated'] = ' '.join(decoder_output[i])
					print_log(logger, od)
					# logger.info('-------------------------------------')
				except:
					# logger.warning('Exception: Failed to generate')
					pdb.set_trace()
					break

		val_loss_epoch += val_loss
		print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)
		batch_num += 1

	val_bleu_epoch = bleu_scorer(refs, hyps)
	if config.mode == 'test':
		results_df = pd.DataFrame([questions, act_eqns, gen_eqns, scores]).transpose()
		results_df.columns = ['Question', 'Actual Equation', 'Generated Equation', 'Score']
		csv_file_path = os.path.join(config.outputs_path, config.dataset+'.csv')
		results_df.to_csv(csv_file_path, index = False)
		return sum(scores)/len(scores)

	val_acc_epoch = val_acc_epoch_cnt/val_acc_epoch_tot

	return val_bleu_epoch, val_loss_epoch/len(val_dataloader), val_acc_epoch


global log_folder
global model_folder
global result_folder
global data_path
global board_path

log_folder = 'logs'
model_folder = 'models'
outputs_folder = 'outputs'
result_folder = './out/'
data_path = '/kaggle/input/svamp-data/data'
board_path = './runs/'

def load_data(config, logger):
	'''
		Loads the data from the datapath in torch dataset form

		Args:
			config (dict) : configuration/args
			logger (logger) : logger object for logging

		Returns:
			dataloader(s) 
	'''
	
	if config.mode == 'train':
		# logger.debug('Loading Training Data...')

		'''Load Datasets'''
		train_set = TextDataset(data_path=data_path, dataset=config.dataset,
								datatype='train', max_length=config.max_length, is_debug=config.debug, is_train=True)
		val_set = TextDataset(data_path=data_path, dataset=config.dataset,
							  datatype='dev', max_length=config.max_length, is_debug=config.debug, grade_info=config.grade_disp, 
							  type_info=config.type_disp, challenge_info=config.challenge_disp)

		'''In case of sort by length, write a different case with shuffle=False '''
		train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=5)
		val_dataloader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

		train_size = len(train_dataloader) * config.batch_size
		val_size = len(val_dataloader)* config.batch_size
		
		msg = 'Training and Validation Data Loaded:\nTrain Size: {}\nVal Size: {}'.format(train_size, val_size)
		# logger.info(msg)

		return train_dataloader, val_dataloader

	elif config.mode == 'test':
		# logger.debug('Loading Test Data...')

		test_set = TextDataset(data_path=data_path, dataset=config.dataset,
							   datatype='test', max_length=config.max_length, is_debug=config.debug)
		test_dataloader = DataLoader(
			test_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

		# logger.info('Test Data Loaded...')
		return test_dataloader

	else:
		# logger.critical('Invalid Mode Specified')
		raise Exception('{} is not a valid mode'.format(config.mode))


'''read arguments'''

kaggle_args = {
    'debug': False,
    'mode': 'train',
    'gpu': -1,
    'dropout': 0.1,
    'heads': 4,
    'encoder_layers': 1,
    'decoder_layers': 1,
    'd_model': 768,
    'd_ff': 256,
    'lr': 0.0001,
    'emb_lr': 1e-5,
    'batch_size': 32,
    'epochs': 5,
    'embedding': 'roberta',
    'emb_name': 'roberta-base',
    'mawps_vocab': True,
    'dataset': 'mawps-asdiv-a_svamp',
    'run_name': 'mawps_try1',
	'logging': 0
}


def main():

    config =  parse_arguments(kaggle_args)

    mode = config.mode
    if mode == 'train':
        is_train = True
    else:
        is_train = False

    ''' Set seed for reproducibility'''
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    '''GPU initialization'''
    device = gpu_init_pytorch(config.gpu)

    if config.full_cv:
        global data_path 
        data_name = config.dataset
        data_path = data_path + data_name + '/'
        config.val_result_path = os.path.join(result_folder, 'CV_results_{}.json'.format(data_name))
        fold_acc_score = 0.0
        folds_scores = []
        for z in range(5):
            run_name = config.run_name + '_fold' + str(z)
            config.dataset = 'fold' + str(z)
            config.log_path = os.path.join(log_folder, run_name)
            config.model_path = os.path.join(model_folder, run_name)
            config.board_path = os.path.join(board_path, run_name)
            config.outputs_path = os.path.join(outputs_folder, run_name)

            vocab1_path = os.path.join(config.model_path, 'vocab1.p')
            vocab2_path = os.path.join(config.model_path, 'vocab2.p')
            config_file = os.path.join(config.model_path, 'config.p')
            log_file = os.path.join(config.log_path, 'log.txt')

            if config.results:
                config.result_path = os.path.join(result_folder, 'val_results_{}_{}.json'.format(data_name, config.dataset))

            if is_train:
                create_save_directories(config.log_path)
                create_save_directories(config.model_path)
                create_save_directories(config.outputs_path)
            else:
                create_save_directories(config.log_path)
                create_save_directories(config.result_path)

            logger = get_logger(run_name, log_file, logging.DEBUG)
            writer = SummaryWriter(config.board_path)

            # logger.debug('Created Relevant Directories')
            # logger.info('Experiment Name: {}'.format(config.run_name))

            '''Read Files and create/load Vocab'''
            if is_train:
                train_dataloader, val_dataloader = load_data(config, logger)

                # logger.debug('Creating Vocab...')

                voc1 = Voc1()
                voc1.create_vocab_dict(config, train_dataloader)

                # Removed
                # voc1.add_to_vocab_dict(config, val_dataloader)

                voc2 = Voc2(config)
                voc2.create_vocab_dict(config, train_dataloader)

                # Removed
                # voc2.add_to_vocab_dict(config, val_dataloader)

                # logger.info('Vocab Created with number of words : {}'.format(voc1.nwords))

                with open(vocab1_path, 'wb') as f:
                    pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(vocab2_path, 'wb') as f:
                    pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)

                # logger.info('Vocab saved at {}'.format(vocab1_path))

            else:
                test_dataloader = load_data(config, logger)
                # logger.info('Loading Vocab File...')

                with open(vocab1_path, 'rb') as f:
                    voc1 = pickle.load(f)
                with open(vocab2_path, 'rb') as f:
                    voc2 = pickle.load(f)

                # logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, voc1.nwords))

            # TO DO : Load Existing Checkpoints here
            checkpoint = get_latest_checkpoint(config.model_path, logger)

            if is_train:
                model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

                # logger.info('Initialized Model')

                if checkpoint == None:
                    min_val_loss = torch.tensor(float('inf')).item()
                    min_train_loss = torch.tensor(float('inf')).item()
                    max_val_bleu = 0.0
                    max_val_acc = 0.0
                    max_train_acc = 0.0
                    best_epoch = 0
                    epoch_offset = 0
                else:
                    epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
                                                                        load_checkpoint(model, config.mode, checkpoint, logger, device)

                with open(config_file, 'wb') as f:
                    pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

                # logger.debug('Config File Saved')

                # logger.info('Starting Training Procedure')
                max_val_acc = train_model(model, train_dataloader, val_dataloader, voc1, voc2, device, config, logger, 
                            epoch_offset, min_val_loss, max_val_bleu, max_val_acc, min_train_loss, max_train_acc, best_epoch, writer)

            else:
                gpu = config.gpu
                mode = config.mode
                dataset = config.dataset
                batch_size = config.batch_size
                # with open(config_file, 'rb') as f:
                #     config = AttrDict(pickle.load(f))
                #     config.gpu = gpu
                #     config.mode = mode
                #     config.dataset = dataset
                #     config.batch_size = batch_size

                # with open(config_file, 'rb') as f:
                #     config = AttrDict(pickle.load(f))
                #     config.gpu = gpu

                model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

                epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
                                                                        load_checkpoint(model, config.mode, checkpoint, logger, device)

                # logger.info('Prediction from')
                od = OrderedDict()
                # od['epoch'] = ep_offset
                od['min_train_loss'] = min_train_loss
                od['min_val_loss'] = min_val_loss
                od['max_train_acc'] = max_train_acc
                od['max_val_acc'] = max_val_acc
                od['max_val_bleu'] = max_val_bleu
                od['best_epoch'] = best_epoch
                print_log(logger, od)

                test_acc_epoch = run_validation(config, model, test_dataloader, voc1, voc2, device, logger, 0)
                # logger.info('Accuracy: {}'.format(test_acc_epoch))

            fold_acc_score += max_val_acc
            folds_scores.append(max_val_acc)

        fold_acc_score = fold_acc_score/5
        store_val_results(config, fold_acc_score, folds_scores)
        # logger.info('Final Val score: {}'.format(fold_acc_score))

    else:
        '''Run Config files/paths'''
        run_name = config.run_name
        config.log_path = os.path.join(log_folder, run_name)
        config.model_path = os.path.join(model_folder, run_name)
        config.board_path = os.path.join(board_path, run_name)
        config.outputs_path = os.path.join(outputs_folder, run_name)

        vocab1_path = os.path.join(config.model_path, 'vocab1.p')
        vocab2_path = os.path.join(config.model_path, 'vocab2.p')
        config_file = os.path.join(config.model_path, 'config.p')
        log_file = os.path.join(config.log_path, 'log.txt')

        if config.results:
            config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))

        if is_train:
            create_save_directories(config.log_path)
            create_save_directories(config.model_path)
            create_save_directories(config.outputs_path)
        else:
            create_save_directories(config.log_path)
            create_save_directories(config.result_path)

        logger = get_logger(run_name, log_file, logging.DEBUG)
        writer = SummaryWriter(config.board_path)

        # logger.debug('Created Relevant Directories')
        # logger.info('Experiment Name: {}'.format(config.run_name))

        '''Read Files and create/load Vocab'''
        if is_train:
            train_dataloader, val_dataloader = load_data(config, logger)

            # logger.debug('Creating Vocab...')

            voc1 = Voc1()
            voc1.create_vocab_dict(config, train_dataloader)

            # Removed
            # voc1.add_to_vocab_dict(config, val_dataloader)

            voc2 = Voc2(config)
            voc2.create_vocab_dict(config, train_dataloader)

            # Removed
            # voc2.add_to_vocab_dict(config, val_dataloader)

            # logger.info('Vocab Created with number of words : {}'.format(voc1.nwords))

            with open(vocab1_path, 'wb') as f:
                pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(vocab2_path, 'wb') as f:
                pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)

            # logger.info('Vocab saved at {}'.format(vocab1_path))

        else:
            test_dataloader = load_data(config, logger)
            # logger.info('Loading Vocab File...')

            with open(vocab1_path, 'rb') as f:
                voc1 = pickle.load(f)
            with open(vocab2_path, 'rb') as f:
                voc2 = pickle.load(f)

            # logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, voc1.nwords))

        # Load Existing Checkpoints here
        checkpoint = get_latest_checkpoint(config.model_path, logger)

        if is_train:
            model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

            # logger.info('Initialized Model')

            if checkpoint == None:
                min_val_loss = torch.tensor(float('inf')).item()
                min_train_loss = torch.tensor(float('inf')).item()
                max_val_bleu = 0.0
                max_val_acc = 0.0
                max_train_acc = 0.0
                best_epoch = 0
                epoch_offset = 0
            else:
                epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
                                                                    load_checkpoint(model, config.mode, checkpoint, logger, device)

            with open(config_file, 'wb') as f:
                pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

            # logger.debug('Config File Saved')

            # logger.info('Starting Training Procedure')
            train_model(model, train_dataloader, val_dataloader, voc1, voc2, device, config, logger, 
                        epoch_offset, min_val_loss, max_val_bleu, max_val_acc, min_train_loss, max_train_acc, best_epoch, writer)

        else:
            gpu = config.gpu
            mode = config.mode
            dataset = config.dataset
            batch_size = config.batch_size
            # with open(config_file, 'rb') as f:
            #     config = AttrDict(pickle.load(f))
            #     config.gpu = gpu
            #     config.mode = mode
            #     config.dataset = dataset
            #     config.batch_size = batch_size

            # with open(config_file, 'rb') as f:
            #     config = AttrDict(pickle.load(f))
            #     config.gpu = gpu

            model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

            epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
                                                                    load_checkpoint(model, config.mode, checkpoint, logger, device)

            # logger.info('Prediction from')
            od = OrderedDict()
            # od['epoch'] = ep_offset
            od['min_train_loss'] = min_train_loss
            od['min_val_loss'] = min_val_loss
            od['max_train_acc'] = max_train_acc
            od['max_val_acc'] = max_val_acc
            od['max_val_bleu'] = max_val_bleu
            od['best_epoch'] = best_epoch
            print_log(logger, od)

            test_acc_epoch = run_validation(config, model, test_dataloader, voc1, voc2, device, logger, 0)
            # logger.info('Accuracy: {}'.format(test_acc_epoch))

# if __name__ == '__main__':
	# main()


###########################################
####### end - code ######################
###########################################

def modify_sentence_and_extract_numbers(sentence):
    # Find all numbers in the sentence
    numbers = re.findall(r'\d+', sentence)
    
    # Create a modified sentence by replacing each number with "number0", "number1", etc.
    modified_sentence = sentence
    for i, num in enumerate(numbers):
        modified_sentence = modified_sentence.replace(num, f'number{i}', 1)
    
    # Convert the numbers from strings to integers
    numbers = [int(num) for num in numbers]
    
    return modified_sentence, numbers

def load_model(model_path, config, voc1, voc2, device):
    model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=None)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Initialize model and vocabs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config =  parse_arguments(kaggle_args)


vocab1_path = './model/vocab1.p'  # Update with actual paths
vocab2_path = './model/vocab2.p'

with open(vocab1_path, 'rb') as f:
    voc1 = pickle.load(f)

with open(vocab2_path, 'rb') as f:
    voc2 = pickle.load(f)

model_path = './model/model_5.pt'  # Path to your trained model file
model = load_model(model_path, config, voc1, voc2, device)


app=FastAPI()

class Input(BaseModel):
    Word_Problem: str


def process_batch(sent1s, voc1, device):
	input_len1 = [len(s) for s in sent1s]
	max_length_1 = max(input_len1)

	sent1s_padded = [pad_seq(s, max_length_1, voc1) for s in sent1s]

	# Convert to [Max_len X Batch]
	sent1_var = Variable(torch.LongTensor(sent1s_padded)).transpose(0, 1)

	sent1_var = sent1_var.to(device)

	return sent1_var, input_len1


@app.get("/")
def read_root():
    return {"msg":"Word Problem Solver"}

@app.post("/predict")
def predict_output(input:Input):
    data = input.dict()
    print("---------------------------------------")
    print(data)
    print("---------------------------------------")
	
    data_in = data['Word_Problem']
	
    device = gpu_init_pytorch(-1)

    with open(vocab1_path, 'rb') as f:
        voc1 = pickle.load(f)
        
    data_in, numbers = modify_sentence_and_extract_numbers(data_in)
    input_problem = sents_to_idx(voc1, [data_in], 80)

    print("---------------------------------------")
    print(data_in, numbers, input_problem)
    print("---------------------------------------")
	
    sent1_var, input_len1 = process_batch(input_problem, voc1, device)

    decoder_output = model.greedy_decode([data_in], sent1_var, None, input_len1, None)
    print("---------------------------------------")
    print(decoder_output)
    print("---------------------------------------")	
    answer = cal_score(decoder_output, numbers)
    print("---------------------------------------")
    print(answer)
    print("---------------------------------------")
    return {
        'answer': str(answer)
    }

if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)