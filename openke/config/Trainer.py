# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm

class Trainer(object):

	def __init__(self, 
				 model = None,
				 data_loader = None,
				 train_times = 1000,
				 alpha = 0.5,
				 use_gpu = True,
				 opt_method = "sgd",
				 save_steps = None,
				 checkpoint_dir = None):

		self.work_threads = 8 #线程
		self.train_times = train_times #训练次数

		self.opt_method = opt_method #优化方法
		self.optimizer = None #优化器
		self.lr_decay = 0   #r_decay即学习率衰退，一般设置为1e-8 ,因为一般情况下循环迭代次数越多的时候，学习率的步伐就应该越来越小，这样才能慢慢接近函数的极值点,。
		self.weight_decay = 0  #权衰量,用于防止过拟合
		self.alpha = alpha  # 网络的基础学习速率,一般设一个很小的值

		self.model = model #模型
		self.data_loader = data_loader #训练数据载入
		self.use_gpu = use_gpu #用gpu
		self.save_steps = save_steps
		self.checkpoint_dir = checkpoint_dir

	def train_one_step(self, data):
		#optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
		self.optimizer.zero_grad()
		loss = self.model({
			'batch_h': self.to_var(data['batch_h'], self.use_gpu), #head
			'batch_t': self.to_var(data['batch_t'], self.use_gpu), #tail
			'batch_r': self.to_var(data['batch_r'], self.use_gpu), #relation
			'batch_y': self.to_var(data['batch_y'], self.use_gpu), #??
			'mode': data['mode']
		})
		loss.backward()
		self.optimizer.step()		 
		return loss.item()

	#开始训练
	def run(self):
		if self.use_gpu:
			self.model.cuda()

		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(
				self.model.parameters(),
				lr=self.alpha,
				lr_decay=self.lr_decay,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		else:  # 啥都不是的话就用SGD
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.alpha,
				weight_decay=self.weight_decay,
			)
		print("Finish initializing...")
		
		training_range = tqdm(range(self.train_times))
		#训练次数
		for epoch in training_range:
			res = 0.0
			#训练数据中dataloader，可迭代，每次nbatches=100取样
			for data in self.data_loader:
				#one_step
				loss = self.train_one_step(data)
				res += loss
			#输出每次的loss与进度
			training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
			
			if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
				print("Epoch %d has finished, saving..." % (epoch))
				self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))





	def set_model(self, model):
		self.model = model

	def to_var(self, x, use_gpu):
		if use_gpu:
			return Variable(torch.from_numpy(x).cuda())
		else:
			return Variable(torch.from_numpy(x))

	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_lr_decay(self, lr_decay):
		self.lr_decay = lr_decay

	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay

	def set_opt_method(self, opt_method):
		self.opt_method = opt_method

	def set_train_times(self, train_times):
		self.train_times = train_times

	def set_save_steps(self, save_steps, checkpoint_dir = None):
		self.save_steps = save_steps
		if not self.checkpoint_dir:
			self.set_checkpoint_dir(checkpoint_dir)

	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir