#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
in this file we put all the parameters here
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'
learning_rate = 0.001
EPOCHS = 30
num_workers = 4
num_views = 3
img_size = 128
batch_size = 64 #512
# num_samples = 3183
num_classes = 40
delay_scale = 1
syn_k = 2  #3
threshold_time = 1800
num_units = 16
dropout = 0.3
# 2 sync
embedding_file_name = 'syn_nonlinear_v1'
# central_file_name = 'central_process'
model40_path = '/training_data/modelnet40png/modelnet40v1png/'   #### change it!


