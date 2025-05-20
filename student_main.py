#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
from pathlib import Path
import torch
import gc
import random
import numpy as np
import os
import csv
from utils.student_utils import run_for_seed

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser.add_argument('--dataset', default='Haptics')
parser.add_argument('--n_channels', default=1, type=int)
parser.add_argument('--output_classes', default=7, type=int)#automated within code
parser.add_argument('--lr', default=0.1, type=float)
# parser.add_argument('--lr_decay_epochs', type=int, default=[40, 60], nargs='+')
# parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
parser.add_argument('--lr_decay_epochs', type=int, default=[25, 30, 35], nargs='+')
parser.add_argument('--lr_decay_gamma', default=0.5, type=float)
parser.add_argument('--batch', default=32, type=int)
parser.add_argument('--epochs', default=800, type=int)
parser.add_argument('--patience', default=50, type=int)
parser.add_argument('--downsample_size', default=100, type=int)
parser.add_argument('--seed', default=1001, type=int)
# parser.add_argument('--seed_array', )
parser.add_argument('--seed_array', type=int, nargs='+', default=[24,10200], help='The array of seed integers')
parser.add_argument('--val_size', default=0.2)
parser.add_argument('--data_folder', default=Path('../ucr_data'))
parser.add_argument('--save_dir', default='student_models/')
parser.add_argument('--teacher_load', default='teacher_models/')
parser.add_argument('--common_dir', default='student_results/')
parser.add_argument('--result_csv', default='test_results_student.csv')
parser.add_argument('--avg_result_csv', default='avg_test_results_student.csv')
parser.add_argument('--max_result_csv', default='max_test_results_student.csv')


#for LSTM hidden size and number of layers
parser.add_argument('--teacher_hidden_size', default=100, type=int)
parser.add_argument('--student_hidden_size', default=16, type=int)
parser.add_argument('--teacher_num_layers', default=3, type=int)
parser.add_argument('--student_num_layers', default=1, type=int)

#for resnet
parser.add_argument('--mid_channels_teacher', default=64, type=int)
parser.add_argument('--mid_channels_student', default=4, type=int)

#distillation ratios
parser.add_argument('--task_ratio', default=1, type=float)
parser.add_argument('--dist_ratio', default=0, type=float)
parser.add_argument('--dtw_rkd_ratio', default=0, type=float)
parser.add_argument('--memory_loss_ratio', default=0, type=float)
parser.add_argument('--hinton_loss_ratio', default=0, type=float)
parser.add_argument('--bench_rkd_dist_ratio', default=0, type=float)
parser.add_argument('--bench_rkd_angle_ratio', default=0, type=float)
parser.add_argument('--bench_fitnet_ratio', default=0, type=float)
parser.add_argument('--bench_attention_ratio', default=0, type=float)
parser.add_argument('--bench_temp_dist_ratio', default=0, type=float)
parser.add_argument('--bench_DKD_ratio', default=0, type=float)
parser.add_argument('--bench_VID_ratio', default=0, type=float)
parser.add_argument('--bench_SaliencyKD_ratio', default=0, type=float)
# parser.add_argument('--ratios', default=[0.01 ,0.1, 0.5, 1 ,10, 100 ,150, 200, 300])
parser.add_argument('--ratios', type=float, nargs='+', default=[0.01 ,0.1], help='The array of distillation ratios')

#for slaiencyKD
parser.add_argument('--num_samples', default=50, type=int)
parser.add_argument('--sub_seq_len', default=5, type=int)

#for fitnets
parser.add_argument('--load', default=None)
parser.add_argument('--last_weight', default=None)

#for HKD and DKD(default weights from orig papaer)
parser.add_argument('--temperature', default=2, type=float)
parser.add_argument('--alpha_DKD', default=1.0, type=float)
parser.add_argument('--beta_DKD', default=8.0, type=float)
parser.add_argument('--dtw_gamma', default=1.2, type=float)

#for VID
parser.add_argument('--init_pred_var_VID', default=5.0, type=float)
parser.add_argument('--eps_VID', default=1e-5, type=float)

parser.add_argument('--base', default='FCNBaselineSmall')
parser.add_argument('--teacher_base',default='FCNBaseline')
parser.add_argument('--l2normalize', choices=['true', 'false'], default='true')
parser.add_argument('--teacher_l2normalize', choices=['true', 'false'], default='true')

inp_args = parser.parse_args()
inp_args

#clear cache
torch.cuda.empty_cache()
gc.collect() 

inp_args.teacher_load = inp_args.teacher_load+ inp_args.dataset + '/best.pth'


# In[2]:


def train_for_seed(inp_args, save_path_attributes):
    current_test_res = [0,0,0,0]
    for random_seed in inp_args.seed_array:
        inp_args.seed = random_seed
        #eliminate randomness
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
            set_random_seed(inp_args.seed)
    
        test_results , student = run_for_seed(inp_args)
        current_test_res= [a+b for a,b in zip(current_test_res , list(test_results.values()))]
    
        file_path = inp_args.common_dir + inp_args.result_csv
        full_list = save_path_attributes +[inp_args.seed]+ [round(f, 4) if type(f) == np.float64  else f for f in  list(test_results.values())]
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(full_list)  
            
    
    current_test_res = [a/len(inp_args.seed_array) for a in current_test_res]
    file_path = inp_args.common_dir + inp_args.avg_result_csv
    full_list = save_path_attributes+ [inp_args.seed]+ [round(f, 4) if type(f) == np.float64  else f for f in  current_test_res]
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(full_list) 
    return current_test_res


# In[3]:


max_test_res_over_distillation_ratios = [-1,-1,-1,-1]
max_ratio = 0.0
for ratio in inp_args.ratios:
    inp_args.bench_SaliencyKD_ratio = ratio
    save_path_attributes = [
        inp_args.dataset,
        inp_args.task_ratio,
        inp_args.hinton_loss_ratio,
        inp_args.memory_loss_ratio,
        inp_args.bench_rkd_dist_ratio,
        inp_args.bench_rkd_angle_ratio,
        inp_args.bench_fitnet_ratio,
        inp_args.bench_attention_ratio,
        inp_args.bench_temp_dist_ratio,
        inp_args.bench_DKD_ratio,
        inp_args.bench_VID_ratio,
        inp_args.bench_SaliencyKD_ratio
    ]
    current_test_res =  train_for_seed(inp_args, save_path_attributes)
    if current_test_res[1] > max_test_res_over_distillation_ratios[1]: #max entry selected based on avg_auc_prc
        max_test_res_over_distillation_ratios = current_test_res
        max_ratio = ratio

file_path = inp_args.common_dir + inp_args.max_result_csv
full_list = save_path_attributes+ [max_ratio]+ [round(100 * f,2) for f in  max_test_res_over_distillation_ratios]
with open(file_path, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(full_list) 


# In[ ]:




