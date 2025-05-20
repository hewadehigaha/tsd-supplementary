#!/usr/bin/env python
# coding: utf-8

#package imports
from __future__ import annotations
import os
import argparse
import random
import json
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, average_precision_score
from matplotlib import pyplot as plt
from typing import cast, Any, Dict, List, Tuple, Optional
from torch.nn import KLDivLoss
import copy
from collections import Counter
import gc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
#util files
from utils.fcnbaseline import FCNBaseline, FCNBaselineSmall , LSTMClassifier,LSTMClassifierTeacherUnrolled
from metric.loss import  RkdDistance, RKdAngle, SoftDtwRkdDistance, MemoryDiscrepency, MemoryDiscrepency_z,TemporalRkdDistance, FitNet, DT2W, AttentionTransfer,DKDLoss,VIDLoss, SaliencyDiscrepency
from utils.sdtw_cuda_loss import SoftDTW



UCR_DATASETS = ['Haptics', 'Worms', 'Computers', 'UWaveGestureLibraryAll',
                'Strawberry', 'Car', 'BeetleFly', 'wafer', 'CBF', 'Adiac',
                'Lighting2', 'ItalyPowerDemand', 'yoga', 'Trace', 'ShapesAll',
                'Beef', 'MALLAT', 'MiddlePhalanxTW', 'Meat', 'Herring',
                'MiddlePhalanxOutlineCorrect', 'FordA', 'SwedishLeaf',
                'SonyAIBORobotSurface', 'InlineSkate', 'WormsTwoClass', 'OSULeaf',
                'Ham', 'uWaveGestureLibrary_Z', 'NonInvasiveFatalECG_Thorax1',
                'ToeSegmentation1', 'ScreenType', 'SmallKitchenAppliances',
                'WordsSynonyms', 'MoteStrain', 'synthetic_control', 'Cricket_X',
                'ECGFiveDays', 'Wine', 'Cricket_Y', 'TwoLeadECG', 'Two_Patterns',
                'Phoneme', 'MiddlePhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
                'DistalPhalanxTW', 'FacesUCR', 'ECG5000', '50words', 'HandOutlines',
                'Coffee', 'Gun_Point', 'FordB', 'InsectWingbeatSound', 'MedicalImages',
                'Symbols', 'ArrowHead', 'ProximalPhalanxOutlineAgeGroup',
                'SonyAIBORobotSurfaceII', 'ChlorineConcentration', 'Plane', 'Lighting7',
                'PhalangesOutlinesCorrect', 'ShapeletSim', 'DistalPhalanxOutlineAgeGroup',
                'uWaveGestureLibrary_X', 'FaceFour', 'RefrigerationDevices', 'ECG200',
                'ToeSegmentation2', 'CinC_ECG_torso', 'BirdChicken', 'OliveOil',
                'LargeKitchenAppliances', 'uWaveGestureLibrary_Y',
                'NonInvasiveFatalECG_Thorax2', 'FISH', 'ProximalPhalanxOutlineCorrect',
                'Cricket_Z', 'FaceAll', 'StarLightCurves', 'ElectricDevices', 'Earthquakes',
                'DiatomSizeReduction', 'ProximalPhalanxTW']

@dataclass
class InputData:
    x: torch.Tensor
    y: torch.Tensor

    def split(self, split_size: float) -> Tuple[InputData, InputData]:
        train_x, val_x, train_y, val_y = train_test_split(
            self.x.numpy(), self.y.numpy(), test_size=split_size, stratify=self.y
        )
        return (InputData(x=torch.from_numpy(train_x), y=torch.from_numpy(train_y)),
                InputData(x=torch.from_numpy(val_x), y=torch.from_numpy(val_y)))

#####################
@staticmethod
def _to_1d_binary(y_true: np.ndarray, y_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(y_true.shape) > 1:
        return np.argmax(y_true, axis=-1), np.argmax(y_preds, axis=-1)

    else:
        return y_true, (y_preds > 0.5).astype(int)

#####################
def load_ucr_data(data_path: Path,
                  encoder: Optional[OneHotEncoder] = None
                  ) -> Tuple[InputData, InputData, OneHotEncoder]:

    experiment = data_path.parts[-1]

    train = np.loadtxt(data_path / f'{experiment}_TRAIN', delimiter=',')
    test = np.loadtxt(data_path / f'{experiment}_TEST', delimiter=',')
    
    if encoder is None:
        encoder = OneHotEncoder(categories='auto', sparse_output=False)
        y_train = encoder.fit_transform(np.expand_dims(train[:, 0], axis=-1))
    else:
        y_train = encoder.transform(np.expand_dims(train[:, 0], axis=-1))
    y_test = encoder.transform(np.expand_dims(test[:, 0], axis=-1))

    # if y_train.shape[1] == 2:
    #     # there are only 2 classes, so there only needs to be one
    #     # output
    #     y_train = y_train[:, 0]
    #     y_test = y_test[:, 0]

    # UCR data is univariate, so an additional dimension is added at index 1 to make it of shape (N, Channels, Length) as the model expects
    train_x = torch.from_numpy(train[:, 1:]).unsqueeze(1).float()
    test_x = torch.from_numpy(test[:, 1:]).unsqueeze(1).float()

    [train_size, n_chan, seq_len] = list(train_x.size())
    [test_size, n_chan, seq_len] = list(test_x.size())
    
    class_distribution = dict(Counter(train[:, 0]))
    print("database name---",opts.dataset,  "class distribution----:", class_distribution)
    opts.output_classes= len(class_distribution)
    # if opts.output_classes == 2:
    #     opts.output_classes =1
        
    #downsmaple the signals
    if (seq_len>opts.downsample_size):
        train_x = F.interpolate(train_x, size=opts.downsample_size, mode='linear',align_corners=True)
        test_x = F.interpolate(test_x, size=opts.downsample_size, mode='linear',align_corners=True)
    print("downsmaple from size:", seq_len, ", to a new size:", opts.downsample_size)

    #swapp axes for LSTM : input tesnsor should be in shape : (batch, seq_len, channels)
    train_x = torch.swapaxes(train_x, 1,2)
    test_x = torch.swapaxes(test_x, 1,2)
    
    train_input = InputData(x=train_x,
                            y=torch.from_numpy(y_train))
    test_input = InputData(x=test_x,
                           y=torch.from_numpy(y_test))
    return train_input, test_input, encoder


#############################
def _load_data(experiment, data_folder, encoder) -> Tuple[InputData, InputData]:
    assert experiment in UCR_DATASETS, \
        f'{experiment} must be one of the UCR datasets: ' \
        f'https://www.cs.ucr.edu/~eamonn/time_series_data/'
    experiment_datapath = data_folder / 'UCR_TS_Archive_2015' / experiment
    if encoder is None:
        train, test, encoder = load_ucr_data(experiment_datapath)
    else:
        train, test, _ = load_ucr_data(experiment_datapath, encoder=encoder)
    return train, test
    

############################
def get_loaders(batch_size: int, mode: str, experiment: str, data_folder: str, encoder: Optional[str] = None,
                val_size: Optional[float] = None) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Return dataloaders of the training / test data

    Arguments
    ----------
    batch_size:
        The batch size each iteration of the dataloader should return
    mode: {'train', 'test', 'both'}
        If 'train', this function should return (train_loader, val_loader)
        If 'test', it should return (test_loader, None)
        If 'both', it souhld return all loaders
    val_size:
        If mode == 'train', the fraction of training data to use for validation
        Ignored if mode == 'test'

    Returns
    ----------
    Tuple of (train_loader, val_loader) if mode == 'train'
    Tuple of (test_loader, None) if mode == 'test'
    Tuple of (train_loader, val_loader, test_loader) if mode == 'both'
    """
    train_data, test_data = _load_data(experiment, data_folder, encoder)

    assert val_size is not None, 'Val size must be defined when loading training data'
    train_data, val_data = train_data.split(val_size)

    train_loader = DataLoader(
        TensorDataset(train_data.x, train_data.y),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    val_loader = DataLoader(
        TensorDataset(val_data.x, val_data.y),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )
    test_loader = DataLoader(
        TensorDataset(test_data.x, test_data.y),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )
    if mode == 'train':
        return train_loader, val_loader
    elif mode == 'test':
        return test_loader, None
    else:
        return train_loader, val_loader, test_loader, train_data


#######################
def train(teacher, student,  optimizer, lr_scheduler, loader, ep, train_data):
    
    student.train()
    teacher.eval()

    hinton_loss_all = []
    dist_loss_all = []
    dtw_loss_all = []
    memory_loss_all = []
    task_loss_all = [] #cross entrophy loss here
    loss_all = []

    #remove one-hot encodings of label vectors
    whole_train_y = train_data.y.argmax(dim=1) 

    # train_iter = tqdm(loader)
    train_iter = loader
    for timeseries, labels in train_iter:
        timeseries, labels = timeseries.cuda(), labels.cuda()

        with torch.no_grad():
            t_linear, memory_t, hn_t, cn_t = teacher(timeseries, True)# memory_t=(batch_size,seq_len,hidden_size) 
            # t_linear, memory_t1, memory_t2, memory_t = teacher(timeseries, True)# memory_t=(batch_size,seq_len,hidden_size)
            memory_t.requires_grad_(False)
            mem_reduced_t = memory_t.reshape(timeseries.size(0), -1)
            # print("mem_reduced_t size",mem_reduced_t.size())
            mem_reduced_t.requires_grad_(False)

        s_linear, memory_s, hn_s, cn_s = student(timeseries, True)# memory_s=(batch_size,seq_len,hidden_size) 
        memory_s.requires_grad_()
        mem_reduced_s = memory_s.reshape(timeseries.size(0), -1)
        # print("mem_reduced_s size",mem_reduced_s.size())
  
        if len(labels.shape) == 1:
                t_loss = F.binary_cross_entropy_with_logits(
                    s_linear, labels.unsqueeze(-1).float(), reduction='mean'
                )
        else:
            t_loss = F.cross_entropy(s_linear, labels.argmax(dim=-1), reduction='mean')
        
        task_loss = opts.task_ratio * t_loss

        if opts.hinton_loss_ratio:
            #hinton's KD loss
            KD_loss = KLDivLoss(reduction= "batchmean")(F.log_softmax(s_linear/opts.temperature, dim=1),
                                 F.softmax(t_linear/opts.temperature, dim=1)) * (opts.temperature * opts.temperature)
            hinton_loss = opts.hinton_loss_ratio * KD_loss
        else:
            hinton_loss = torch.tensor([0.0]).cuda()

        
        if opts.bench_SaliencyKD_ratio:
            #remove one-hot encodings of label vectors
            batch_labels = labels.argmax(dim=1)  
            bench_saliencyKD_loss = opts.bench_SaliencyKD_ratio * saliencyKD_criterion(student, teacher, timeseries, batch_labels, train_data.x.cuda(), whole_train_y.cuda(), opts.num_samples, opts.sub_seq_len,opts.temperature)
        else:
            bench_saliencyKD_loss = torch.tensor([0.0]).cuda()

        
        if opts.memory_loss_ratio:
            
            memory_loss =  opts.memory_loss_ratio * (hidden_state_criterion(memory_s, memory_t)+hidden_state_criterion_z(memory_s, memory_t))
            # memory_loss = opts.memory_loss_ratio * hidden_state_criterion(memory_s, memory_t)+opts.memory_loss_ratio * hidden_state_criterion(memory_s, memory_t1) + opts.memory_loss_ratio * hidden_state_criterion(memory_s, memory_t2)
        else:
            memory_loss = torch.tensor([0.0]).cuda()

        if opts.dtw_rkd_ratio:
            dtw_rkd_loss = opts.dtw_rkd_ratio * (dtw_rkd_criterion(memory_s, memory_t, opts.dtw_gamma))
        else:
            dtw_rkd_loss = torch.tensor([0.0]).cuda()

        if opts.dist_ratio:
            # dist_loss = opts.dist_ratio * dist_criterion(mem_reduced_s, mem_reduced_t)
            dist_loss = opts.dist_ratio * dist_criterion(memory_s, memory_t)
        else:
            dist_loss=torch.tensor([0.0]).cuda()

        #benchmark approcehs
        if opts.bench_rkd_dist_ratio:
            bench_rkd_dist_loss = opts.bench_rkd_dist_ratio * dist_criterion_bench(mem_reduced_s, mem_reduced_t)
        else:
            bench_rkd_dist_loss = torch.tensor([0.0]).cuda()

        if opts.bench_rkd_angle_ratio:
            bench_rkd_angle_loss = opts.bench_rkd_angle_ratio * angle_criterion_bench(mem_reduced_s, mem_reduced_t)
        else:
            bench_rkd_angle_loss = torch.tensor([0.0]).cuda()

        #train Fitnets in 2 stages, first set all other ratios including task_ratio to be zero except bench_fitnet_ratio.
        #Above stage intialize student upto guided layer
        #then fine tune the student with only task loss in second stage, set opts.load=True to load the fitnet weights from 1st stage
        if opts.bench_fitnet_ratio:
            bench_fitnet_loss = opts.bench_fitnet_ratio * fitnet_criterion(memory_s, memory_t)
        else:
            bench_fitnet_loss = torch.tensor([0.0]).cuda()

        if opts.bench_attention_ratio:
            bench_attention_loss = opts.bench_attention_ratio * attention_criterion(memory_s, memory_t)
        else:
            bench_attention_loss = torch.tensor([0.0]).cuda()

        if opts.bench_temp_dist_ratio:
            bench_temp_dist_loss = opts.bench_temp_dist_ratio * temp_dist_criterion(memory_s, memory_t,opts.dtw_gamma)
        else:
            bench_temp_dist_loss = torch.tensor([0.0]).cuda()

        if opts.bench_DKD_ratio:
            bench_DKD_loss = opts.bench_DKD_ratio * DKD_criterion(s_linear, t_linear, labels)
        else:
            bench_DKD_loss = torch.tensor([0.0]).cuda()

        if opts.bench_VID_ratio:
            bench_VID_loss = opts.bench_VID_ratio * VID_criterion(memory_s, memory_t)
        else:
            bench_VID_loss = torch.tensor([0.0]).cuda()
    
        loss = task_loss + memory_loss + dtw_rkd_loss + dist_loss + hinton_loss + bench_rkd_angle_loss + bench_rkd_dist_loss+ bench_fitnet_loss+bench_attention_loss+bench_temp_dist_loss+bench_DKD_loss+bench_VID_loss+bench_saliencyKD_loss
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        task_loss_all.append(task_loss.item())
        memory_loss_all.append(memory_loss.item())
        dist_loss_all.append(bench_rkd_angle_loss.item())  
        dtw_loss_all.append(bench_VID_loss.item())
        hinton_loss_all.append(hinton_loss.item())
        loss_all.append(loss.item())

        # train_iter.set_description("[Train][Epoch %d] Task: %.5f, Dist: %.5f, Angle: %.5f" %
                                   # (ep, task_loss.item(), dtw_loss1.item(), dtw_loss1.item()))
        
    mean_epoch_train_loss = torch.Tensor(loss_all).mean()
    mean_epoch_task_loss = torch.Tensor(task_loss_all).mean()
    mean_epoch_memory_loss = torch.Tensor(memory_loss_all).mean()
    mean_epoch_dist_loss = torch.Tensor(dist_loss_all).mean()
    mean_epoch_dtw_loss = torch.Tensor(dtw_loss_all).mean()
    mean_epoch_hinton_loss = torch.Tensor(hinton_loss_all).mean()

    # print('[Epoch %d] Loss: %.5f, Task: %.5f, Memory: %.5f,Dist: %.5f,DTW: %.5f,Hinton: %.5f\n' %\
    #       (ep, mean_epoch_train_loss, mean_epoch_task_loss,
    #       mean_epoch_memory_loss,mean_epoch_dist_loss, 
    # mean_epoch_dtw_loss, mean_epoch_hinton_loss))
    return mean_epoch_train_loss, mean_epoch_task_loss, mean_epoch_memory_loss, mean_epoch_dist_loss, mean_epoch_dtw_loss, mean_epoch_hinton_loss

###########################
def eval(net, loader, ep):
    net.eval()
    # test_iter = tqdm(loader)
    test_iter=loader
    outputs_all, labels_all = [], []
    test_results: Dict[str, float] = {}

    with torch.no_grad():
        for timeseries, labels in test_iter:
            timeseries, labels = timeseries.cuda(), labels.cuda()
            outputs = net(timeseries)
            
            if len(labels.shape) == 1:
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=-1)
            outputs_all.append(outputs.cpu().detach().numpy())
            labels_all.append(labels.cpu().detach().numpy()) 
            

    true_np, preds_np = np.concatenate(labels_all), np.concatenate(outputs_all)

    test_results['roc_auc_score'] = roc_auc_score(true_np, preds_np)

    precision = dict()
    recall = dict()
    auc_pr = dict()

    if opts.output_classes < 2:
        for i in range(opts.output_classes):
            precision[i], recall[i], _ = precision_recall_curve(true_np, preds_np)
            auc_pr[i] = auc(recall[i], precision[i])
    else:
        for i in range(opts.output_classes):
            precision[i], recall[i], _ = precision_recall_curve(true_np[:, i], preds_np[:, i])
            auc_pr[i] = auc(recall[i], precision[i])
        
    average_auc_pr = sum(auc_pr.values()) / opts.output_classes
    test_results['average_auc_pr'] = average_auc_pr
    average_precision = average_precision_score(true_np, preds_np, average="micro")
    test_results['average_precision'] = average_precision
    test_results['accuracy_score'] = accuracy_score(
       *_to_1d_binary(true_np, preds_np)
    )
    return test_results

def validate(net, loader, ep):
    net.eval()
    epoch_val_loss = []
    # val_iter = tqdm(loader, ncols=80)
    val_iter=loader
    for timeseries, labels in val_iter:
        timeseries, labels = timeseries.cuda(), labels.cuda()
        with torch.no_grad():
            output = net(timeseries)
            if len(labels.shape) == 1:
                val_loss = F.binary_cross_entropy_with_logits(
                    output, labels.unsqueeze(-1).float(), reduction='mean'
                ).item()
            else:
                val_loss = F.cross_entropy(output,
                                           labels.argmax(dim=-1), reduction='mean').item()
            epoch_val_loss.append(val_loss)
            
    mean_epoch_val_loss = np.mean(epoch_val_loss)
    return mean_epoch_val_loss

##############################
def plot_train_history(train_loss, val_loss, task_loss, mem_loss,  hinton_loss, dtw_loss, dist_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss,label='val_loss')
    plt.plot(task_loss,label='task_loss(CE)')
    plt.plot(mem_loss,label='mem_loss')
    plt.plot(hinton_loss,label='hinton_loss')
    plt.plot(dtw_loss,label='dtw_loss')
    plt.plot(dist_loss,label='dist_loss')
    plt.legend()
    plt.show




#####################################################
def run_for_seed(inp_args):
    global opts
    opts = inp_args

    #set randomness fixed again
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
        set_random_seed(opts.seed)

    global fitnet_criterion, DKD_criterion, VID_criterion,dist_criterion_bench, angle_criterion_bench, dist_criterion, hidden_state_criterion, hidden_state_criterion_z, angle_criterion ,dtw_rkd_criterion,attention_criterion, temp_dist_criterion, saliencyKD_criterion
    fitnet_criterion = FitNet(opts.student_hidden_size, opts.teacher_hidden_size).cuda()
    DKD_criterion = DKDLoss(opts.alpha_DKD, opts.beta_DKD, opts.temperature)
    VID_criterion = VIDLoss(opts.init_pred_var_VID, opts.eps_VID , opts.student_hidden_size, opts.teacher_hidden_size).cuda()
    dist_criterion_bench = RkdDistance()
    angle_criterion_bench = RKdAngle()
    dist_criterion = TemporalRkdDistance()
    angle_criterion = RKdAngle()
    dtw_rkd_criterion = SoftDtwRkdDistance()
    hidden_state_criterion = MemoryDiscrepency()
    hidden_state_criterion_z = MemoryDiscrepency_z()
    attention_criterion = AttentionTransfer()
    temp_dist_criterion = DT2W()
    saliencyKD_criterion = SaliencyDiscrepency()

    loader_train_sample, loader_train_eval, loader_eval, train_data= get_loaders(batch_size= opts.batch, mode ='both',
                val_size=opts.val_size, experiment=opts.dataset, data_folder=opts.data_folder)


    student = LSTMClassifier(input_dim=opts.n_channels, hidden_dim=opts.student_hidden_size, layer_dim=opts.student_num_layers, output_dim=opts.output_classes).cuda()#input_dim, hidden_dim, layer_dim, output_dim
    if opts.load is not None:
        #for fitnet second stage, to load model from first stage(task=0, fitnet=1)
        # opts.load=opts.save_dir + opts.dataset + '_' + str(opts.hinton_loss_ratio) + '_' + str(opts.memory_loss_ratio)+ '_'+  str(opts.bench_rkd_dist_ratio) + '_'+ str(opts.bench_rkd_angle_ratio) +'_1_'+ str(opts.bench_attention_ratio) + '_' + str(opts.bench_temp_dist_ratio)+'/'+ '/best.pth'
        opts.load = (f"{opts.save_dir}{opts.dataset}_0.0_{opts.hinton_loss_ratio}_"
                f"{opts.memory_loss_ratio}_{opts.bench_rkd_dist_ratio}_"
                f"{opts.bench_rkd_angle_ratio}_1.0_{opts.bench_attention_ratio}_"
                f"{opts.bench_temp_dist_ratio}_{opts.bench_DKD_ratio}_{opts.bench_VID_ratio}/best.pth")
        student.load_state_dict(torch.load(opts.load))
        print("Loaded Model from %s for student" % opts.load)


    teacher= LSTMClassifier(input_dim=opts.n_channels, hidden_dim=opts.teacher_hidden_size, layer_dim=opts.teacher_num_layers, output_dim=opts.output_classes).cuda()
    teacher.load_state_dict(torch.load(opts.teacher_load))

    student = student.cuda()
    teacher = teacher.cuda()

    learnable_params = list(student.parameters())
    if opts.bench_VID_ratio:
        learnable_params += VID_criterion.get_extra_learnable_parameters()

    optimizer = optim.Adam(learnable_params, lr=opts.lr, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)


    train_loss: List[float] = []
    val_loss: List[float] = []
    task_loss: List[float] = []
    mem_loss: List[float] = []
    hinton_loss: List[float] = []
    dtw_loss: List[float] = []
    dist_loss: List[float] = []
    
    best_val_loss = np.inf
    patience_counter = 0
    best_state_dict = None
    best_epoch= 0
    best_test_auc =0
    best_test_auc_epoch=0


    # validation and test results from teacher model
    print("validation and test results for teacher model-------")
    # print(eval(teacher, loader_train_eval, 0))
    print(eval(teacher, loader_eval, 0))

    # initial validation and test results from student model before training
    # print("initial validation and test results from student model before training")
    # print(eval(student, loader_train_eval, 0))
    # print(eval(student, loader_eval, 0))

    #model training
    for epoch in range(1, opts.epochs+1):
        mean_epoch_train_loss, mean_epoch_task_loss, mean_epoch_memory_loss, mean_epoch_dist_loss, mean_epoch_dtw_loss, mean_epoch_hinton_loss = train(teacher, student, optimizer, lr_scheduler, loader_train_sample, epoch, train_data)
        
        mean_epoch_val_loss = validate(student, loader_train_eval, epoch)
        
        test_results = eval(student, loader_eval, epoch)
        if test_results['roc_auc_score'] > best_test_auc: #only for information, not used for desicion making
            best_test_auc = test_results['roc_auc_score'] 
            best_test_auc_epoch = epoch
    
        train_loss.append(mean_epoch_train_loss)
        val_loss.append(mean_epoch_val_loss)
        task_loss.append(mean_epoch_task_loss)
        mem_loss.append(mean_epoch_memory_loss)
        hinton_loss.append(mean_epoch_hinton_loss)
        dtw_loss.append(mean_epoch_dtw_loss)
        dist_loss.append(mean_epoch_dist_loss)
    
    
        # print(f'Epoch: {epoch}, '
        #       f'Train loss: {train_loss[-1]}, '
        #       f'Val loss: {val_loss[-1]}')
    
            
        if val_loss[-1] < best_val_loss:
            best_val_loss = val_loss[-1]
            best_state_dict = copy.deepcopy(student.state_dict())
            best_epoch= epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == opts.patience:
                if best_state_dict is not None:
                    if opts.last_weight is None:
                        print('Loading best weights----!')
                        student.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict)) #load best weight, other wise last weights
                print('Early stopping!')
                break
    
    #in-case of patience+bestepoch > total_epoch
    student.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict))

    val_results = eval(student, loader_train_eval, epoch)
    print("validation results:--", val_results)
    print("best_epoch for validation loss----:", best_epoch, "best_test_auc----", best_test_auc, "best_test_auc_epoch---",best_test_auc_epoch)
    test_results = eval(student, loader_eval, epoch)
    print("final results:--", test_results, "seed:---", opts.seed)
    # plt.figure()
    # plot_train_history(train_loss, val_loss, task_loss, mem_loss, hinton_loss,dtw_loss,dist_loss)
    # plot_feature_space(student, loader_eval, 'plots/11.pdf', method='tsne')
    # plot_feature_space(teacher, loader_eval, 'plots/teacher1.pdf', method='tsne')
    # plot_hidden_state_heatmaps(student, loader_eval, 'plots/11.pdf')
    # plot_hidden_state_heatmaps(teacher, loader_eval, 'plots/11.pdf')
    return test_results , student  


def plot_feature_space(net, loader, path, method='tsne'):
    net.eval()

    all_memory = []
    all_labels = []

    with torch.no_grad():
        for timeseries, labels in loader:
            timeseries, labels = timeseries.cuda(), labels.cuda()
            # Forward pass to get the outputs and memory
            outputs, memory, hn, cn = net(timeseries, True)

            # plot_hidden_states_as_heatmap(memory)
            # Reshape memory: (batch_size, seq_len, feature_size) -> (batch_size * seq_len, feature_size)
            batch_size, seq_len, feature_size = memory[:,-6:-1,:].shape
            memory_reshaped = memory[:,-6:-1,:].reshape(batch_size * seq_len, feature_size)
            
            # Store memory and labels
            all_memory.append(memory_reshaped.cpu().numpy())
            labels_class = torch.argmax(labels, dim=1)
            # Repeat labels to match memory reshaped size
            labels_repeated = labels_class.repeat_interleave(seq_len)
            all_labels.append(labels_repeated.cpu().numpy())

    # Stack all memory and labels into arrays
    all_memory = np.concatenate(all_memory, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    #Apply dimensionality reduction (t-SNE or PCA)
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Invalid method! Choose 'tsne' or 'pca'")
    
    memory_2d = reducer.fit_transform(all_memory)

    # Plot the reduced memory in 2D feature space
    plt.figure(figsize=(8, 6))
    # Define 8 highly contrasting colors
    contrasting_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
                          '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    
    # Create a ListedColormap
    cmap = mcolors.ListedColormap(contrasting_colors)
    scatter = plt.scatter(memory_2d[:, 0], memory_2d[:, 1], c=all_labels, cmap=cmap, s=50)
    plt.savefig(path, format='pdf')
    # plt.title(f'Feature Space Visualization (Memory) - {method.upper()}')
    plt.show()


def plot_hidden_state_heatmaps(net, loader, path):
    net.eval()
    whole_memory=[]
    with torch.no_grad():
        for timeseries, labels in loader:
            timeseries, labels = timeseries.cuda(), labels.cuda()
            # Forward pass to get the outputs and memory
            outputs, memory, hn, cn = net(timeseries, True)
            whole_memory.append(memory.cpu().numpy()) 

    # Stack all memory into arrays
    whole_memory = np.concatenate(whole_memory, axis=0)
    plot_hidden_states_stacked(whole_memory)
    
def plot_hidden_states_stacked(hidden_states):
    """
    Plots the hidden states for the entire test set stacked in the hidden size dimension as a heatmap.
    
    Args:
        hidden_states (torch.Tensor): Tensor of shape (test_size, seq_len, hidden_size)
    """
    # Reshape hidden states: (batch_size, seq_len, hidden_size) -> (seq_len, batch_size * hidden_size)
    test_size, seq_len, hidden_size = hidden_states.shape
    
    # Reshape to stack in the hidden size dimension
    hidden_states_stacked = hidden_states.reshape(seq_len, test_size * hidden_size)

    # Plot heatmap using matplotlib
    plt.figure(figsize=(12, 6))  # Adjust width for more hidden states
    plt.imshow(hidden_states_stacked.T, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # Add labels and title
    plt.colorbar(label='Hidden State Activation')
    plt.xlabel('Time Steps')
    plt.ylabel('Stacked Hidden Units (Batch x Hidden Size)')
    plt.title('Hidden States Stacked in Hidden Size Dimension')
    
    plt.show()




 







