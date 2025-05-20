import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sdtw_cuda_loss import SoftDTW
import numpy as np
from metric.utils import pdist, rkd_sdtw, return_relative_memory_diff_matrix,return_relative_memory_diff_matrix_z, temporal_rkd,reduce_channels,vid_loss, dkd_loss,conv1x1, generate_random_pairs,generate_start_end_pairs, generate_start_end_pairs_dynamic, generate_start_end_pairs_dynamic_stepwise


__all__ = ['L1Triplet', 'L2Triplet', 'ContrastiveLoss', 'RkdDistance', 'RKdAngle', 'HardDarkRank','TemporalRkdDistance','VIDLoss','DKDLoss']


#VID implementation
class VIDLoss(nn.Module):
    """
    Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation
    """

    def __init__(self, init_pred_var=5.0, eps = 1e-5, num_student_chan=16, num_teacher_chan=100):
        super().__init__()
        self.init_pred_var = init_pred_var
        self.eps = eps
        self.init_vid_modules(num_student_chan, num_teacher_chan)

    def init_vid_modules(self, s, t):
        self.regressor = nn.Sequential(
            conv1x1(s, t), nn.ReLU(), conv1x1(t, t), nn.ReLU(), conv1x1(t, t)
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(self.init_pred_var - self.eps) - 1.0) * torch.ones(t)
        )
        
    def get_extra_learnable_parameters(self):
        extra_parameters = list(self.regressor.parameters())+[self.log_scale]
        return extra_parameters

    def get_extra_parameters(self):
        num_p = 0
        for p in self.regressor.parameters():
            num_p += p.numel()
        return num_p

    def forward(self, student, teacher): #expect student =(batch_size, num_channels, height, width)
        # make compatible with conv2d regressor which expects 4 dimension at least(batch_size, num_channels, height, width)
        #(batch_size, num_of_time_steps, Channels_in)-->(batch_size, Channels_in, 1, num_of_time_steps)
        student = torch.swapaxes(student, 1,2)
        teacher = torch.swapaxes(teacher, 1,2)
       
        student = student.unsqueeze(2) 
        teacher = teacher.unsqueeze(2)
        loss_vid = vid_loss(
                self.regressor,
                self.log_scale,
                student,
                teacher,
                self.eps,
            )
        return loss_vid


"""Decoupled Knowledge Distillation(CVPR 2022)"""
class DKDLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=8.0, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, student, teacher, labels):
        loss= dkd_loss(student, teacher, labels, self.alpha, self.beta, self.temperature)
        return loss


class _Triplet(nn.Module):
    def __init__(self, p=2, margin=0.2, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.p = p
        self.margin = margin

        # update distance function accordingly
        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=(p==2))

        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        loss = F.triplet_margin_loss(anchor_embed, positive_embed, negative_embed,
                                     margin=self.margin, p=self.p, reduction='none')

        if not self.reduce:
            return loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class L2Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=2, margin=margin, sampler=sampler)


class L1Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=1, margin=margin, sampler=sampler)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__()
        self.margin = margin
        self.sampler = sampler

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        pos_loss = (F.pairwise_distance(anchor_embed, positive_embed, p=2)).pow(2)
        neg_loss = (self.margin - F.pairwise_distance(anchor_embed, negative_embed, p=2)).clamp(min=0).pow(2)

        loss = torch.cat((pos_loss, neg_loss))
        return loss.mean()


class HardDarkRank(nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack([torch.logsumexp(ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss


class FitNet(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.transform = nn.Conv2d(in_feature, out_feature, 1, bias=False)
        self.transform.weight.data.uniform_(-0.005, 0.005)

    def get_extra_learnable_parameters(self):
        extra_parameters = list(self.transform.parameters())
        return extra_parameters

    def forward(self, student, teacher):
        if student.dim() == 2: 
            student = student.unsqueeze(2).unsqueeze(3) # make compatible with conv2d layer which expects 4 dimension at least
            teacher = teacher.unsqueeze(2).unsqueeze(3)

        #nn.Conv2d expects its input to be  (batch_size, Channels_in, Height, Width) and outputs --> (batch_size, Channels_out, Height1, Width1) after convolution. For time series data we arrange input to be shape (batch_size, Channels_in, 1, num_of_time_steps)
        # in_feature = Channels_in = (student feature dimension usually smaller), out_feature = Channels_out = (teacher feature dimension usually larger)
        if student.dim() == 3:
            # print("student.dim() is three",student.size()) 
            # make compatible with conv2d layer which expects 4 dimension at least
            #(batch_size, num_of_time_steps, Channels_in)-->(batch_size, Channels_in, 1, num_of_time_steps)
            student = torch.swapaxes(student, 1,2)
            teacher = torch.swapaxes(teacher, 1,2)
           
            student = student.unsqueeze(2) 
            teacher = teacher.unsqueeze(2)
            # print("student.dim() after",student.size()) 

        return (self.transform(student) - teacher).pow(2).mean()


class AttentionTransfer(nn.Module): #expects shape [batch_size, channels, height, width]
    def forward(self, student, teacher):
        #for time series height set to 1, width=len_seq, since it fltten out last 2 dimensions i do not insert another dimension for height
         #student=teacher=(batch_size, num_of_time_steps, Channels_in)-->(batch_size, Channels_in, num_of_time_steps)
        student = torch.swapaxes(student, 1,2)
        teacher = torch.swapaxes(teacher, 1,2)
        
        s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))

        return (s_attention - t_attention).pow(2).mean()


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss

####DT2W : align student and teacher feature map with softDTW distance
class DT2W(nn.Module):
    def forward(self, student, teacher, gamma): #teacher -->  (batch_size, len_time_series, num_chan_teacher)
        with torch.no_grad():
            # Apply the channel reduction since num_chan_student < num_chan_teacher
            reduced_teacher = reduce_channels(teacher, student.size(2))
            
        sdtw = SoftDTW(use_cuda=True, gamma=gamma)
        # # Compute the loss value
        dt2w = sdtw(student, reduced_teacher)  # expects student and teacher in form  (batch_size, len_time_series, num_chan)
        return dt2w.mean()
    

####### added 
class SoftDtwRkdDistance(nn.Module):
    def forward(self, student, teacher, gamma):
        with torch.no_grad():
            dtw_teacher = rkd_sdtw(teacher, gamma)
            # mean_td = dtw_teacher[dtw_teacher>0].mean()
            mean_td = dtw_teacher.mean()#softdtw can be negative
            dtw_teacher = dtw_teacher / mean_td

        dtw_student = rkd_sdtw(student, gamma)
        # mean_ts = dtw_student[dtw_student>0].mean()
        mean_ts = dtw_student.mean()#softdtw can be negative
        dtw_student = dtw_student / mean_ts

        loss = F.smooth_l1_loss(dtw_student, dtw_teacher, reduction='mean')
        return loss

class TemporalRkdDistance(nn.Module):
    def forward(self, student, teacher):#STUDENT=(batch_size,seq_len,hidden_size) 
        with torch.no_grad():
            rkd_teacher = temporal_rkd(teacher)
            mean_td = rkd_teacher[rkd_teacher>0].mean()
            rkd_teacher = rkd_teacher / mean_td

        rkd_student = temporal_rkd(student)
        mean_ts = rkd_student[rkd_student>0].mean()
        rkd_student = rkd_student / mean_ts

        loss = F.smooth_l1_loss(rkd_student, rkd_teacher, reduction='mean')
        return loss
    
class MemoryDiscrepency(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            relative_memory_diff_teacher = return_relative_memory_diff_matrix(teacher) # batch_size * (seq_len-1)

        relative_memory_diff_student = return_relative_memory_diff_matrix(student)# batch_size * (seq_len-1)

        # Calculate loss for each sample in the batch and then take mean
        loss_per_sample = F.smooth_l1_loss(relative_memory_diff_student, relative_memory_diff_teacher,reduction='none') #apply quadratic loss for small differences and a linear loss for large differences. hence smooth than direct substraction (shape of loss_per_sample:# batch_size * (seq_len-1))
        # mean_loss_per_sample = torch.sum(loss_per_sample, dim=1) #shape: batch size
        mean_loss_per_sample = torch.mean(loss_per_sample, dim=1) #shape: batch size
        total_loss = torch.mean(mean_loss_per_sample)
        return total_loss

class MemoryDiscrepency_z(nn.Module):
    def forward(self, student, teacher, pairs=100, z='z'):
        _batch_size, seq_len, _hidden_dim = student.size()
        if z==1:
            sub_seq_ids_start, sub_seq_ids_end = generate_start_end_pairs_dynamic(z=1, num_pairs=pairs, min_value=0, max_value=seq_len-1)
        elif z==2:
            sub_seq_ids_start, sub_seq_ids_end = generate_start_end_pairs_dynamic(z=2, num_pairs=pairs, min_value=0, max_value=seq_len-1)
        else:
            sub_seq_ids_start, sub_seq_ids_end = generate_start_end_pairs_dynamic(z='z', num_pairs=pairs, min_value=0, max_value=seq_len-1)
        with torch.no_grad():
            relative_memory_diff_teacher = return_relative_memory_diff_matrix_z(teacher, sub_seq_ids_start, sub_seq_ids_end) # batch_size * num_sub_seq

        relative_memory_diff_student = return_relative_memory_diff_matrix_z(student, sub_seq_ids_start, sub_seq_ids_end)# batch_size * num_sub_seq

        # Calculate loss for each sample in the batch and then take mean
        loss_per_sample = F.smooth_l1_loss(relative_memory_diff_student, relative_memory_diff_teacher,reduction='none') #apply quadratic loss for small differences and a linear loss for large differences. hence smooth than direct substraction (shape of loss_per_sample:# batch_size * (seq_len-1))
        # mean_loss_per_sample = torch.sum(loss_per_sample, dim=1) #shape: batch size
        mean_loss_per_sample = torch.mean(loss_per_sample, dim=1) #shape: batch size
        total_loss = torch.mean(mean_loss_per_sample)
        return total_loss

# class MemoryDiscrepency_z(nn.Module):
#     def forward(self, student, teacher, pairs=100):
#         _batch_size, seq_len, _hidden_dim = student.size()
#         sub_seq_ids_start, sub_seq_ids_end = generate_start_end_pairs(num_pairs=pairs, min_value=0, max_value=seq_len-1)
#         with torch.no_grad():
#             relative_memory_diff_teacher = return_relative_memory_diff_matrix_z(teacher, sub_seq_ids_start, sub_seq_ids_end) # batch_size * num_sub_seq

#         relative_memory_diff_student = return_relative_memory_diff_matrix_z(student, sub_seq_ids_start, sub_seq_ids_end)# batch_size * num_sub_seq

#         # Calculate loss for each sample in the batch and then take mean
#         loss_per_sample = F.smooth_l1_loss(relative_memory_diff_student, relative_memory_diff_teacher,reduction='none') #apply quadratic loss for small differences and a linear loss for large differences. hence smooth than direct substraction (shape of loss_per_sample:# batch_size * (seq_len-1))
#         # mean_loss_per_sample = torch.sum(loss_per_sample, dim=1) #shape: batch size
#         mean_loss_per_sample = torch.mean(loss_per_sample, dim=1) #shape: batch size
#         total_loss = torch.mean(mean_loss_per_sample)
#         return total_loss
        

class MemoryDiscrepency_dynamic_stepwise(nn.Module):
    def forward(self, student, teacher, ep):
        _batch_size, seq_len, _hidden_dim = student.size()
        sub_seq_ids_start, sub_seq_ids_end = generate_start_end_pairs_dynamic_stepwise(ep, num_pairs=100, min_value=0, max_value=seq_len-1, ep_break=50)
        with torch.no_grad():
            relative_memory_diff_teacher = return_relative_memory_diff_matrix_z(teacher, sub_seq_ids_start, sub_seq_ids_end) # batch_size * num_sub_seq

        relative_memory_diff_student = return_relative_memory_diff_matrix_z(student, sub_seq_ids_start, sub_seq_ids_end)# batch_size * num_sub_seq

        # Calculate loss for each sample in the batch and then take mean
        loss_per_sample = F.smooth_l1_loss(relative_memory_diff_student, relative_memory_diff_teacher,reduction='none') #apply quadratic loss for small differences and a linear loss for large differences. hence smooth than direct substraction (shape of loss_per_sample:# batch_size * (seq_len-1))
        # mean_loss_per_sample = torch.sum(loss_per_sample, dim=1) #shape: batch size
        mean_loss_per_sample = torch.mean(loss_per_sample, dim=1) #shape: batch size
        total_loss = torch.mean(mean_loss_per_sample)
        return total_loss

# Sigmoid transition function for dynamic weights
def dynamic_weighting(epoch, warmup=300):
    if epoch < warmup:
        scale = 0.4*(warmup//3) # 0.4 * one third of warmup
        
        # Sigmoid transitions for w1 and w3
        w1 = 1 / (1 + torch.exp(torch.tensor((epoch - warmup // 3) / scale)))
        w3 = 1 - (1 / (1 + torch.exp(torch.tensor((epoch - 2 * warmup // 3) / scale))))
        
        # For w2, we combine two sigmoid functions to make it peak in the middle
        sigmoid_increase = 1 / (1 + torch.exp(torch.tensor((epoch - warmup // 3) / scale)))
        sigmoid_decrease = 1 / (1 + torch.exp(torch.tensor(-(epoch - 2 * warmup // 3) / scale)))
        w2 = sigmoid_increase * sigmoid_decrease  # Peak in the middle, 0 at both ends
        w2=0
        
        # Normalize weights to sum to 1
        total_weight = w1 + w2 + w3
        return w1 / total_weight, w2 / total_weight, w3 / total_weight
    else:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)
    
class MemoryDiscrepency_dynamic_z(nn.Module):
    def forward(self, student, teacher, ep):
        _batch_size, seq_len, _hidden_dim = student.size()
        sub_seq_ids_start_1, sub_seq_ids_end_1 = generate_start_end_pairs_dynamic(z=1, num_pairs=100, min_value=0, max_value=seq_len-1)
        # sub_seq_ids_start_2, sub_seq_ids_end_2 = generate_start_end_pairs_dynamic(z=2, num_pairs=100, min_value=0, max_value=seq_len-1)
        sub_seq_ids_start_z, sub_seq_ids_end_z = generate_start_end_pairs_dynamic(z='z', num_pairs=100, min_value=0, max_value=seq_len-1)
        
        with torch.no_grad():
            relative_memory_diff_teacher_1 = return_relative_memory_diff_matrix_z(teacher, sub_seq_ids_start_1, sub_seq_ids_end_1) # batch_size * num_sub_seq
            # relative_memory_diff_teacher_2 = return_relative_memory_diff_matrix_z(teacher, sub_seq_ids_start_2, sub_seq_ids_end_2) #
            relative_memory_diff_teacher_z = return_relative_memory_diff_matrix_z(teacher, sub_seq_ids_start_z, sub_seq_ids_end_z) # batch_size * num_sub_seq

        relative_memory_diff_student_1 = return_relative_memory_diff_matrix_z(student, sub_seq_ids_start_1, sub_seq_ids_end_1)# batch_size * num_sub_seq
        # relative_memory_diff_student_2 = return_relative_memory_diff_matrix_z(student, sub_seq_ids_start_2, sub_seq_ids_end_2)# batch_size * num_sub_seq
        relative_memory_diff_student_z = return_relative_memory_diff_matrix_z(student, sub_seq_ids_start_z, sub_seq_ids_end_z)# batch_size * num_sub_seq

        # Calculate loss for each sample in the batch and then take mean
        loss_per_sample_1 = F.smooth_l1_loss(relative_memory_diff_student_1, relative_memory_diff_teacher_1,reduction='none')
        mean_loss_per_sample_1 = torch.mean(loss_per_sample_1, dim=1) #shape: batch size
        total_loss_1 = torch.mean(mean_loss_per_sample_1)

        # loss_per_sample_2 = F.smooth_l1_loss(relative_memory_diff_student_2, relative_memory_diff_teacher_2,reduction='none')
        # mean_loss_per_sample_2 = torch.mean(loss_per_sample_2, dim=1) #shape: batch size
        # total_loss_2 = torch.mean(mean_loss_per_sample_2)


        # Calculate loss for each sample in the batch and then take mean
        loss_per_sample_z = F.smooth_l1_loss(relative_memory_diff_student_z, relative_memory_diff_teacher_z,reduction='none') 
        mean_loss_per_sample_z = torch.mean(loss_per_sample_z, dim=1) #shape: batch size
        total_loss_z = torch.mean(mean_loss_per_sample_z)

        w1, w2, w3 = dynamic_weighting(ep, warmup=300)
        total_loss = w3 * total_loss_1 + w1 * total_loss_z
        return total_loss
        
        