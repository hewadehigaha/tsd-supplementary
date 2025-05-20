import torch
from utils.sdtw_cuda_loss import SoftDTW
import torch.nn.functional as F

__all__ = ['pdist', 'rkd_sdtw']


def conv1x1(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=1, padding=0, bias=False, stride=stride
    )

def vid_loss(regressor, log_scale, f_s, f_t, eps=1e-5):
    # pool for dimentsion match
    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    else:
        pass
    pred_mean = regressor(f_s)
    pred_var = torch.log(1.0 + torch.exp(log_scale)) + eps
    pred_var = pred_var.view(1, -1, 1, 1).to(pred_mean)
    neg_log_prob = 0.5 * ((pred_mean - f_t) ** 2 / pred_var + torch.log(pred_var))
    loss = torch.mean(neg_log_prob)
    return loss

def dkd_loss_non_reduced(logits_input, logits_target, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask_onehot(logits_input, target)
    other_mask = _get_other_mask_onehot(logits_input, target)
    pred_input = F.softmax(logits_input / temperature, dim=-1)
    pred_target = F.softmax(logits_target / temperature, dim=-1)
    pred_input = cat_mask(pred_input, gt_mask, other_mask)
    pred_target = cat_mask(pred_target, gt_mask, other_mask)
    log_pred_input = torch.log(pred_input)

    tckd_loss = (
        F.kl_div(log_pred_input, pred_target, reduction='none')
        * (temperature**2)
    )
    tckd_loss = tckd_loss.sum(dim=1)# take element-wise sum per each input sample
    
    pred_target_part2 = F.softmax(
        logits_target / temperature - 1000.0 * gt_mask, dim=-1
    ).clamp(min=1e-10) #avoid Nan values
    log_pred_input_part2 = F.log_softmax(
        logits_input / temperature - 1000.0 * gt_mask, dim=-1
    )
    nckd_loss = (
        F.kl_div(log_pred_input_part2, pred_target_part2, reduction='none')
        * (temperature**2)
    )
    nckd_loss = nckd_loss.sum(dim=1) # take element-wise sum per each input sample
    
    # print("tckd_loss.size()------", tckd_loss)
    # print("nckd_loss.size()------", nckd_loss)
    return  alpha * tckd_loss + beta * nckd_loss


def _get_top_n_mask_onehot(logits, n):  # for one-hot labels
    # Get the indices of the maximum n class activations
    _, mx_indices = torch.topk(logits, n, largest=True)  # Get indices of n largest logits

    # Initialize the mask with zeros
    mask = torch.zeros_like(logits, dtype=torch.float)

    # Set the max n class activations to one
    mask.scatter_(1, mx_indices, 1)  # Mask out the max n indices

    return mask


def dkd_loss_top_n(logits_input, logits_target, target, temperature, n):
    gt_mask = _get_top_n_mask_onehot(logits_target, n)
    other_mask = ~gt_mask.bool()
    pred_input = F.softmax(logits_input / temperature, dim=-1)
    pred_target = F.softmax(logits_target / temperature, dim=-1)
    pred_input = cat_mask(pred_input, gt_mask, other_mask)
    pred_target = cat_mask(pred_target, gt_mask, other_mask)
    log_pred_input = torch.log(pred_input)

    tckd_loss = (
        F.kl_div(log_pred_input, pred_target, reduction='none')
        * (temperature**2)
    )
    tckd_loss = tckd_loss.sum(dim=1)# take element-wise sum per each input sample

    return tckd_loss 
    
def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss
    
def _get_gt_mask(logits, target): #for one hot labels
    return target.bool()
    
def _get_other_mask(logits, target): #for one hot labels
    return ~target.bool()

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
    
def _get_gt_mask_onehot(logits, target):#for single class targets
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask_onehot(logits, target):#for class targets
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask



def reduce_channels(teacher, num_chan_s): #randomly selects num_chan_s number of channels
    # Randomly select channels
    selected_indices = torch.randperm(teacher.size(2))[:num_chan_s]
    reduced_teacher = teacher[:, :, selected_indices]
    return reduced_teacher


def rkd_sdtw(f_map, gamma): # f_map shape: batch size * timeseq_len * n_channels(or dimensions)
    [bathc_size, seq_len, n_chan] = f_map.size()
    f_map_square = f_map.repeat(bathc_size,1,1).view(bathc_size, bathc_size, seq_len, n_chan)
    f_map_square_transpose = torch.swapaxes(f_map_square, 0, 1)
    upper_triangular_indices = torch.triu_indices(bathc_size,bathc_size, offset=1)
    x = f_map_square_transpose[[upper_triangular_indices[0], upper_triangular_indices[1]]]
    y = f_map_square[[upper_triangular_indices[0], upper_triangular_indices[1]]]
    sdtw = SoftDTW(use_cuda=True, gamma=gamma)
    # # Compute the loss value
    rkd_sdtw = sdtw(x, y)  # Just like any torch.nn.xyzLoss()
    return rkd_sdtw

def temporal_rkd(f_map):
    [bathc_size, seq_len, n_chan] = f_map.size()#f_map=(batch_size,seq_len,hidden_size) for LSTM
    f_map_square = f_map.repeat(bathc_size,1,1).view(bathc_size, bathc_size, seq_len, n_chan)
    f_map_square_transpose = torch.swapaxes(f_map_square, 0, 1)
    upper_triangular_indices = torch.triu_indices(bathc_size,bathc_size, offset=1)
    x = f_map_square_transpose[[upper_triangular_indices[0], upper_triangular_indices[1]]]
    y = f_map_square[[upper_triangular_indices[0], upper_triangular_indices[1]]]
    # # Compute the loss value
    rkd_sdtw = euclidian_dist_2_D(x, y)  # Just like any torch.nn.xyzLoss()
    return rkd_sdtw

def euclidian_dist_2_D(x, y): #x=(num_pairs_for_batch * seq_len * ouput_chan)
    # x=torch.swapaxes(x, 1,2) # x= num_pairs_for_batch  * ouput_chan * seq_len
    # y=torch.swapaxes(y, 1,2) # not needed results are same 
    x1=torch.flatten(x, start_dim=1, end_dim=-1)
    y1=torch.flatten(y, start_dim=1, end_dim=-1)
    euc_dist = torch.cdist(torch.unsqueeze(x1, 1), torch.unsqueeze(y1, 1), p=2)
    return torch.flatten(euc_dist)#size = num_pairs_for_batch

    

#calculate influence of x_t on hidden states, given the {x_0, ..,x_(t-1)} occured
def return_relative_memory_diff_matrix(memory_map):
    differences = memory_map[:, 1:, :] - memory_map[:, :-1, :] #difference along seq_len dimension,  H_(t+1) - H_(t)
    diff =torch.linalg.norm(differences, dim=2) # take norm of defference vector , diff = batch_size * (seq_len-1)
    current_hidden_state_norm = torch.linalg.norm(memory_map[:, :-1, :], dim=2) # ||H_(t)||
    relative_diff = diff/current_hidden_state_norm # take the change between consecutive hidden states relative
                                                   #to the first hidden state of each pairs 
                                                   #: {H_(t+1) - H_(t) } / ||H_(t)|| :,  relative hidden state change at each time step
    return relative_diff

#calculate influence of x_t:t+z on hidden states, given the {x_0, ..,x_(t-1)} occured
def return_relative_memory_diff_matrix_z(memory_map, sub_seq_ids_start, sub_seq_ids_end):
    differences = memory_map[:, sub_seq_ids_end, :] - memory_map[:, sub_seq_ids_start, :] #difference of hidden states between different subsequences defined by start and end coordinates, shape: batch_size, num_sub_seq, hidden_size
    diff =torch.linalg.norm(differences, dim=2) # take norm of defference vector , diff = batch_size * (seq_len-1)
    start_hidden_state_norm = torch.linalg.norm(memory_map[:, sub_seq_ids_start, :], dim=2) # ||H_(t)||
    relative_diff = diff/start_hidden_state_norm # take the change between start and end sub_seq hidden states relative to the start hidden state of each pair {H_(start_i) - H_(end_i) } / ||H_(start_i)|| :,  relative hidden state change for each su_seq
    
    return relative_diff
    
def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)# a^2 + b^2 - 2a.b element wise

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def recall(embeddings, labels, K=[]):
    D = pdist(embeddings, squared=True)
    knn_inds = D.topk(1 + max(K), dim=1, largest=False, sorted=True)[1][:, 1:]

    """
    Check if, knn_inds contain index of query image.
    """
    assert ((knn_inds == torch.arange(0, len(labels), device=knn_inds.device).unsqueeze(1)).sum().item() == 0)

    selected_labels = labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)
    correct_labels = labels.unsqueeze(1) == selected_labels

    recall_k = []

    for k in K:
        correct_k = (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)
    return recall_k

def generate_random_pairs(num_pairs, min_value=1, max_value=100):
    # Generate two random tensors of shape (num_pairs,)
    start_ids = torch.randint(min_value, max_value, (num_pairs,))
    end_ids = torch.randint(min_value, max_value, (num_pairs,))
    
    # Stack start_ids and end_ids together and take the min and max along the 0th dimension(to ensure start<end)
    start_end = torch.stack([start_ids, end_ids], dim=0)
    start_ids, end_ids = torch.min(start_end, dim=0)[0], torch.max(start_end, dim=0)[0]

    return start_ids, end_ids

def generate_start_end_pairs(num_pairs, min_value=1, max_value=100):
    #generate random pairs
    rand_start, rand_end = generate_random_pairs(num_pairs, min_value, max_value)
    # Generate sub_seq start and end coordinates by setting z=1,2,5
    # random_permutation_5 = torch.randperm((max_value-5) - min_value + 1) + min_value
    # random_permutation_1 = torch.randperm((max_value-1) - min_value + 1) + min_value
    # random_permutation_2 = torch.randperm((max_value-2) - min_value + 1) + min_value
    # random_permutation_5=random_permutation_5[:num_pairs]
    # random_permutation_1=random_permutation_1[:num_pairs]
    # random_permutation_2=random_permutation_2[:num_pairs]

    # # Select the first num_numbers elements to get unique random numbers
    # start_ids = torch.cat((rand_start,random_permutation_2,random_permutation_1), dim=0)
    # end_ids = torch.cat(( rand_end, random_permutation_2+2,random_permutation_1+1), dim=0)

    # return start_ids, end_ids
    return rand_start, rand_end

def generate_start_end_pairs_dynamic_stepwise(ep, num_pairs, min_value=1, max_value=100, ep_break=50):
    if ep < 100:
        random_permutation_1 = torch.randperm((max_value-1) - min_value + 1) + min_value
        random_permutation_1=random_permutation_1[:num_pairs]
        return  random_permutation_1, random_permutation_1+1
        # #generate random pairs
        # rand_start, rand_end = generate_random_pairs(num_pairs, min_value, max_value)
        # return rand_start, rand_end

    elif ep < 150:
        random_permutation_2 = torch.randperm((max_value-2) - min_value + 1) + min_value
        random_permutation_2=random_permutation_2[:num_pairs]
        return  random_permutation_2, random_permutation_2+2

    else: 
        #generate random pairs
        rand_start, rand_end = generate_random_pairs(num_pairs, min_value, max_value)
        return rand_start, rand_end
        # random_permutation_1 = torch.randperm((max_value-1) - min_value + 1) + min_value
        # random_permutation_1=random_permutation_1[:num_pairs]
        # return  random_permutation_1, random_permutation_1+1
        
        # Generate sub_seq start and end coordinates by setting z=1,2,5
        # random_permutation_5 = torch.randperm((max_value-5) - min_value + 1) + min_value
        # random_permutation_1 = torch.randperm((max_value-1) - min_value + 1) + min_value
        # random_permutation_2 = torch.randperm((max_value-2) - min_value + 1) + min_value
        # random_permutation_5=random_permutation_5[:num_pairs]
        # random_permutation_1=random_permutation_1[:num_pairs]
        # random_permutation_2=random_permutation_2[:num_pairs]
    
        # # Select the first num_numbers elements to get unique random numbers
        # start_ids = torch.cat((rand_start,random_permutation_2,random_permutation_1), dim=0)
        # end_ids = torch.cat(( rand_end, random_permutation_2+2,random_permutation_1+1), dim=0)
    
        # return start_ids, end_ids
    
def generate_start_end_pairs_dynamic(z, num_pairs, min_value=1, max_value=100):
    if z==1 :
        random_permutation_1 = torch.randperm((max_value-1) - min_value + 1) + min_value
        random_permutation_1=random_permutation_1[:num_pairs]
        return  random_permutation_1, random_permutation_1+1


    elif z == 2:
        random_permutation_2 = torch.randperm((max_value-2) - min_value + 1) + min_value
        random_permutation_2=random_permutation_2[:num_pairs]
        return  random_permutation_2, random_permutation_2+2

    else: 
        #generate random pairs
        rand_start, rand_end = generate_random_pairs(num_pairs, min_value, max_value)
        return rand_start, rand_end