from typing import Optional
import torch
import torch.nn as nn
from torch.autograd import Variable
import gc
import torch.nn.functional as F
import ipdb

def kl_div_with_logit(q_logit, p_logit):
    ### return a matrix without mean over samples.
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1)
    qlogp = ( q *logp).sum(dim=1)

    return qlogq - qlogp


def consistency_loss(Cri_CE_noreduce, cluster_labels, logits_w, logits_w_adv, logits_s, target_gt_for_visual, name='ce', T=1.0, p_cutoff=0.0,
                        use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()

    if name == 'L2':
        raise NotImplementedError
        # assert logits_w.size() == logits_s.size()
        # return F.mse_loss(logits_s, logits_w, reduction='mean')
    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask_binary =  max_idx.eq(cluster_labels) & max_probs.ge(p_cutoff)
        mask = mask_binary.float()
        if mask.mean().item() == 0:
            acc_selected = 0
        else:
            acc_selected = (target_gt_for_visual[mask_binary] == max_idx[mask_binary]).float().mean().item()
        if use_hard_labels:
            masked_loss = Cri_CE_noreduce(logits_s, max_idx) * mask
        else:
            raise NotImplementedError
            # pseudo_label = torch.softmax(logits_w / T, dim=-1)
            # masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), acc_selected

    else:
        assert Exception('Not Implemented consistency_loss')

def TarDisClusterLoss(args, epoch, output, target, softmax=True, em=False):
    if softmax:
        prob_p = F.softmax(output, dim=1)
    else:
        prob_p = output / output.sum(1, keepdim=True)
    if em:
        prob_q = prob_p
    else:
        prob_q1 = Variable(torch.cuda.FloatTensor(prob_p.size()).fill_(0))
        prob_q1.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).cuda()) # assigned pseudo labels
        if epoch == 0:
            prob_q = prob_q1
        else:
            prob_q2 = prob_p / prob_p.sum(0, keepdim=True).pow(0.5)
            prob_q2 /= prob_q2.sum(1, keepdim=True)
            prob_q = (1 - 1) * prob_q1 + 1 * prob_q2
    
    if softmax:
        loss = - (prob_q * F.log_softmax(output, dim=1)).sum(1).mean()
    else:
        loss = - (prob_q * prob_p.log()).sum(1).mean()
    
    return loss

def SrcClassifyLoss(output, target, index, src_cs):
    prob_p = F.softmax(output, dim=1)
    prob_q = Variable(torch.cuda.FloatTensor(prob_p.size()).fill_(0))
    prob_q.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).cuda())
    loss = - (src_cs[index] *(prob_q * F.log_softmax(output, dim=1)).sum(1)).mean()
    return loss
def accuracy_cluster(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        
    return res

def spherical_k_means(target_features, target_targets, epoch, c, num_classes):
    c_tar = c.data.clone()
    best_prec = 0
    for itr in range(5):
        torch.cuda.empty_cache()
        dist_xt_ct_temp = target_features.unsqueeze(1) * c_tar.unsqueeze(0)
        dist_xt_ct = 0.5 * (1 - dist_xt_ct_temp.sum(2) / (target_features.norm(2, dim=1, keepdim=True) * c_tar.norm(2, dim=1, keepdim=True).t() + 1e-6))
        _, idx_sim = (-1 * dist_xt_ct).data.topk(1, 1, True, True)
        prec1 = accuracy_cluster(-1 * dist_xt_ct.data, target_targets, topk=(1,))[0].item()
        is_best = prec1 > best_prec
        if is_best:
            best_prec = prec1
        print('Epoch %d, Spherical K-means clustering %d, Prec@1 %.3f' % (epoch, itr, prec1))

        c_tar_temp = torch.cuda.FloatTensor(num_classes, c_tar.size(1)).fill_(0)
        for k in range(num_classes):
            c_tar_temp[k] += (target_features[idx_sim.squeeze(1) == k] / (target_features[idx_sim.squeeze(1) == k].norm(2, dim=1, keepdim=True) + 1e-6)).sum(0)

        c_tar = c_tar_temp.clone()
        
        del dist_xt_ct_temp
        gc.collect()
        torch.cuda.empty_cache()
    
    del target_features
    del target_targets
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    return best_prec, c_tar, idx_sim.cpu()