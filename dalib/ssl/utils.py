import torch
from utils.utils import to_cuda
import numpy as np
import torch.nn as nn

def filter_samples(samples, threshold=0.05):
    batch_size_full = len(samples['data'])
    min_dist = torch.min(samples['dist2center'], dim=1)[0]
    mask = min_dist < threshold

    filtered_data = [samples['data'][m] 
		for m in range(mask.size(0)) if mask[m].item() == 1]
    filtered_label = torch.masked_select(samples['label'], mask)
    filtered_gt = torch.masked_select(samples['gt'], mask) \
                     if samples['gt'] is not None else None

    filtered_samples = {}
    filtered_samples['data'] = filtered_data
    filtered_samples['label'] = filtered_label
    filtered_samples['gt'] = filtered_gt

    assert len(filtered_samples['data']) == filtered_samples['label'].size(0)
    print('select %f' % (1.0 * len(filtered_data) / batch_size_full))

    return filtered_samples

def filter_class(labels, num_min, num_classes):
    filted_classes = []
    for c in range(num_classes):   
        mask = (labels == c)
        count = torch.sum(mask).item()
        if count >= num_min:
            filted_classes.append(c)

    return filted_classes

def split_samples_classwise(samples, num_classes):
    data = samples['data'] 
    label = samples['label']
    gt = samples['gt']
    samples_list = []
    for c in range(num_classes):
        mask = (label == c)
        data_c = [data[k] for k in range(mask.size(0)) if mask[k].item() == 1]
        label_c = torch.masked_select(label, mask)
        gt_c = torch.masked_select(gt, mask) if gt is not None else None
        samples_c = {}
        samples_c['data'] = data_c
        samples_c['label'] = label_c
        samples_c['gt'] = gt_c
        samples_list.append(samples_c)

    return samples_list

def adjust_learning_rate_exp(lr, optimizer, iters, decay_rate=0.1, decay_step=25):
    lr = lr * (decay_rate ** (iters // decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def adjust_learning_rate_RevGrad(lr, optimizer, max_iter, cur_iter, alpha=10, beta=0.75):
    p = 1.0 * cur_iter / (max_iter - 1)
    lr = lr / pow(1.0 + alpha * p, beta)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def adjust_learning_rate_inv(lr, optimizer, iters, alpha=0.001, beta=0.75):
    lr = lr / pow(1.0 + alpha * iters, beta)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def set_param_groups(net, lr_mult_dict):
    params = []
    modules = net.module._modules
    for name in modules:
        module = modules[name]
        if name in lr_mult_dict:
            params += [{'params': module.parameters(), 'lr_mult': lr_mult_dict[name]}]
        else:
            params += [{'params': module.parameters(), 'lr_mult': 1.0}]

    return params

def get_centers(model, dataloader, num_classes, key='feat'):        
    centers = 0 
    refs = to_cuda(torch.LongTensor(range(num_classes)).unsqueeze(1))
    source_path2label = {}
    # for i, (input, target, img_path, _) in enumerate(dataloader):
    #     print('soruce center calculation', i)
    #     with torch.no_grad():
    #         input = input.to(device)
    #         feature_source, _, _ = model(input)
    #     feature_source = feature_source.cpu()
    #     batchsize = feature_source.size(0)
    #     for j in range(batchsize):
    #         img_label = target[j]
    #         source_feature_list[img_label].append(feature_source[j].view(1, feature_source.size(1)))
    data_gt, data_paths = [], []
    for i, (input, target, img_path, _) in enumerate(dataloader):
        # print('soruce center calculation', i)
        with torch.no_grad():
            data = input.cuda()
            gt = target.cuda()
            data_paths += img_path
            data_gt += [target.cuda()]
            # batch_size = data.size(0)
            output, _, _ = model(data)
        feature = output.data 
        feat_len = feature.size(1)
        gt = gt.unsqueeze(0).expand(num_classes, -1)
        mask = (gt == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
        feature = feature.unsqueeze(0)
        # update centers
        centers += torch.sum(feature * mask, dim=1)

    data_gt = torch.cat(data_gt, dim=0) \
                if len(data_gt)>0 else None

    for i in range(len(dataloader.dataset)):
        source_path2label[data_paths[i]] = data_gt[i].item()
    return centers, source_path2label

