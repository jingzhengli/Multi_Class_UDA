import random
import time
import warnings
import sys
import argparse
import os.path as osp
import shutil
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import gc
import itertools
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

sys.path.append('../../..')
from dalib.adaptation.mdd_ssl import ClassificationMarginDisparityDiscrepancy\
    as MarginDisparityDiscrepancy, ImageClassifier
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
from data.prepare_data import generate_dataloader_sc as Dataloader
from data.prepare_data import UniformBatchSampler
from spherecluster import SphericalKMeans
from dalib.ssl.uda import consistency_loss, TarDisClusterLoss, SrcClassifyLoss
from config.config import cfg, cfg_from_file, cfg_from_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    cudnn.benchmark = True
    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True).to(device)
    num_classes = cfg.DATASET.NUM_CLASSES
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 width=args.bottleneck_dim).to(device)
    mdd = MarginDisparityDiscrepancy(args.margin).to(device)
    # define optimizer and lr_scheduler
    # The learning rate of the classiﬁers are set 10 times to that of the feature extractor by default.
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
    dataloaders = Dataloader()
    if args.phase == 'test':
        acc1 = validate(dataloaders['test'], classifier, args)
        print(acc1)
        return
    # def plot_confusion_matrix(net, dataloader):
    init_imgs = dataloaders['target_train_dataset'].imgs[:]
    if cfg.STRENGTHEN.DATALOAD == 'normal':
        dataloaders['target_train_dataset'].imgs = init_imgs[:]
        train_target_loader = torch.utils.data.DataLoader(
        dataloaders['target_train_dataset'], batch_size=cfg.TRAIN.SOURCE_BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, sampler=None)
        clustering_lables_with_path, source_centers, target_centers, acc_cluster_label = download_feature_and_clustering(dataloaders['source_cluster'],
                                        dataloaders['target_cluster'], classifier, num_classes)
    else:
        clustering_lables_with_path, source_centers, target_centers, acc_cluster_label = download_feature_and_clustering(dataloaders['source_cluster'],
                                        dataloaders['target_cluster'], classifier, num_classes)
        category_index_list, imgs = generate_category_index_list_imgs(clustering_lables_with_path, dataloaders['target_train_dataset'], num_classes)
        min_num_cate = cfg.STRENGTHEN.PERCATE  ## just a large number
        for i in range(len(category_index_list)):
            list_len = len(category_index_list[i])
            if min_num_cate > list_len:
                min_num_cate = list_len
        if min_num_cate < cfg.STRENGTHEN.PERCATE:  ### in case of some target category has few samples, we return to the normal dataloader
            dataloaders['target_train_dataset'].imgs = init_imgs[:]
            train_target_loader = torch.utils.data.DataLoader(
                dataloaders['target_train_dataset'], batch_size=cfg.TRAIN.SOURCE_BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, pin_memory=True, sampler=None)
        else:
            if cfg.STRENGTHEN.DATALOAD == 'hard':
                dataloaders['target_train_dataset'].imgs = init_imgs[:]
                uniformbatchsampler = UniformBatchSampler(cfg.STRENGTHEN.PERCATE, category_index_list, imgs)
                train_target_loader = torch.utils.data.DataLoader(dataloaders['target_train_dataset'],
                                                                    num_workers=cfg.NUM_WORKERS, pin_memory=True,
                                                                    batch_sampler=uniformbatchsampler)
            elif cfg.STRENGTHEN.DATALOAD == 'soft':
                dataloaders['target_train_dataset'].imgs = imgs  # udpate the image lists
                weights = make_weights_for_balanced_classes(dataloaders['target_train_dataset'].imgs, num_classes)
                weights = torch.DoubleTensor(weights)
                sampler_t = torch.utils.data.sampler.WeightedRandomSampler(weights, len(
                    weights))  #### sample instance uniformly for each category
                train_target_loader = torch.utils.data.DataLoader(
                    dataloaders['target_train_dataset'], batch_size=cfg.TRAIN.SOURCE_BATCH_SIZE, shuffle=False,
                    drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, sampler=sampler_t
                )
            else:
                raise NotImplementedError
    train_source_iter = ForeverDataIterator(dataloaders['source'])
    train_target_iter = ForeverDataIterator(train_target_loader)
    # start training
    best_acc1 = 0.
    acc_cluster_label_all = []
    acc_cluster_label_all.append(acc_cluster_label.item())
    acc_all = []
    for epoch in range(args.epochs):

        lamb = adaptation_factor(epoch * 1.0 / args.epochs)
        # train for one epoch
        train(train_source_iter, train_target_iter, dataloaders, classifier, mdd, optimizer, criterion_center_loss,
              lr_scheduler, epoch, args, clustering_lables_with_path,num_classes, lamb)

        # evaluate on validation set
        # acc1 = validate(val_loader, classifier, args)
        acc1 = validate(dataloaders['test'], classifier, args)
        acc_all.append(acc1)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
        print("lr = {:3.6f}".format(lr_scheduler._last_lr[1]))
        if epoch % cfg.STRENGTHEN.CLUSTER_FREQ == 0:    #   and not epoch == 0
            if not cfg.STRENGTHEN.DATALOAD == 'normal':
                clustering_lables_with_path, source_centers, target_centers, acc_cluster_label = download_feature_and_clustering(
                    dataloaders['source_cluster'],
                    dataloaders['target_cluster'], classifier, num_classes)
                category_index_list, imgs = generate_category_index_list_imgs(clustering_lables_with_path,
                                                                                    dataloaders['target_train_dataset'], num_classes)
                min_num_cate = cfg.STRENGTHEN.PERCATE  ## just a large number
                for i in range(len(category_index_list)):
                    list_len = len(category_index_list[i])
                    if min_num_cate > list_len:
                        min_num_cate = list_len
                if min_num_cate < cfg.STRENGTHEN.PERCATE:  ### in case of some target category has few samples, we return to the normal dataloader
                    dataloaders['target_train_dataset'].imgs = init_imgs[:]
                    train_target_loader = torch.utils.data.DataLoader(
                        dataloaders['target_train_dataset'], batch_size=cfg.TRAIN.SOURCE_BATCH_SIZE, shuffle=True,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True, sampler=None)
                else:
                    if cfg.STRENGTHEN.DATALOAD == 'hard':
                        dataloaders['target_train_dataset'].imgs = init_imgs[:]
                        uniformbatchsampler = UniformBatchSampler(cfg.STRENGTHEN.PERCATE, category_index_list, imgs)
                        train_target_loader = torch.utils.data.DataLoader(dataloaders['target_train_dataset'],
                                                                    num_workers=cfg.NUM_WORKERS, pin_memory=True,
                                                                    batch_sampler=uniformbatchsampler)
                        train_target_iter = ForeverDataIterator(train_target_loader)
                    elif cfg.STRENGTHEN.DATALOAD == 'soft':
                        dataloaders['target_train_dataset'].imgs = imgs  ################ udpate the image lists
                        weights = make_weights_for_balanced_classes(dataloaders['target_train_dataset'].imgs, num_classes)
                        weights = torch.DoubleTensor(weights)
                        sampler_t = torch.utils.data.sampler.WeightedRandomSampler(weights, len(
                            weights))  #### sample instance uniformly for each category
                        train_target_loader = torch.utils.data.DataLoader(
                            dataloaders['target_train_dataset'], batch_size=cfg.TRAIN.SOURCE_BATCH_SIZE, shuffle=False,
                            drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, sampler=sampler_t
                        )
                        train_target_iter = ForeverDataIterator(train_target_loader)
                    else:
                        raise NotImplementedError
            else:
                clustering_lables_with_path, source_centers, target_centers, acc_cluster_label = download_feature_and_clustering(
                    dataloaders['source_cluster'],
                   dataloaders['target_cluster'], classifier, num_classes)
                dataloaders['target_train_dataset'].imgs = init_imgs[:]
                train_target_loader = torch.utils.data.DataLoader(
                dataloaders['target_train_dataset'], batch_size=cfg.TRAIN.SOURCE_BATCH_SIZE, shuffle=True,
                drop_last=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, sampler=None)
                # classifier.moving_feature_centeriod_t = torch.from_numpy(target_centers)
                # classifier.moving_feature_centeriod_s = source_centers
                train_target_iter = ForeverDataIterator(train_target_loader)
                acc_cluster_label_all.append(acc_cluster_label.item())
    print("best_acc1 = {:3.1f}".format(best_acc1))
    print(acc_cluster_label_all)
    print(acc_all)

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(dataloaders['test'], classifier, args)
    print("test_acc1 = {:3.1f}".format(acc1))
    logger.close()

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, dataloaders,
          classifier: ImageClassifier, mdd: MarginDisparityDiscrepancy, optimizer: SGD, criterion_center_loss,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, clustering_lables_with_path,num_classes, lamb):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    acc_selecteds = AverageMeter('select Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, acc_selecteds, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    classifier.train()
    mdd.train()

    criterion = nn.CrossEntropyLoss().to(device)
    Cri_CE_noreduce = nn.CrossEntropyLoss(reduction='none').to(device)

    end = time.time()
    for iters in range(args.iters_per_epoch):
        optimizer.zero_grad()
        # optimizer_centloss.zero_grad()
        (x_s, x_s_aug), labels_s, index_s = next(train_source_iter)
        x_t, x_t_aug, labels_t, target_path, indexs= next(train_target_iter)
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        x_t_aug = x_t_aug.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)
        cluster_gt = torch.zeros(labels_t.size(), dtype=torch.int64)
        labels_t = labels_t.to(device)
        for i in range(len(target_path)):
            cluster_gt[i] = torch.as_tensor(clustering_lables_with_path[target_path[i]].astype(np.float64))
        cluster_gt = cluster_gt.to(device)
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        out_feature, outputs, outputs_adv = classifier(x)
        feature_source, feature_target = out_feature.chunk(2, dim=0)
        y_s, y_t = outputs.chunk(2, dim=0)
        y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

        # compute cross entropy loss on source domain
        cls_loss = criterion(y_s, labels_s)
        # compute margin disparity discrepancy between domains
        # for adversarial classifier, minimize negative mdd is equal to maximize mdd
        transfer_loss = -mdd(y_s, y_s_adv, y_t, y_t_adv)
        feature_aug, y_t_aug, _ = classifier(x_t_aug)
        # ssl_loss is the consistency regularity loss
        ssl_loss, _, acc_selected = consistency_loss(Cri_CE_noreduce,cluster_gt, y_t, y_t_adv, y_t_aug, labels_t, 'ce', T=1, p_cutoff=0.95, use_hard_labels=True)
        # 'semantic' indicates that our method takes not only intra-class centroid discrepancy into account but also the inter-class centroid discrepancy.
        if args.module == 'semantic' or args.module == 'global_semantic' or args.module == 'self_semantic':
            n, d = feature_target.shape
            # image number in each class
            ones = torch.ones_like(cluster_gt.cpu(), dtype=torch.float)
            zeros = torch.zeros(num_classes)
            s_n_classes = zeros.scatter_add(0, labels_s.cpu(), ones)
            t_n_classes = zeros.scatter_add(0, cluster_gt.cpu(), ones)
            # image number cannot be 0, when calculating centroids
            ones = torch.ones_like(s_n_classes)
            s_n_classes = torch.max(s_n_classes, ones).to(device)
            t_n_classes = torch.max(t_n_classes, ones).to(device)
            # calculating centroids, sum and divide
            zeros = torch.zeros(num_classes, d).to(device)
            s_sum_feature = zeros.scatter_add(0, torch.transpose(labels_s.repeat(d, 1), 1, 0), feature_source)
            t_sum_feature = zeros.scatter_add(0, torch.transpose(cluster_gt.repeat(d, 1), 1, 0), feature_target)
            current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(num_classes, 1))
            current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(num_classes, 1))
            decay = 0.001
            s_centroid = (1-decay) * classifier.moving_feature_centeriod_s + decay * current_s_centroid
            t_centroid = (1-decay) * classifier.moving_feature_centeriod_t + decay * current_t_centroid
            #############inter and intra#########################
            if args.module == 'semantic':
                s_expand = s_centroid.unsqueeze(1).expand(num_classes, num_classes, args.bottleneck_dim) #shape is (bs_A, bs_T, feat_len)
                t_expand = t_centroid.unsqueeze(0).expand(num_classes, num_classes, args.bottleneck_dim)
                dist = (((s_expand - t_expand))**2).sum(2)   #shape (class, class)
                I = torch.FloatTensor(np.eye(num_classes),).to(device)
                E = torch.FloatTensor(np.ones((num_classes, num_classes))).to(device)
                normalize_1 = num_classes
                normalize_2 = num_classes * num_classes - num_classes
                semantic_loss = torch.sum(dist*I)/normalize_1 - torch.sum(dist*(E-I))/normalize_2
                classifier.moving_feature_centeriod_s = s_centroid.detach()
                classifier.moving_feature_centeriod_t = t_centroid.detach()
            else:
                MSEloss = nn.MSELoss().to(device)
                semantic_loss = MSEloss(s_centroid, t_centroid)
                classifier.moving_feature_centeriod_s = s_centroid.detach()
                classifier.moving_feature_centeriod_t = t_centroid.detach()
        if args.module == 'ssl':
            loss = cls_loss + transfer_loss * args.trade_off + ssl_loss
        elif args.module == 'semantic' or args.module == 'global_semantic' or args.module == 'self_semantic':
            loss = cls_loss + transfer_loss * args.trade_off + lamb * semantic_loss + ssl_loss
        else:
            loss = cls_loss + transfer_loss * args.trade_off + ssl_loss + lamb * cluster_loss
        classifier.step()

        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_t.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        acc_selecteds.update(acc_selected, x_t.size(0))
        # center_losses.update(center_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iters % args.print_freq == 0:
            progress.display(iters)

def adaptation_factor(x):
	if x>= 1.0:
		return 1.0
	den = 1.0 + math.exp(-10 * x)
	lamb = 2.0 / den - 1.0
	return lamb

def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            # compute output
            _, output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if confmat:
            print(confmat.format(classes))
    return top1.avg

def download_feature_and_clustering(train_loader, val_loader, model, num_classes):
    model.eval()
    image_paths = []
    GT_labels = []
    source_feature_list = []
    for i in range(num_classes):
        source_feature_list.append([])  ######### each for one categoty
    for i, (input, target, img_path, _) in enumerate(train_loader):
        print('soruce center calculation', i)
        with torch.no_grad():
            input = input.to(device)
            feature_source, _, _ = model(input)
        feature_source = feature_source.cpu()
        batchsize = feature_source.size(0)
        for j in range(batchsize):
            img_label = target[j]
            source_feature_list[img_label].append(feature_source[j].view(1, feature_source.size(1)))

    target_feature_list = []
    for i, (input, target, img_path, indexs) in enumerate(val_loader):
        print('target feature calculation', i)
        with torch.no_grad():
            input = input.to(device)
            feature_target, p, _ = model(input)
            #p_distribution[indexs, :] = p
        batchsize = feature_target.size(0)
        feature_target = feature_target.cpu()
        for j in range(batchsize):
            GT_labels.append(target[j].item())
            image_paths.append(img_path[j])
            target_feature_list.append(feature_target[j].view(1, feature_target.size(1)))

    ########################################### calculte P
    # q = p_distribution**2 / torch.sum(p_distribution, dim=0)
    # q = q / torch.sum(q, dim=1, keepdim=True)
    feature_matrix = torch.cat(target_feature_list, dim=0)
    feature_matrix = F.normalize(feature_matrix, dim=1, p=2)
    feature_matrix = feature_matrix.numpy()
    ########################################### calculte source category center
    for i in range(num_classes):
        source_feature_list[i] = torch.cat(source_feature_list[i], dim=0)  ########## K * [num * dim]
        source_feature_list[i] = F.normalize(source_feature_list[i].mean(0), dim=0, p=2)
        source_feature_list[i] = source_feature_list[i].numpy()
    source_feature_array = np.array(source_feature_list)
    print('use the original cnn features to play cluster')

    kmeans = SphericalKMeans(n_clusters=num_classes, random_state=0, init=source_feature_array,
                                max_iter=500).fit(feature_matrix)
    Ind = kmeans.labels_
    target_centers = kmeans.cluster_centers_
    print(Ind)
    #print(GT_labels)
    gt_label_array = np.array(GT_labels)
    acc_count = torch.zeros(num_classes)
    all_count = torch.zeros(num_classes)
    for i in range(len(gt_label_array)):
        all_count[gt_label_array[i]] += 1
        if gt_label_array[i] == Ind[i]:
            acc_count[gt_label_array[i]] += 1

    acc_for_each_class1 = acc_count / all_count
    acc_cluster_label = sum(gt_label_array == Ind) / gt_label_array.shape[0]
    print(acc_cluster_label)
    corresponding_labels = []
    for i in range(len(Ind)):
        corresponding_labels.append(Ind[i])

    clustering_label_for_path = {image_paths[i]: corresponding_labels[i] for i in range(len(corresponding_labels))}
    return clustering_label_for_path, source_feature_array, target_centers, acc_cluster_label
def generate_category_index_list_imgs(clusering_labels_for_path, train_t_dataset, num_classes):
    images = []
    for i in range(len(train_t_dataset.imgs)):
        path = train_t_dataset.imgs[i][0]
        target = clusering_labels_for_path[path]
        item = (path, target)
        images.append(item)
    category_index_list = []
    for i in range(num_classes):
        list_temp = []
        for j in range(len(images)):
            if i == images[j][1]:
                list_temp.append(j)
        category_index_list.append(list_temp)
    return category_index_list, images
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    # weight_per_class[-1] = weight_per_class[-1]  ########### adjust the cate-weight for unknown category.
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    # dataset_names = sorted(
    #     name for name in datasets.__dict__
    #     if not name.startswith("__") and callable(datasets.__dict__[name])
    # )

    parser = argparse.ArgumentParser(description='Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='../../../experiments/configs/Office31/office31_train_amazon2webcam_cfg_SC.yaml', type=str)
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int)
    parser.add_argument('--margin', type=float, default=4., help="margin gamma")
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.004, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0002, type=float)
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='logs/our_method/Office31_A2W_semantic',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--module", type=str, default='semantic', choices=['ssl', 'semantic', 'cluster', 'global_semantic', 'self_semantic'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")

    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis', 'plot_confusion_matrix'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)

