import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import datasets
import models as models
import matplotlib.pyplot as plt
import torchvision.models as torch_models
from extra_setting import *
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import utils
import scipy.io as sio
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import cv2
import seaborn as sns
import operator
from scipy import misc

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch end2end ade Training')
parser.add_argument('-d', '--dataset', default='ade', help='dataset name')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--gpu', default='0', help='index of gpus to use')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('--resume', default='./ade/checkpoint_vgg16bn_hp.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


def main():
    global args, best_prec1
    args = parser.parse_args()

    # select gpus
    args.gpu = args.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    # data loader
    assert callable(datasets.__dict__[args.dataset])
    get_dataset = getattr(datasets, args.dataset)
    num_classes = datasets._NUM_CLASSES[args.dataset]
    train_loader, val_loader = get_dataset(
        batch_size=args.batch_size, num_workers=args.workers)

    # create model
    model_main = models.__dict__['vgg16_bn'](pretrained=True)
    model_main.classifier[-1] = nn.Linear(model_main.classifier[-1].in_features, num_classes)
    model_main = torch.nn.DataParallel(model_main, device_ids=range(len(args.gpu))).cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_main.module.load_state_dict(checkpoint['state_dict_m'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model_ahp_trunk = models.__dict__['vgg16_bn'](pretrained=True)
    model_ahp_trunk.classifier[-1] = nn.Linear(model_ahp_trunk.classifier[-1].in_features, 1000)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_ahp_trunk.load_state_dict(checkpoint['state_dict_ahp_trunk'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model_ahp_trunk = torch.nn.DataParallel(model_ahp_trunk, device_ids=range(len(args.gpu))).cuda()

    model_ahp_hp = models.__dict__['ahp_net_hp_res50_presigmoid']()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_ahp_hp.load_state_dict(checkpoint['state_dict_ahp_hp'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model_ahp_hp = torch.nn.DataParallel(model_ahp_hp, device_ids=range(len(args.gpu))).cuda()


    # generate predicted difficulty score
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_f = nn.CrossEntropyLoss(reduce=False).cuda()
    prec1, prec5, all_correct_te, all_predicted_te, all_class_dis_te = validate(val_loader, model_main, model_ahp_trunk,
                                                              model_ahp_hp, criterion, criterion_f)
    all_predicted_te = all_predicted_te.astype(int)
    np.save('./ade/all_correct_vgg16_te.npy', all_correct_te)
    np.save('./ade/all_predicted_vgg16_te.npy', all_predicted_te)
    np.save('./ade/all_class_dis_vgg16_te.npy', all_class_dis_te)

    all_correct_te = np.load('./ade/all_correct_vgg16_te.npy')
    all_predicted_te = np.load('./ade/all_predicted_vgg16_te.npy')
    all_class_dis_te = np.load('./ade/all_class_dis_vgg16_te.npy')

    difficulty_scores_te, difficulty_te_idx_each = save_predicted_difficulty(train_loader, val_loader, model_ahp_trunk, model_ahp_hp)
    np.save('./ade/difficulty_scores_te_vgg16.npy', difficulty_scores_te)
    np.save('./ade/difficulty_te_idx_each_vgg16.npy', difficulty_te_idx_each)

    difficulty_scores_te = np.load('./ade/difficulty_scores_te_vgg16.npy')
    difficulty_te_idx_each = np.load('./ade/difficulty_te_idx_each_vgg16.npy')


    # pickup K hardnest examples
    test_info = zip(all_correct_te, difficulty_scores_te, difficulty_te_idx_each)
    test_info = sorted(test_info, key=lambda test: test[1])  # from small to large
    all_correct_te, difficulty_scores_te, difficulty_te_idx_each = [list(l) for l in zip(*test_info)]
    all_correct_te = np.array(all_correct_te)
    difficulty_scores_te = np.array(difficulty_scores_te)
    difficulty_te_idx_each = np.array(difficulty_te_idx_each)

    K = 100
    K_idx_incor_classified = difficulty_te_idx_each[-K:]
    K_idx_incor_classified = K_idx_incor_classified.astype(int)

    imlist = []
    imclass = []

    with open('./ade/ADE_gt_val.txt', 'r') as rf:
        for line in rf.readlines():
            impath, imlabel, imindex = line.strip().split()
            imlist.append(impath)
            imclass.append(imlabel)

    picked_list = []
    picked_class_list = []
    for i in range(K):
        picked_list.append(imlist[K_idx_incor_classified[i]])
        picked_class_list.append(imclass[K_idx_incor_classified[i]])

    attr_map_hp = AttrMap_hp(model_ahp_trunk, model_ahp_hp, target_layer_names=["42"], use_cuda=True)
    attr_map_cls = AttrMap_cls(model_main, target_layer_names=["42"], use_cuda=True)

    com_extracted_attributes = np.load('./ade/com_extracted_attributes_001.npy')

    picked_seg_list = []
    for i in range(K):
        img_name = picked_list[i]
        img_name = img_name[:27] + "annotations" + img_name[33:-3] + "png"
        picked_seg_list.append(img_name)

    topK_prob_predicted_classes, _ = largest_indices_each_example(all_class_dis_te, 5)
    picked_topK_prob_predicted_classes = topK_prob_predicted_classes[K_idx_incor_classified, :]

    # save ade hard info
    adehard = './ade/ADEhard_gt_val.txt'
    fl = open(adehard, 'w')
    for ii in range(K):
        example_info = picked_list[ii] + " " + picked_class_list[ii] + " " + str(K_idx_incor_classified[ii])
        fl.write(example_info)
        fl.write("\n")
    fl.close()

    # data loader
    assert callable(datasets.__dict__['adehard'])
    get_dataset = getattr(datasets, 'adehard')
    num_classes = datasets._NUM_CLASSES['adehard']
    _, val_hard_loader = get_dataset(
        batch_size=1, num_workers=args.workers)


    remaining_mask_size_pool = np.arange(0.01, 1.0, 0.01)
    IOU = insecurity_extraction(val_hard_loader, attr_map_hp, attr_map_cls,
                                                                     picked_list, 3, com_extracted_attributes,
                                                                     picked_seg_list,
                                                                     picked_topK_prob_predicted_classes,
                                                                     remaining_mask_size_pool)


    print(IOU)

    np.save('./ade/hardness_predictor_vgg16_layer42_IG_IOU.npy', IOU)



def validate(val_loader, model_main, model_ahp_trunk, model_ahp_hp, criterion, criterion_f):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_main.eval()
    model_ahp_trunk.eval()
    model_ahp_hp.eval()
    end = time.time()

    all_correct_te = []
    all_predicted_te = []
    all_class_dis = np.zeros((1, 1040))
    for i, (input, target, index) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        output = model_main(input)
        class_dis = F.softmax(output, dim=1)
        class_dis = class_dis.data.cpu().numpy()
        all_class_dis = np.concatenate((all_class_dis, class_dis), axis=0)

        p_i_m = torch.max(output, dim=1)[1]
        all_predicted_te = np.concatenate((all_predicted_te, p_i_m), axis=0)
        p_i_m = p_i_m.long()
        p_i_m[p_i_m - target == 0] = -1
        p_i_m[p_i_m > -1] = 0
        p_i_m[p_i_m == -1] = 1
        correct = p_i_m.float()
        all_correct_te = np.concatenate((all_correct_te, correct), axis=0)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time,
                top1=top1, top5=top5))


    all_class_dis = all_class_dis[1:, :]
    return top1.avg, top5.avg, all_correct_te, all_predicted_te, all_class_dis


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def largest_indices_each_example(all_response, topK):
    topK_maxIndex = np.zeros((np.size(all_response, 0), topK), dtype=np.int16)
    topK_maxValue = np.zeros((np.size(all_response, 0), topK))
    for i in range(np.size(topK_maxIndex, 0)):
        arr = all_response[i, :]
        topK_maxIndex[i, :] = np.argsort(arr)[-topK:][::-1]
        topK_maxValue[i, :] = np.sort(arr)[-topK:][::-1]
    return topK_maxIndex, topK_maxValue


def save_predicted_difficulty(train_loader, val_loader, model_ahp_trunk, model_ahp_hp):
    model_ahp_trunk.eval()
    model_ahp_hp.eval()

    hardness_scores_val = []
    hardness_scores_idx_val = []
    for i, (input, target, index) in enumerate(val_loader):
        input = input.cuda()
        trunk_output = model_ahp_trunk(input)
        predicted_hardness_scores, _ = model_ahp_hp(trunk_output)
        scores = predicted_hardness_scores.data.cpu().numpy().squeeze()
        hardness_scores_val = np.concatenate((hardness_scores_val, scores), axis=0)
        index = index.numpy()
        hardness_scores_idx_val = np.concatenate((hardness_scores_idx_val, index), axis=0)

    return hardness_scores_val, hardness_scores_idx_val


def save_checkpoint(state, filename='checkpoint_res.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
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





class FeatureExtractor_hp():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules['module']._modules['features']._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]

        x = x.view(x.size(0), -1)
        x = self.model._modules['module'].classifier(x)
        return outputs, x


class FeatureExtractor_cls():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules['module']._modules['features']._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs_hp():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model_hp_trunk, model_hp_head, target_layers):
        self.model_hp_trunk = model_hp_trunk
        self.model_hp_head = model_hp_head
        self.feature_extractor = FeatureExtractor_hp(self.model_hp_trunk, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        _, output = self.model_hp_head(output)
        return target_activations, output


class ModelOutputs_cls():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor_cls(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model._modules['module'].classifier(output)  # travel many fc layers
        return target_activations, output


def preprocess_image(img):

    means = [0.4706145, 0.46000465, 0.45479808]
    stds = [0.26668432, 0.26578658, 0.2706199]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam

def show_segment_on_image(img, mask, mark_locs=None, is_cls=True):
    img = np.float32(img)

    mask = np.concatenate((mask[:, :, np.newaxis], mask[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
    img = np.uint8(255 * mask * img)
    if is_cls == False:
        if np.sum(mark_locs) > 0:
            x, y = np.where(mark_locs == 1)
            for i in range(np.size(x)):
                cv2.circle(img, (y[i], x[i]), 2, (0,0,255))
    return img



class AttrMap_hp:
    def __init__(self, model_hp_trunk, model_hp_head, target_layer_names, use_cuda):
        self.model_hp_trunk = model_hp_trunk
        self.model_hp_head = model_hp_head
        self.model_hp_trunk.eval()
        self.model_hp_head.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model_hp_trunk = model_hp_trunk.cuda()
            self.model_hp_head = model_hp_head.cuda()

        self.extractor = ModelOutputs_hp(self.model_hp_trunk, self.model_hp_head, target_layer_names)

    def forward(self, input):
        return self.model_hp_head(self.model_hp_trunk(input))

    def __call__(self, input, refer_input, step):

        gradients = np.zeros((step+1, 512, 14, 14))
        for i_step in range(step+1):
            cur_input = refer_input + (i_step / step) * (input - refer_input)
            if self.cuda:
                features, output = self.extractor(cur_input.cuda())
            else:
                features, output = self.extractor(cur_input)

            if i_step == 0:
                refer_features = features.copy()
            self.model_hp_trunk.zero_grad()
            self.model_hp_head.zero_grad()
            output.backward(retain_graph=True)
            grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
            gradients[i_step, :, :, :] = grads_val.squeeze()

        grads_val = np.mean(gradients, axis=0)

        gradients = np.copy(grads_val)
        gradients[gradients < 0.0] = 0.0
        gradients = gradients.squeeze()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        heatmaps = (target - refer_features[-1].cpu().data.numpy()[0, :]) * gradients
        heatmaps[heatmaps < 0.0] = 0.0
        heatmaps = np.sum(heatmaps, axis=0)

        return heatmaps


class AttrMap_cls:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs_cls(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, TopKclass = 5, topK_prob_predicted_classes=None, refer_input=None, step=50):

        gradients = np.zeros((step + 1, 512, 14, 14, np.size(topK_prob_predicted_classes)))
        for i_step in range(step + 1):
            cur_input = refer_input + (i_step / step) * (input - refer_input)
            if self.cuda:
                features, output = self.extractor(cur_input.cuda())
            else:
                features, output = self.extractor(cur_input)

            if i_step == 0:
                refer_features = features.copy()

            for i_cls in range(np.size(topK_prob_predicted_classes)):
                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0][topK_prob_predicted_classes[i_cls]] = 1
                one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
                if self.cuda:
                    one_hot = torch.sum(one_hot.cuda() * output)
                else:
                    one_hot = torch.sum(one_hot * output)
                self.model.zero_grad()
                one_hot.backward(retain_graph=True)
                grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
                grads_val = grads_val.squeeze()
                gradients[i_step, :, :, :, i_cls] = grads_val
        all_grads_val = np.mean(gradients, axis=0)

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        classifier_heatmaps = np.zeros((np.size(target,2), np.size(target,2), np.size(topK_prob_predicted_classes)))
        for i_cls in range(np.size(topK_prob_predicted_classes)):
            heatmaps = (target - refer_features[-1].cpu().data.numpy()[0, :]) * all_grads_val[:, :, :, i_cls]
            heatmaps = np.sum(heatmaps, axis=0)
            classifier_heatmaps[:, :, i_cls] = heatmaps

        return classifier_heatmaps



def insecurity_extraction(val_loader, attr_map_hp, attr_map_cls, imglist, topKcls, com_extracted_attributes, seg_list, topK_prob_predicted_classes, remaining_mask_size_pool):

    IOU = np.zeros((len(imglist), np.size(remaining_mask_size_pool)))

    for i, (input, target, index) in enumerate(val_loader):
        print('processing sample', i)

        img = cv2.imread(imglist[i])

        # make reference image tensor
        refer_img = np.float32(np.zeros((224, 224, 3)))
        refer_img = preprocess_image(refer_img)

        difficulty_heatmaps = attr_map_hp(input, refer_img, 50)

        classifier_heatmaps = attr_map_cls(input, 1040, topK_prob_predicted_classes[i, :], refer_img, 50)
        classifier_heatmaps[classifier_heatmaps < 0] = 1e-7

        seg_img = misc.imread(seg_list[i])
        seg_img = np.resize(seg_img, (224, 224))

        confusion_classes = np.argsort(classifier_heatmaps, axis=2)[:, :, -topKcls:]
        confusion_classes = np.sort(confusion_classes, axis=2)

        misclass_pairs = np.zeros((1, 2))
        for i_cls in range(topKcls):
            for j_cls in range(i_cls+1, topKcls):
                cur_misclass_pairs = np.concatenate((np.reshape(confusion_classes[:,:,i_cls].squeeze(), (-1, 1)), np.reshape(confusion_classes[:,:,j_cls].squeeze(), (-1, 1))), axis=1)
                misclass_pairs = np.concatenate((misclass_pairs, cur_misclass_pairs), axis=0)

        misclass_pairs = np.unique(misclass_pairs, axis=0)
        misclass_pairs = misclass_pairs[~np.all(misclass_pairs == 0, axis=1)]
        misclass_pairs = misclass_pairs.astype(int)

        atom_num = np.size(misclass_pairs, axis=0)

        for i_remain in range(np.size(remaining_mask_size_pool)):
            remaining_mask_size = remaining_mask_size_pool[i_remain]
            noeffect_atom = 0
            total_IOU_i = 0
            for i_atom in range(atom_num):

                insecurity = classifier_heatmaps[:, :, misclass_pairs[i_atom, 0]].squeeze() * classifier_heatmaps[:, :, misclass_pairs[i_atom, 1]].squeeze() * difficulty_heatmaps
                insecurity = cv2.resize(insecurity, (224, 224))
                insecurity_mask = np.copy(insecurity)

                threshold = np.sort(insecurity_mask.flatten())[int(-remaining_mask_size * 224 * 224)]
                insecurity_mask[insecurity_mask > threshold] = 1
                insecurity_mask[insecurity_mask < 1] = 0

                com_attributes = com_extracted_attributes[topK_prob_predicted_classes[i, misclass_pairs[i_atom, 0]], topK_prob_predicted_classes[i, misclass_pairs[i_atom, 1]]]


                if len(com_attributes) == 0:
                    continue
                com_attributes = np.array(com_attributes)

                commom_seg_img = np.zeros((224, 224))
                for i_com in range(np.size(com_attributes)):
                    commom_seg_img[seg_img == com_attributes[i_com]] = 1
                IOU_i = np.sum(insecurity_mask * commom_seg_img) / np.sum(insecurity_mask + commom_seg_img - insecurity_mask * commom_seg_img)
                total_IOU_i = total_IOU_i + IOU_i
            IOU[i, i_remain] = total_IOU_i / (atom_num - noeffect_atom)

    return np.nanmean(IOU, axis=0)



if __name__ == '__main__':
    main()



