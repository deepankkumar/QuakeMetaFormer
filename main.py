import os
import time
import argparse
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import cohen_kappa_score
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor,load_pretained
from torch.utils.tensorboard import SummaryWriter
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('MetaFG training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path',default='./imagenet', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    
    parser.add_argument('--num-workers', type=int, 
                        help="num of workers on dataloader ")
    
    parser.add_argument('--lr', type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float,
                        help='weight decay (default: 0.05 for adamw)')
    
    parser.add_argument('--min-lr', type=float,
                        help='learning rate')
    parser.add_argument('--warmup-lr', type=float,
                        help='warmup learning rate')
    parser.add_argument('--epochs', type=int,
                        help="epochs")
    parser.add_argument('--warmup-epochs', type=int,
                        help="epochs")
    
    parser.add_argument('--dataset', type=str,
                        help='dataset')
    parser.add_argument('--remove-attribute', type=str, help='Attribute to remove from the dataset', default=None)
    parser.add_argument('--lr-scheduler-name', type=str,
                        help='lr scheduler name,cosin linear,step')
    
    parser.add_argument('--pretrain', type=str,
                        help='pretrain')
    
    parser.add_argument('--tensorboard', action='store_true', help='using tensorboard')
    parser.add_argument('--perturbation_feature', type=int, help='Feature to perturbate', default=-1)
       
    # give classes weights for imbalanced dataset as argument
    parser.add_argument('--class-weights', type=float, nargs='+', help='class weights for imbalanced dataset')
    # sample argument: --class-weights 0.0064 0.0202 0.4884 0.0822 0.4028 
    
    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    # meta_encoding
    parser.add_argument("--meta_encoding", type=str, default='resnorm', help='meta encoder type')
    # fuse_location
    parser.add_argument("--fuse_location", type=str, default='4', help='location of fusion')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config
import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=4, device='cpu'):
        super(FocalLoss, self).__init__(weight)
        # focusing hyper-parameter gamma
        self.gamma = gamma

        # class weights will act as the alpha parameter
        self.weight = weight
        
        # using device (cpu or gpu)
        self.device = device
        
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, _input, _target):
        focal_loss = 0

        for i in range(len(_input)):
            # -log(pt)
            # print(_input[i].size())
            # print(_target[i].size())
            cur_ce_loss = self.ce_loss(_input[i].view(-1, _input[i].size()[-1]), _target[i].view(-1))
            # pt
            pt = torch.exp(-cur_ce_loss)

            if self.weight is not None:
                # alpha * (1-pt)^gamma * -log(pt)
                cur_focal_loss = self.weight[_target[i]] * ((1 - pt) ** self.gamma) * cur_ce_loss
            else:
                # (1-pt)^gamma * -log(pt)
                cur_focal_loss = ((1 - pt) ** self.gamma) * cur_ce_loss
                
            focal_loss = focal_loss + cur_focal_loss

        if self.weight is not None:
            focal_loss = focal_loss / self.weight.sum()
            return focal_loss.to(self.device)
        
        focal_loss = focal_loss / torch.tensor(len(_input))    
        return focal_loss.to(self.device)


def main(config):
    
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Use Focal loss for imbalanced dataset 
    # Calculate class weights
    # device = torch('cuda' if torch.cuda.is_available() else 'cpu')
    # Calculate class weights 
    # 1.36, 5.06, 39.03, 23.367 --- 1.423, 4.497, 108.27, 18.300, 89.08]
    # 0.0064, 0.0202, 0.4884, 0.0822, 0.4028]
    class_weights = torch.FloatTensor([0.34, 1.26, 9.8, 5.85]).cuda()
    # class_weights = torch.FloatTensor([0.0064, 0.0202, 0.4884, 0.0822, 0.4028]).cuda()
    print("***********************************")
    print("Class weights",config.CLASS_WEIGHTS)
    
    if config.CLASS_WEIGHTS is not None:
        class_weights = torch.FloatTensor(config.CLASS_WEIGHTS).cuda()
        class_weights *= 2  # Increase class weights by multiplying by 10

    elif config.DATA.DATASET == 'Turkey_smaller_EQ':   
        class_weights = torch.FloatTensor([0.0064, 0.0202, 0.4884, 0.0822, 0.4028]).cuda()
        class_weights *= 2  # Increase class weights by multiplying by 10

    else:
        class_weights = torch.FloatTensor([0.34, 1.26, 9.8, 5.85]).cuda()
        class_weights *= 2  # Increase class weights by multiplying by 10
        
    print("***********************************")
    print("Class weights",class_weights)
    
    # criterion = FocalLoss(weight=class_weights, gamma=4, device='cuda')
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    max_accuracy = 0.0
    max_f1 = 0.0
    max_auroc = 0.0
    if config.MODEL.PRETRAINED:
        load_pretained(config,model_without_ddp,logger)
        if config.EVAL_MODE:
            acc1, acc2, loss, _, _ = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            return

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            # config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        logger.info(f"**********normal test***********")
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc2, loss, _, _ = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.DATA.ADD_META:
            logger.info(f"**********mask meta test***********")
            acc1, acc2, loss, _,_ = validate(config, data_loader_val, model,mask_meta=True)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)      
        train_one_epoch_local_data(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
        
        logger.info(f"**********normal test***********")
        acc1, acc2, loss, f1, auc_roc = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        max_f1 = max(max_f1, f1)
        max_auroc = max(max_auroc, auc_roc)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        logger.info(f'Max f1: {max_f1:.2f}%')
        logger.info(f'Max auroc: {max_auroc:.2f}%')
        if dist.get_rank() == 0 and acc1 >= max_accuracy or (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)) or f1 >= max_f1 or auc_roc >= max_auroc:
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        if config.DATA.ADD_META:
            logger.info(f"**********mask meta test***********")
            acc1, acc2, loss, f1, auc_roc = validate(config, data_loader_val, model,mask_meta=True)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
#         data_loader_train.terminate()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
def train_one_epoch_local_data(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler,tb_logger=None):
    model.train()
    if hasattr(model.module,'cur_epoch'):
        model.module.cur_epoch = epoch
        model.module.total_epoch = config.TRAIN.EPOCHS
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, data in enumerate(data_loader):
        if config.DATA.ADD_META:
            samples, targets,meta = data
            meta = [m.float() for m in meta]
            meta = torch.stack(meta,dim=0)
            meta = meta.cuda(non_blocking=True)
        else:
            samples, targets= data
            meta = None

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)
        if config.DATA.ADD_META:
            outputs = model(samples,meta)
        else:
            outputs = model(samples)
        
        # print(outputs)
        # print(targets)
        
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
@torch.no_grad()


def validate(config, data_loader, model, mask_meta=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    merge_toggle = False
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc2_meter = AverageMeter()
    
    # Initialize lists to store all predictions and targets
    all_preds = []
    all_targets = []
    all_data = []
    # Initialize list to store all output probabilities
    all_outputs = []
    
    end = time.time()
    for idx, data in enumerate(data_loader):
        if config.DATA.ADD_META:
            images, target, meta = data
            meta = [m.float() for m in meta]
            meta = torch.stack(meta,dim=0)
            if mask_meta:
                meta = torch.zeros_like(meta)
            meta = meta.cuda(non_blocking=True)
        else:
            images, target = data
            meta = None
        
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # compute output
        if config.DATA.ADD_META:
            output = model(images, meta)
        else:
            output = model(images)

        if merge_toggle:
            # Convert output probabilities to softmax
            output = torch.nn.functional.softmax(output, dim=1)
            
            # Merge classes 2, 3, and 4 into one class (class 2)
            output[:, 2] = output[:, 2:5].sum(dim=1)
            
            # Keep only the first three columns (classes 0, 1, and the new merged class 2)
            output = output[:, :3]
            target = torch.where(torch.isin(target, torch.tensor([2, 3, 4], device=target.device)), torch.tensor(2, device=target.device), target)

        # Convert output probabilities to predicted class
        _, preds = torch.max(output, 1)

        # Append predictions and targets to lists
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        # Append output probabilities to list
        all_outputs.extend(torch.nn.functional.softmax(output, dim=1).cpu().numpy())
        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc2 = accuracy(output, target, topk=(1, 2))
        
        acc1 = reduce_tensor(acc1)
        acc2 = reduce_tensor(acc2)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc2_meter.update(acc2.item(), target.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@2 {acc2_meter.val:.3f} ({acc2_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    # Calculate precision, recall, and F1 score across all batches
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    
    # get the kappa score
    kappa = cohen_kappa_score(all_targets, all_preds)
    from sklearn.metrics import roc_curve, auc



    # Calculate AUC-ROC
    all_targets_bin = label_binarize(all_targets, classes=np.unique(all_targets))
    auc_roc = roc_auc_score(all_targets_bin, all_outputs, multi_class='ovr')
    auc_roc_total = auc_roc
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = all_targets_bin.shape[1]
    for i in range(n_classes):
        # Convert lists to numpy arrays
        all_targets_bin = np.array(all_targets_bin)
        all_outputs = np.array(all_outputs)

        # Now you can index them as 2D arrays
        fpr[i], tpr[i], _ = roc_curve(all_targets_bin[:, i], all_outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.rcParams['font.size'] = 11
    lw = 2
    lss = [(0,(1,1)), '--', ':', '-.',(0,(3,1,1,1)),(0,(5,1))]
    colors = cycle(['blue', 'darkorange', 'purple','red','green','pink'])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_targets_bin.ravel(), all_outputs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # if config.EVAL_MODE then save all_preds and all_targets to a csv file
    if config.EVAL_MODE:
        save_csv_path = os.path.join(config.OUTPUT, f'predictions_epoch_{config.TAG}.csv')
        # print all_data
        print(f"Saving predictions to {save_csv_path}")
        df = pd.DataFrame({'Predictions': all_preds, 'Targets': all_targets})
        # save to csv
        df.to_csv(save_csv_path, index=False)
        

    
    
    if config.EVAL_MODE:
        # Plot ROC curve for each class
        for i, color, ls in zip(range(n_classes), colors, lss):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, ls=ls,
                    label='Class {0} (AUC = {1:0.2f})'
                        ''.format(i, roc_auc[i]))

        # Plot micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"], color='black', lw=lw, alpha=0.0, 
                label='Average ROC curve (AUC = {0:0.2f})'
                    ''.format(auc_roc_total))

            

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        # save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
       
        file_name = f'roc_plot_epoch_{config.TAG}.png'
        save_roc_aoc_path = os.path.join(config.OUTPUT, f'roc_plot.png')
        print(f"Saving ROC plot to {save_roc_aoc_path}")
        plt.savefig(save_roc_aoc_path)

    logger.info(f' * AUC-ROC: {auc_roc:.3f}') 
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@2 {acc2_meter.avg:.3f}')
    logger.info(f' * Precision: {precision:.3f}  Recall: {recall:.3f}  F1 Score: {f1:.3f}')
    logger.info(f' * Kappa Score: {kappa:.3f}')

    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds) 
    # Normalize confusion matrix
    # Normalize confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cmap = sns.cm.rocket_r

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(cm * 100, annot=True, fmt=".2f", xticklabels=range(5), cmap=cmap, yticklabels=range(5), annot_kws={"size": 12})
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # # Manually set color bar ticks
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks(np.linspace(0, 100, 5))
    path_confusion_matrix = os.path.join(config.OUTPUT, f'confusion_matrix_plot.png')
    plt.savefig(path_confusion_matrix)


    
    return acc1_meter.avg, acc2_meter.avg, loss_meter.avg, f1, auc_roc
@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}",local_rank=config.LOCAL_RANK)

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
