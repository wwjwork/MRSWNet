import datetime
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from functools import partial

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.net_parts import Resnet18
from nets.fpn_parts import Res_SA_UP_DOWN
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (get_anchors, get_classes,
                         seed_everything, show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    gpunum          = 1
    Cuda            = True
    seed            = 11 
    distributed     = False 
    sync_bn         = False 
    fp16            = False 
    num_workers     = 4 
    train_annotation_path   = 'MRTB-SWT_train.txt'         
    val_annotation_path     = 'MRTB-SWT_val.txt'          
    classes_path    = 'model_data/voc_classes.txt'      
    anchors_path    = 'model_data/yolo_anchors.txt'     
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] 
    input_shape     = [640, 640]    
    mosaic              = False      
    mosaic_prob         = 0      
    mixup               = False      
    mixup_prob          = 0      
    special_aug_ratio   = 0      
    label_smoothing     = 0      
    chanel      = 32                  
    depth       = [ 2,  2,4,4,4,4]    
    num_heads   = [ 4,  4,4,4,4,4]    
    mlp_ratio   = [ 4,  4,4,4,4,4]    
    sr_ratio    = [[4],[4],[4],[16,8,4,2],[8,4,2],[4,2]]    
    cr_ratio    = [[4],[4],[3],[ 0,1,2,3],[0,1,2],[0,1]]    
    model_path          = '' 
    pretrained          = False     
    Init_Epoch          = 0     
    Freeze_Train        = False  
    Freeze_Epoch        = 0      
    Freeze_batch_size   = 0      
    UnFreeze_Epoch      = 750    
    Unfreeze_batch_size = 16     
    optimizer_type      = "adam"                                               
    Init_lr             = 1e-3              
    Min_lr              = Init_lr * 0.01    
    momentum            = 0.937             
    weight_decay        = 0                 
    lr_decay_type       = "cos"            
    save_period         = 1                
    save_dir            = 'logs'            
    eval_flag           = True             
    eval_period         = 1               

    seed_everything(seed)
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank      = 0
        rank            = 0

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    backbone = Resnet18()
    model = Res_SA_UP_DOWN(backbone, anchors_mask, num_classes, chanel=chanel, depth=depth,
                    num_heads=num_heads, mlp_ratio=mlp_ratio, sr_ratio=sr_ratio, cr_ratio=cr_ratio)
    if not pretrained:
        weights_init(model)
    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, anchors_mask, label_smoothing)
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S') 
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    if fp16:
        from torch.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    ema = ModelEMA(model_train)

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path = classes_path, anchors_path = anchors_path, anchors_mask = anchors_mask, model_path = model_path, \
            input_shape = input_shape, Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, \
            Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4

    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' or optimizer_type == 'Adagrad' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' or optimizer_type == 'Adagrad' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam'    : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'     : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True),
            'Adagrad' : optim.Adagrad(pg0, Init_lr_fit)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        

        if ema:
            ema.updates     = epoch_step * Init_Epoch

        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length=UnFreeze_Epoch, \
                                        mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, 
                                        special_aug_ratio=special_aug_ratio)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length=UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)

        train_sampler   = None
        val_sampler     = None
        shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' or optimizer_type == 'Adagrad' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' or optimizer_type == 'Adagrad' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
       
                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size
                    
                if ema:
                    ema.updates     = epoch_step * epoch
                    
                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, \
                                             pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, \
                                             pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                UnFreeze_flag   = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, \
                          gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
