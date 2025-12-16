from torchvision import transforms
from utils import *

from datetime import datetime
import torch
# import torchio as tio
import torch.nn as nn

class setting_config:
    """
    the config of training setting.
    """

    network = 'segmamba'
    # model_config = {
    #     'num_classes': 1, 
    #     'input_channels': 3, 
    #     # ----- VM-UNet ----- #
    #     'depths': [2,2,2,2],
    #     'depths_decoder': [2,2,2,1],
    #     'drop_path_rate': 0.2,
    #     'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
    # }

    # datasets = 'isic18' 
    # if datasets == 'isic18':
    #     data_path = './data/isic2018/'
    # elif datasets == 'isic17':
    #     data_path = './data/isic2017/'
    # else:
    #     raise Exception('datasets in not right!')
    test_data_path = '/data/birth/lmx/work/Class_projects/bxg/CT_Report/valid_total_processed_data.h5'
    train_data_path='/data/birth/lmx/work/Class_projects/bxg/CT_Report/random_total_processed_data.h5'
    # data_path ='/data/birth/lmx/work/Class_projects/bxg/random_2000_processed_data.h5'

    # criterion = OrganSegmentationLoss(loss_type='soft', w_seg=1.0)
    # criterion = OrganSegmentationLoss(loss_type='dice', w_seg=1.0)


    pretrained_path = './pre_trained/'
    # resume_path='/data/birth/lmx/work/Class_projects/bxg/VM-UNet/results/vmunet__Friday_08_November_2024_12h_07m_49s/checkpoints/latest.pth'
    num_classes = 6
    num_abnormal_classes=16
    # input_size_h = 256
    # input_size_w = 256
    input_channels = 1
    n_base_filters=32
    # distributed = False
    # local_rank = -1
    model_depth=4
    num_workers = 8
    seed = 42
    world_size = None
    # rank = None
    # amp = False
    gpu_id = '4,5,6,7'
    batch_size =8#32q
    epochs = 150

    work_dir = 'results/' + network + '_'  + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 30
    val_interval = 30
    save_interval = 5
    threshold = 0.5

    block_size=64

    initial_segmentation_weight = 2
    abnormal_loss_weight=1
    weight_decay_rate=10
    segmentation_loss_weight=1

    
    segmentation_criterion = OrganSegmentationLoss(loss_type='dice', w_seg=1.0)
    # pos_weight=10.0
    abnormal_criterion = nn.BCEWithLogitsLoss()


    # # 修改后的 3D 数据增强流程，包含重采样
    # train_transformer = tio.Compose([
    #     # MyNormalize(),
    #     # MyResample(target_spacing=(1.0, 1.0, 1.0)),  # 对体素进行重采样，确保一致性
    #     # MyResize(target_shape=(128, 256, 256)),  # 调整图像大小，指定目标形状为 (D, H, W) 原来为(256, 512, 512)
    #     # MyRandomFlip(p=0.5, axes=(0, 1, 2)),
    #     # MyRandomRotation(p=0.5, degree=(-10, 10)),
    #     MyToTensor()  # 最后转换为 Tensor
    #     #有可能还需要直方图均衡化
    # ])

    # test_transformer = tio.Compose([
    #     # MyNormalize(),
    #     # MyResample(target_spacing=(1.0, 1.0, 1.0)),  # 确保测试数据也进行同样的重采样
    #     # MyResize(target_shape=(128, 256, 256)),  # 调整图像大小，指定目标形状为 (D, H, W)
    #     MyToTensor()  # 最后转换为 Tensor
    # ])


    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01 # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9 # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6 # default: 1e-6 – term added to the denominator to improve numerical stability 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'Adagrad':
        lr = 0.01 # default: 0.01 – learning rate
        lr_decay = 0 # default: 0 – learning rate decay
        eps = 1e-10 # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = 0.001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability 
        weight_decay = 0.0001 # default: 0 – weight decay (L2 penalty) 
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = 0.001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-2 # default: 1e-2 – weight decay coefficient
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond 
    elif opt == 'Adamax':
        lr = 2e-3 # default: 2e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'ASGD':
        lr = 0.01 # default: 1e-2 – learning rate 
        lambd = 1e-4 # default: 1e-4 – decay term
        alpha = 0.75 # default: 0.75 – power for eta update
        t0 = 1e6 # default: 1e6 – point at which to start averaging
        weight_decay = 0 # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        momentum = 0 # default: 0 – momentum factor
        alpha = 0.99 # default: 0.99 – smoothing constant
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        etas = (0.5, 1.2) # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50) # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes 
    elif opt == 'SGD':
        lr = 0.01 # – learning rate
        momentum = 0.9 # default: 0 – momentum factor 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
        dampening = 0 # default: 0 – dampening for momentum
        nesterov = False # default: False – enables Nesterov momentum 
    
    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5 # – Period of learning rate decay.
        gamma = 0.5 # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150] # – List of epoch indices. Must be increasing.
        gamma = 0.1 # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR':
        gamma = 0.99 #  – Multiplicative factor of learning rate decay.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'CosineAnnealingLR':
        T_max = 50 # – Maximum number of iterations. Cosine function period.
        eta_min = 0.00001 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1.  
    elif sch == 'ReduceLROnPlateau':
        mode = 'min' # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        factor = 0.1 # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10 # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        threshold = 0.0001 # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel' # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        cooldown = 0 # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0 # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08 # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50 # – Number of iterations for the first restart.
        T_mult = 2 # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 1e-6 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1. 
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20
