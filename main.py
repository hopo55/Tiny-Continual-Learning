from cProfile import run
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from collections import OrderedDict

import learners
import dataloaders
from dataloaders.utils import *


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
cudnn.deterministic = True

dataset = 'CIFAR100'

def run(seed):
    # prepare dataloader
    if dataset == 'CIFAR10':
        Dataset = dataloaders.iCIFAR10
        num_classes = 10
    elif dataset == 'CIFAR100':
        Dataset = dataloaders.iCIFAR100
        num_classes = 100
    elif dataset == 'TinyIMNET':
        Dataset = dataloaders.iTinyIMNET
        num_classes = 200
    else:
        Dataset = dataloaders.H5Dataset
        num_classes = 100

    # load tasks
    # Five shuffled classes are assigned to one task.
    rand_split = True
    class_order = np.arange(num_classes).tolist()
    class_order_logits = np.arange(num_classes).tolist()
    if seed > 0 and rand_split:
        random.shuffle(class_order)

    tasks = []
    tasks_logits = []
    p = 0
    first_split_size = 5
    other_split_size = 5
    
    while p < num_classes:
        inc = other_split_size if p > 0 else first_split_size
        tasks.append(class_order[p:p+inc])
        tasks_logits.append(class_order_logits[p:p+inc])
        p += inc
    num_tasks = len(tasks)
    task_names = [str(i+1) for i in range(num_tasks)]

    # number of transforms per image
    # Use fix-match loss with classifier
    k = 2 # Append transform image and buffer image
    ky = 1 # Not append transform for memory buffer
    
    # datasets and dataloaders
    dataroot = 'data'
    labeled_samples = 10000 # image per task of CIFAR dataset 
    unlabeled_task_samples = -1
    l_dist = 'super' # if l_dist is super, then resample task
    ul_dist = None
    validation = True
    repeat = 1
    
    train_aug = True
    train_transform = dataloaders.utils.get_transform(dataset=dataset, phase='train', aug=train_aug)
    train_transformb = dataloaders.utils.get_transform(dataset=dataset, phase='train', aug=train_aug, hard_aug=True)
    test_transform  = dataloaders.utils.get_transform(dataset=dataset, phase='test', aug=train_aug)

    train_dataset = Dataset(dataroot, dataset, labeled_samples, unlabeled_task_samples, train=True, lab = True,
                            download=True, transform=TransformK(train_transform, train_transform, ky), l_dist=l_dist, ul_dist=ul_dist,
                            tasks=tasks, seed=seed, rand_split=rand_split, validation=validation, kfolds=repeat)
    train_dataset_ul = Dataset(dataroot, dataset, labeled_samples, unlabeled_task_samples, train=True, lab = False,
                            download=True, transform=TransformK(train_transform, train_transformb, k), l_dist=l_dist, ul_dist=ul_dist,
                            tasks=tasks, seed=seed, rand_split=rand_split, validation=validation, kfolds=repeat)
    test_dataset  = Dataset(dataroot, dataset, train=False,
                            download=False, transform=test_transform, l_dist=l_dist, ul_dist=ul_dist,
                            tasks=tasks, seed=seed, rand_split=rand_split, validation=validation, kfolds=repeat)

    # in case tasks reset
    tasks = train_dataset.tasks

    # Prepare the Learner (model)
    learner_config = {'num_classes': num_classes,
                      'lr': 0.1,
                      'ul_batch_size': 128,
                      'tpr': 0.05,
                      'oodtpr': 0.005,
                      'momentum': 0.9,
                      'weight_decay': 5e-4,
                      'schedule': [120, 160, 180, 200],
                      'schedule_type': 'decay',
                      'model_type': "resnet",
                      'model_name': "WideResNet_28_2_cifar",
                      'ood_model_name': 'WideResNet_DC_28_2_cifar',
                      'out_dim': 100,
                      'optimizer': 'SGD',
                      'gpuid': [0],
                      'pl_flag': True,
                      'fm_loss': True,
                      'weight_aux': 1,
                      'memory': 400,
                      'distill_loss': 'C',
                      'co': 1.,
                      'FT': True,
                      'DW': True,
                      'num_labeled_samples': labeled_samples,
                      'num_unlabeled_samples': unlabeled_task_samples,
                      'super_flag': l_dist == "super",
                      'no_unlabeled_data': True
                      }
    learner = learners.distillmatch.DistillMatch(learner_config)
    print(learner_config['model_type'])

    oracle_flag = True
    acc_table = OrderedDict()
    acc_table_pt = OrderedDict()
    if len(task_names) > 1 and oracle_flag:
        run_ood = {}
    else:
        run_ood = None

    log_dir = "outputs/out"
    save_table = []
    save_table_pc = -1 * np.ones((num_tasks,num_tasks))
    pl_table = [[],[],[],[]]
    temp_dir = log_dir + '/temp'
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)

    # for oracle
    out_dim_add = 0

    # Training
    max_task = -1
    if max_task > 0:
        max_task = min(max_task, len(task_names))
    else:
        max_task = len(task_names)

    for i in range(max_task):
        train_name = task_names[i]


if __name__ == '__main__':
    run(seed)