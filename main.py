from cProfile import run
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

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
    k = 2
    ky = 1 # what is ky?? -> k of Transform
    
    # datasets and dataloaders
    dataroot = 'data'
    labeled_samples = 10000 # image per task of CIFAR dataset 
    unlabeled_task_samples = -1
    l_dist = 'super'
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

    tasks = train_dataset.tasks


if __name__ == '__main__':
    run(seed)