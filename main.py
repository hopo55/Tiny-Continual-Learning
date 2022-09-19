import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from torch.utils.data import DataLoader

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
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

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
    validation = False
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
    workers = 8
    batch_size = 64
    ul_batch_size = 128
    learner_config = {'num_classes': num_classes,
                      'lr': 0.1,
                      'ul_batch_size': 128,
                      'tpr': 0.05, # tpr for ood calibration of class network
                      'oodtpr': 0.05, # tpr for ood calibration of ood network
                      'momentum': 0.9,
                      'weight_decay': 5e-4,
                    #   'schedule': [120, 160, 180, 200], # schedule and epoch(schedule[-1])
                      'schedule': [1, 2, 3, 4],
                      'schedule_type': 'decay',
                      'model_type': "resnet",
                      'model_name': "WideResNet_28_2_cifar",
                      'ood_model_name': 'WideResNet_DC_28_2_cifar',
                      'out_dim': 100,
                      'optimizer': 'SGD',
                      'gpuid': [0],
                      'pl_flag': True, # use pseudo-labeled ul data for DM
                      'fm_loss': True, # Use fix-match loss with classifier -> Consistency Regularization / eq.4 -> unsupervised loss
                      'weight_aux': 1.0,
                      'memory': 4000,
                      'distill_loss': 'C',
                      'co': 1., # out-of-distribution confidence loss ratio
                      'FT': True, # finetune distillation -> 이거 필요한가???
                      'DW': True, # dataset balancing
                      'num_labeled_samples': labeled_samples,
                      'num_unlabeled_samples': unlabeled_task_samples,
                      'super_flag': l_dist == "super",
                      'no_unlabeled_data': False
                      }

    learner = learners.distillmatch.DistillMatch(learner_config)

    acc_table = OrderedDict()
    acc_table_pt = OrderedDict()
    run_ood = {}

    log_dir = "outputs/CIFAR100-10k/realistic/dm"
    save_table = []
    save_table_pc = -1 * np.ones((num_tasks,num_tasks))
    pl_table = [[],[],[],[]]
    temp_dir = log_dir + '/temp'
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)

    # Training
    max_task = -1
    if max_task > 0:
        max_task = min(max_task, len(task_names))
    else:
        max_task = len(task_names)

    for i in range(max_task):
        # set seeds
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)

        train_name = task_names[i]
        print('======================', train_name, '=======================')

        # load dataset for task
        task = tasks_logits[i]
        prev = sorted(set([k for task in tasks_logits[:i] for k in task])) # prev where classes learned so far are stored

        # current class와 prev class 모두 load
        train_dataset.load_dataset(prev, i, train=True)
        train_dataset_ul.load_dataset(prev, i, train=True)
        out_dim_add = len(task)

        # load dataset with memory(coreset)
        train_dataset.append_coreset(only=False)

        # load dataloader
        train_loader_l = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=int(workers / 2))
        train_loader_ul = DataLoader(train_dataset_ul, batch_size=ul_batch_size, shuffle=True, drop_last=False, num_workers=int(workers / 2))
        train_loader_ul_task = DataLoader(train_dataset_ul, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=int(workers / 2))
        train_loader = dataloaders.SSLDataLoader(train_loader_l, train_loader_ul) # return labeled data, unlabeled data

        # add valid class to classifier
        learner.add_valid_output_dim(out_dim_add) # return number of classes learned to the current task

        # Learn
        # load test dataset dataloader
        test_dataset.load_dataset(prev, i, train=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers)

        model_save_dir = log_dir + '/models/repeat-'+str(seed+1)+'/task-'+task_names[i]+'/'
        if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)

        learner.learn_batch(train_loader, train_dataset, train_dataset_ul, model_save_dir, test_loader)
        
        # Evaluate
        acc_table[train_name] = OrderedDict()
        acc_table_pt[train_name] = OrderedDict()
        for j in range(i+1):
            val_name = task_names[j]
            print('validation split name:', val_name)
            test_dataset.load_dataset(prev, j, train=True)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers)

            # validation
            acc_table[val_name][train_name] = learner.validation(test_loader)
            save_table_pc[i,j] = acc_table[val_name][train_name]

            # past task validation
            acc_table_pt[val_name][train_name] = learner.validation(test_loader, task_in = tasks_logits[j])

        save_table.append(np.mean([acc_table[task_names[j]][train_name] for j in range(i+1)]))

        # Evaluate PL
        if i+1 < len(task_names):
            test_dataset.load_dataset(prev, len(task_names)-1, train=False)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers)
            stats = learner.validation_pl(test_loader)
            names = ['stats-fpr','stats-tpr','stats-de']
            for ii in range(3):
                pl_table[ii].append(stats[ii])
                save_file = temp_dir + '/'+names[ii]+'_table.csv'
                np.savetxt(save_file, np.asarray(pl_table[ii]), delimiter=",", fmt='%.2f')

            run_ood['tpr'] = pl_table[1]
            run_ood['fpr'] = pl_table[0]
            run_ood['de'] = pl_table[2]

        # save temporary results
        save_file = temp_dir + '/acc_table.csv'
        np.savetxt(save_file, np.asarray(save_table), delimiter=",", fmt='%.2f')
        save_file_pc = temp_dir + '/acc_table_pc.csv'
        np.savetxt(save_file_pc, np.asarray(save_table_pc), delimiter=",", fmt='%.2f')

    return acc_table, acc_table_pt, task_names, run_ood

if __name__ == '__main__':
    acc_table, acc_table_pt, task_names, run_ood = run(seed)
    print(acc_table)