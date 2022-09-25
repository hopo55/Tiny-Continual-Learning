"""
Code adapted from https://github.com/facebookresearch/GradientEpisodicMemory
                    &
                  https://github.com/kuangliu/pytorch-cifar
                    &
Supervised Contrastive Replay: Revisiting the Nearest Class Mean Classifier
in Online Class-Incremental Continual Learning
                  https://github.com/RaptorMai/online-continual-learning
"""
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from models.layers import CosineScaling

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.last = nn.Linear(nf * 8 * block.expansion, num_classes, bias=bias)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.last(x)
        return x

    def forward(self, x):
        out = self.features(x)
        logits = self.logits(out)
        return logits

class ResNet_DC(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet_DC, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.last_planes = nf * 8 * block.expansion
        self.bn_last = nn.BatchNorm2d(self.last_planes)
        self.last = CosineScaling(self.last_planes, num_classes)
        self.S_D = nn.Sequential(nn.Linear(self.last_planes, 1), nn.BatchNorm1d(1), nn.Sigmoid())


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        S_YD = self.last(x)
        S_D = self.S_D(x)
        return S_YD / S_D

    def forward(self, x):
        out = self.features(x)
        logits = self.logits(out)
        return logits

    def ood_forward(self,x):
        x = self.features(x)
        return self.last(x), self.S_D(x)

class NearestClassMean(nn.Module):
    """
    "Online Continual Learning for Embedded Devices"
    This is an implementation of the Nearest Class Mean algorithm for streaming learning.
    Code from https://github.com/tyler-hayes/Embedded-CL
    """

    def __init__(self, input_shape, num_classes, backbone=None, device='cuda'):
        """
        Init function for the NCM model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        """

        super(NearestClassMean, self).__init__()

        # NCM parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)

        # setup weights for NCM
        self.muK = torch.zeros((num_classes, input_shape)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.num_updates = 0

    @torch.no_grad()
    def fit(self, x, y, item_ix):
        """
        Fit the NCM model to a new sample (x,y).
        :param item_ix:
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        print('x shape', len(x.shape))
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        print('y shape', len(y.shape))
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        # update class means
        self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
        self.cK[y] += 1
        self.num_updates += 1

    @torch.no_grad()
    def find_dists(self, A, B):
        """
        Given a matrix of points A, return the indices of the closest points in A to B using L2 distance.
        :param A: N x d matrix of points
        :param B: M x d matrix of points for predictions
        :return: indices of closest points in A
        """
        M, d = B.shape
        with torch.no_grad():
            B = torch.reshape(B, (M, 1, d))  # reshaping for broadcasting
            square_sub = torch.mul(A - B, A - B)  # square all elements
            dist = torch.sum(square_sub, dim=2)
        return -dist

    @torch.no_grad()
    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)

        scores = self.find_dists(self.muK, X)

        # mask off predictions for unseen classes
        not_visited_ix = torch.where(self.cK == 0)[0]
        min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
        scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
            len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes

        # return predictions or probabilities
        if not return_probas:
            return scores.cpu()
        else:
            return torch.softmax(scores, dim=1).cpu()

    @torch.no_grad()
    def ood_predict(self, x):
        return self.predict(x, return_probas=True)

    @torch.no_grad()
    def evaluate_ood_(self, test_loader):
        print('\nTesting OOD on %d images.' % len(test_loader.dataset))

        num_samples = len(test_loader.dataset)
        scores = torch.empty((num_samples, self.num_classes))
        labels = torch.empty(num_samples).long()
        start = 0
        for test_x, test_y in test_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device)
            ood_scores = self.ood_predict(batch_x_feat)
            end = start + ood_scores.shape[0]
            scores[start:end] = ood_scores
            labels[start:end] = test_y.squeeze()
            start = end

        return scores, labels

    @torch.no_grad()
    def fit_batch(self, batch_x, batch_y, batch_ix):
        # fit NCM one example at a time
        for x, y in zip(batch_x, batch_y):
            self.fit(x.cpu(), y.view(1, ), None)

    @torch.no_grad()
    def train_(self, train_loader):
        # print('\nTraining on %d images.' % len(train_loader.dataset))

        for batch_x, batch_y, batch_ix in train_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(batch_x.to(self.device))
            else:
                batch_x_feat = batch_x.to(self.device)

            self.fit_batch(batch_x_feat, batch_y, batch_ix)

    @torch.no_grad()
    def evaluate_(self, test_loader):
        print('\nTesting on %d images.' % len(test_loader.dataset))

        num_samples = len(test_loader.dataset)
        probabilities = torch.empty((num_samples, self.num_classes))
        labels = torch.empty(num_samples).long()
        start = 0
        for test_x, test_y in test_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device)
            probas = self.predict(batch_x_feat, return_probas=True)
            end = start + probas.shape[0]
            probabilities[start:end] = probas
            labels[start:end] = test_y.squeeze()
            start = end
        return probabilities, labels

    def save_model(self, save_path, save_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """
        # grab parameters for saving
        d = dict()
        d['muK'] = self.muK.cpu()
        d['cK'] = self.cK.cpu()
        d['num_updates'] = self.num_updates

        # save model out
        torch.save(d, os.path.join(save_path, save_name + '.pth'))

    def load_model(self, save_file):
        """
        Load the model parameters into StreamingLDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        # load parameters
        print('\nloading ckpt from: %s' % save_file)
        d = torch.load(os.path.join(save_file))
        self.muK = d['muK'].to(self.device)
        self.cK = d['cK'].to(self.device)
        self.num_updates = d['num_updates']


def Reduced_ResNet18(out_dim=10, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], out_dim, nf, bias)

def Reduced_ResNet18_DC(out_dim=10, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet_DC(BasicBlock, [2, 2, 2, 2], out_dim, nf, bias)

def ResNet18(out_dim=10, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], out_dim, nf, bias)

'''
See https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

def ResNet34(out_dim=10, nf=64, bias=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], out_dim, nf, bias)

def ResNet50(out_dim=10, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], out_dim, nf, bias)


def ResNet101(out_dim=10, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], out_dim, nf, bias)


def ResNet152(out_dim=10, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], out_dim, nf, bias)