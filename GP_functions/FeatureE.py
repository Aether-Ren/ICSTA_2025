"""
File: FeatureE.py
Author: Hongjin Ren
Description: Feature extractor for deep kernel learning

"""

#############################################################################
## Package imports
#############################################################################

import torch


#############################################################################
## 
#############################################################################


## NN(FeatureExtractor)

class FeatureExtractor_1(torch.nn.Sequential):
    def __init__(self, train_x):
        super(FeatureExtractor_1, self).__init__()
        self.add_module('linear1', torch.nn.Linear(train_x.size(-1), 512))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(512, 256))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(256, 128))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(128, 16))


class FeatureExtractor_2(torch.nn.Sequential):
    def __init__(self, train_x):
        super(FeatureExtractor_2, self).__init__()
        self.add_module('linear1', torch.nn.Linear(train_x.size(-1), 256))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(256, 128))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(128, 32))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(32, 16))


class FeatureExtractor_3(torch.nn.Sequential):
    def __init__(self, train_x):
        super(FeatureExtractor_3, self).__init__()
        self.add_module('linear1', torch.nn.Linear(train_x.size(-1), 128))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(128, 256))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(256, 64))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(64, 16))


class FeatureExtractor_4(torch.nn.Sequential):
    def __init__(self, train_x):
        super(FeatureExtractor_4, self).__init__()
        self.add_module('linear1', torch.nn.Linear(train_x.size(-1), 256))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(256, 128))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(128, 64))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(64, 32))
        self.add_module('relu4', torch.nn.ReLU())
        self.add_module('linear5', torch.nn.Linear(32, 16))


class FeatureExtractor_5(torch.nn.Sequential):
    def __init__(self, train_x):
        super(FeatureExtractor_5, self).__init__()
        self.add_module('linear1', torch.nn.Linear(train_x.size(-1), 128))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(128, 256))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(256, 64))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('bn3', torch.nn.BatchNorm1d(64))
        self.add_module('linear4', torch.nn.Linear(64, 16))
        


class FeatureExtractor_6(torch.nn.Sequential):
    def __init__(self, train_x):
        super(FeatureExtractor_6, self).__init__()
        self.add_module('linear1', torch.nn.Linear(train_x.size(-1), 256))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(256, 128))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(128, 64))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('bn3', torch.nn.BatchNorm1d(64))
        self.add_module('linear4', torch.nn.Linear(64, 16))
        

class FeatureExtractor_7(torch.nn.Sequential):
    def __init__(self, train_x):
        super(FeatureExtractor_7, self).__init__()
        self.add_module('linear1', torch.nn.Linear(train_x.size(-1), 512))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(512, 256))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(256, 128))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('bn3', torch.nn.BatchNorm1d(128))
        self.add_module('linear4', torch.nn.Linear(128, 64))
        self.add_module('relu4', torch.nn.ReLU())
        self.add_module('bn4', torch.nn.BatchNorm1d(64))
        self.add_module('linear5', torch.nn.Linear(64, 16))
        
        

class FeatureExtractor_8(torch.nn.Sequential):
    def __init__(self, train_x):
        super(FeatureExtractor_8, self).__init__()
        self.add_module('linear1', torch.nn.Linear(train_x.size(-1), 256))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(256, 128))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('bn2', torch.nn.BatchNorm1d(128))
        self.add_module('linear3', torch.nn.Linear(128, 64))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('bn3', torch.nn.BatchNorm1d(64))
        self.add_module('linear4', torch.nn.Linear(64, 8))