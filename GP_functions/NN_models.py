"""
File: NN_models.py
Author: Hongjin Ren
Description: Neural Network

"""

#############################################################################
## Package imports
#############################################################################

import torch


#############################################################################
## NN
#############################################################################
class NN_4(torch.nn.Module):
    def __init__(self, train_x, train_y):
        super(NN_4, self).__init__()

        self.fc1 = torch.nn.Linear(train_x.size(-1), 128) 
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(128)

        self.fc2 = torch.nn.Linear(128, 256) 
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(256)

        self.fc3 = torch.nn.Linear(256, 16)
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm1d(16)

        self.fc4 = torch.nn.Linear(16, train_y.size(-1))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.fc4(x)
        return x
    


class NN_5(torch.nn.Module):
    def __init__(self, train_x, train_y):
        super(NN_5, self).__init__()

        self.fc1 = torch.nn.Linear(train_x.size(-1), 128) 
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(128)

        self.fc2 = torch.nn.Linear(128, 256) 
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(256)

        self.fc3 = torch.nn.Linear(256, 64) 
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm1d(64)

        self.fc4 = torch.nn.Linear(64, 16)
        self.relu4 = torch.nn.ReLU()
        self.bn4 = torch.nn.BatchNorm1d(16)

        self.fc5 = torch.nn.Linear(16, train_y.size(-1))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.fc5(x)
        return x
    



class NN_5_1(torch.nn.Module):
    def __init__(self, train_x, train_y):
        super(NN_5_1, self).__init__()

        self.fc1 = torch.nn.Linear(train_x.size(-1), 128)
        self.relu1 = torch.nn.LeakyReLU()
        self.bn1 = torch.nn.LayerNorm(128)
        self.dropout1 = torch.nn.Dropout(p=0.2)

        self.fc2 = torch.nn.Linear(128, 256)
        self.relu2 = torch.nn.LeakyReLU()
        self.bn2 = torch.nn.LayerNorm(256)
        self.dropout2 = torch.nn.Dropout(p=0.2)

        self.fc3 = torch.nn.Linear(256, 256)
        self.relu3 = torch.nn.LeakyReLU()
        self.bn3 = torch.nn.LayerNorm(256)
        self.dropout3 = torch.nn.Dropout(p=0.2)

        self.fc4 = torch.nn.Linear(256, 64)
        self.relu4 = torch.nn.LeakyReLU()
        self.bn4 = torch.nn.LayerNorm(64)
        self.dropout4 = torch.nn.Dropout(p=0.2)

        self.fc_mu = torch.nn.Linear(64, train_y.size(-1))


    def forward(self, x):
        x = self.dropout1(self.bn1(self.relu1(self.fc1(x))))
        x = self.dropout2(self.bn2(self.relu2(self.fc2(x))))
        x = self.dropout3(self.bn3(self.relu3(self.fc3(x))))
        x = self.dropout4(self.bn4(self.relu4(self.fc4(x))))

        x = self.fc_mu(x)

        # x = torch.sigmoid(x)
        # x = x * (5 - 0.1) + 0.1

        return x
    

class NN_5_2(torch.nn.Module):
    def __init__(self, train_x, train_y):
        super(NN_5_2, self).__init__()

        self.fc1 = torch.nn.Linear(train_x.size(-1), 128)
        self.relu1 = torch.nn.LeakyReLU()
        self.bn1 = torch.nn.LayerNorm(128)
        self.dropout1 = torch.nn.Dropout(p=0.2)

        self.fc2 = torch.nn.Linear(128, 256)
        self.relu2 = torch.nn.LeakyReLU()
        self.bn2 = torch.nn.LayerNorm(256)
        self.dropout2 = torch.nn.Dropout(p=0.2)

        self.fc3 = torch.nn.Linear(256, 256)
        self.relu3 = torch.nn.LeakyReLU()
        self.bn3 = torch.nn.LayerNorm(256)
        self.dropout3 = torch.nn.Dropout(p=0.2)

        self.fc4 = torch.nn.Linear(256, 128)
        self.relu4 = torch.nn.LeakyReLU()
        self.bn4 = torch.nn.LayerNorm(128)
        self.dropout4 = torch.nn.Dropout(p=0.2)

        # The output layer is divided into two parts, one part predicts the mean, and the other part predicts the logarithm of the variance.
        self.fc_mu = torch.nn.Linear(128, train_y.size(-1))  # predicted mean
        self.fc_logvar = torch.nn.Linear(128, train_y.size(-1))  # Logarithm of prediction variance

    def forward(self, x):
        x = self.dropout1(self.bn1(self.relu1(self.fc1(x))))
        x = self.dropout2(self.bn2(self.relu2(self.fc2(x))))
        x = self.dropout3(self.bn3(self.relu3(self.fc3(x))))
        x = self.dropout4(self.bn4(self.relu4(self.fc4(x))))

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)  # Log variance
        return mu, logvar