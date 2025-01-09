import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1,self).__init__()
        self.cn1 = nn.Conv3d(3,16,kernel_size=(1, 3, 3))
        self.cn2 = nn.Conv3d(16,16,kernel_size=(1, 3, 3))
        self.cn3 = nn.Conv3d(16,64,kernel_size=(1, 3, 3))
        self.cn4 = nn.Conv3d(64,64,kernel_size=(1, 3, 3))
        self.cn5 = nn.Conv3d(64,256,kernel_size=(1, 3, 3))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(256)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(160000,128)
        

    def forward(self, x):
        x = x.float() 
        x = self.cn1(x)
        x = self.relu(x)
        x = self.cn2(x)
        x = self.relu(x)
        x = F.max_pool3d(x,kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.cn3(x)
        x = self.relu(x)
        x = self.cn4(x)
        x = self.relu(x)
        x = F.max_pool3d(x,kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.cn5(x)
        x = self.relu(x)
        x = F.max_pool3d(x,kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.bn(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        return x
    
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2,self).__init__()
        self.cn1 = nn.Conv3d(3,32,kernel_size=(1, 5, 5))
        self.cn2 = nn.Conv3d(32,32,kernel_size=(1,5, 5))
        self.cn3 = nn.Conv3d(32,128,kernel_size=(1,5, 5))
        self.cn4 = nn.Conv3d(128,128,kernel_size=(1, 5, 5))
        self.cn5 = nn.Conv3d(128,512,kernel_size=(1, 5, 5))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(512)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(270848,128)
        

    def forward(self, x):
        x = x.float() 
        x = self.cn1(x)
        x = self.relu(x)
        x = self.cn2(x)
        x = self.relu(x)
        x = F.max_pool3d(x,kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.cn3(x)
        x = self.relu(x)
        x = self.cn4(x)
        x = self.relu(x)
        x = F.max_pool3d(x,kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.cn5(x)
        x = self.relu(x)
        x = F.max_pool3d(x,kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.bn(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        return x


class TwoTowerCNN(nn.Module):
    def __init__(self, CNN1, CNN2):
        super(TwoTowerCNN,self).__init__()
        self.CNN1 = CNN1
        self.CNN2 = CNN2
        self.fc = nn.Linear(256,5)


    def forward(self, x):
        a = self.CNN1(x)
        b = self.CNN2(x)
        x = torch.cat((a, b), dim=1) 
        x = self.fc(x)
        # classification layer 
        x = F.softmax(x, dim=1)
        return x


