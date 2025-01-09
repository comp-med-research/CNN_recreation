import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1:
    def __init__(self):
        super(CNN1,self).__init__()
        self.input = nn.Embed(224,224,3)
        self.cn1 = nn.Conv3d(3,3,16)
        self.cn2 = nn.Conv3d(3,3,64)
        self.cn3 = nn.Conv3d(3,3,256)
        self.relu = nn.ReLU()
        self.maxpool = F.max_pool3d(2,2)
        self.bn = nn.BatchNorm3d(128)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(160000,128)
        

    def forward(self, x):
        x = self.cn1(x)
        x = self.cn1(x)
        x = self.maxpool(x)
        x = self.cn2(x)
        x = self.cn2(x)
        x = self.maxpool(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = torch.flatten(x)
        x = self.fc(x)
        
        return x
    
class CNN2:
    def __init__(self):
        super(CNN2,self).__init__()
        self.embed = nn.Embedding()
        self.cn1 = nn.Conv3d(5,5,32)
        self.cn2 = nn.Conv3d(5,5,128)
        self.cn3 = nn.Conv3d(5,5,512)
        self.relu = nn.ReLU()
        self.maxpool = F.max_pool3d(2,2)
        self.bn = nn.BatchNorm3d(128)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(270848,128)
        
    
    def forward(self, x):
        x = self.cn1(x)
        x = self.cn1(x)
        x = self.maxpool(x)
        x = self.cn2(x)
        x = self.cn2(x)
        x = self.maxpool(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = torch.flatten(x)
        x = self.fc(x)
        return x


class TwoTowerCNN:
    def __init__(self, CNN1, CNN2):
        super(TwoTowerCNN,self).__init__()
        self.CNN1 = CNN1
        self.CNN2 = CNN2
        self.fc = nn.Linear(256,5)


    def forward(self, x):
        a = self.CNN1(x)
        b = self.CNN2(x)
        x = torch.cat((a, b), dim=0) 
        x = self.fc(x)
        # classification layer 
        x = F.softmax(x, dim=1)
        return x


