import os
import sys
import copy
import math
import numpy as np

import torch 
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch_geometric.nn import GCNConv,NNConv
from torch_geometric.data.batch import Batch
from torch_geometric.data.batch import Data as GraphData

from dataset import VisionDataset,MergeDataset



###################################For tactile#######################################
class TactileGCN(nn.Module):
    def __init__(self,Flag_Merge=False):
        super().__init__()
        self.Flag_Merge=Flag_Merge

        edge_sequentail=torch.nn.Sequential(
            nn.Linear(19,128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,7*128),
            # nn.BatchNorm1d(7*128),
            nn.ReLU(),
        )

        self.nnConv=NNConv(in_channels=7,out_channels=128,nn=edge_sequentail)#change node_feature to n*128; leverage edge_feature
        self.conv1 = GCNConv(128,256)
        self.linear=nn.Linear(1536,128)#Final MLP to process
        self.bn1=nn.BatchNorm1d(128)

        self.feature_linear=nn.Sequential(
            nn.Linear(1536,512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
        )
        

        #For pose estimate
        self.pose_linear=nn.Linear(128,7)

        self.edge_index=torch.tensor([[0,1,2,3,4],[5,5,5,5,5]],dtype=torch.long).cuda()#the last pose is BaseTHand
        
    def forward(self,data):
        

        ###If single data input###
        # data=GraphData(x=data[0],edge_index=self.edge_index,edge_attr=data[1])
        # x=x.view(1,-1)

        ###If batch data input###
        node_feature=self.nnConv(x=data.x,edge_index=data.edge_index,edge_attr=data.edge_attr)#384,128
        x=F.relu(self.conv1(node_feature,data.edge_index))
        x=x.view(data.num_graphs,-1)
        

        #For pose estimate#
        tactile_feature=self.feature_linear(x)#output x shape is 128
        x=self.pose_linear(tactile_feature)#output x shape is 7

        #Normalize the ori(maybe process later)
        # x=torch.unsqueeze(x,dim=0)#change to 1,7
        # xyz,ori=x[:,:3],x[:,3:]
        # ori = ori / torch.unsqueeze(torch.sqrt(torch.sum(torch.square(ori), dim=1)), dim=0).T#normalize ori
        # x = torch.cat([xyz, ori], dim=1)

        if self.Flag_Merge:
            return x,tactile_feature
        return x

def test_tactile_output():
    tactileGCN = TactileGCN().cuda().eval()

    node_data=torch.randn(6,7).cuda()
    edge_data=torch.randn(5,19).cuda()
    edge_index=torch.tensor([[0,1,2,3,4],[5,5,5,5,5]],dtype=torch.long).cuda()#the last pose is BaseTHand

    a=GraphData(x=node_data,edge_index=edge_index,edge_attr=edge_data)
    b=GraphData(x=node_data,edge_index=edge_index,edge_attr=edge_data*5)
    input_data=Batch.from_data_list([a,a,b])

    # print("input_data is:")
    # print(input_data)
    # print(input_data.x)
    # print(input_data.edge_index)
    # print(input_data.edge_attr)
    # print(input_data.batch)
    # print(input_data.ptr)
    
    output_data=tactileGCN(input_data)

    print(output_data.shape)
    print("Final output data is:")
    print(output_data)


###################################For Vision#######################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
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
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks,input_channel=3,num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        
        self.linear1 = nn.Linear(8192, 1024)#For specify 128*128 input
        # self.linear1 = nn.Linear(32768, 1024)#For specify 256,256 input
        self.bn_final=nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, num_classes)
        self.linear = nn.Linear(8192, num_classes)#For specify 128*128 input


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print("out shape is:",out.shape)
        out =  F.relu(self.bn_final(self.linear1(out)))#add batch normal
        out = self.linear2(out)
        # out = self.linear(out)
        return out

def ResNet18(output_feature):
    return ResNet(BasicBlock, [2, 2, 2, 2],input_channel=4,num_classes=output_feature)

class ImageCNN(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """
    def __init__(self,Flag_Merge=False):
        super().__init__()
        self.Flag_Merge=Flag_Merge

        self.res18=ResNet18(128)
        self.pose_linear=nn.Linear(128,7)

    def forward(self,inputs):
        image_feature=self.res18(inputs)
        output=self.pose_linear(image_feature)

        ###Use for norm output###
        xyz,ori=output[:,:3],output[:,3:]
        ori=ori/torch.unsqueeze(torch.sqrt(torch.sum(torch.square(ori),dim=1)),dim=0).T
        # return xyz,ori
        output=torch.cat([xyz,ori],dim=1)
        ###Use for norm output###

        if self.Flag_Merge:
            return output,image_feature

        return output
    
###################################For Merge#######################################
class MergeModel(nn.Module):
    def __init__(self,Flag_Merge=False):
        super().__init__()
        self.Flag_Merge=Flag_Merge
        self.tactileGCN=TactileGCN(Flag_Merge=True)
        self.imageCNN=ImageCNN(Flag_Merge=True)

        self.pose_linear=nn.Linear(256,7)

    def forward(self,rgbd_data,tactile_data):
        _,image_feature=self.imageCNN(rgbd_data)
        _,tactile_feature=self.tactileGCN(tactile_data)

        merge_feature=torch.concatenate([image_feature,tactile_feature],dim=1)
        predict_pose=self.pose_linear(merge_feature)

        if self.Flag_Merge:
            return predict_pose,merge_feature
        else:
            return predict_pose

###################################For SelectLSTM#######################################
class SelectLSTM(nn.Module):
    def __init__(self,embedding_dim=128,hidden_dim=256,target_dim=3):
        super(SelectLSTM,self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        
        self.feature_linear=nn.Linear(512,256)
        self.pose_linear=nn.Linear(21,256)
        self.merge_linear=nn.Linear(512,128)
        
        self.lstm=nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True
        )
        self.output_linear1=nn.Linear(self.hidden_dim,64)
        self.output_linear2=nn.Linear(64,target_dim)
        self.softmax=nn.LogSoftmax(dim=1)
        
    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),torch.zeros(1, 1, self.hidden_dim))
        
    def forward(self,inputs):
        merge_feature,input_pose=inputs
        merge_feature=self.feature_linear(merge_feature)
        pose_feature=self.pose_linear(input_pose)
        global_feature=torch.cat([merge_feature,pose_feature],dim=2)        
        global_feature=self.merge_linear(global_feature)
        
        out,hidden_data=self.lstm(global_feature)#Here just input a serial data, hidden data isn't needed
        
        x=self.output_linear1(out[:,-1,:])#just select final data feature
        x=self.output_linear2(x)
        result=self.softmax(x)
        
        return result

def test_selectLSTM():
    selectLSTM=SelectLSTM()
    inputs=[torch.randn(2,20,512),torch.randn(2,20,21)]
    out=selectLSTM(inputs)
    print("out shape is:")
    print(out.shape)
    


if __name__ == '__main__':
    test_tactile_output()
    # test_vision_output()
    # test_merge_output()
    # test_selectLSTM()
