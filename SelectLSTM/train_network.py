import os
import shutil
import logging
import argparse
import datetime
import numpy as np
import open3d as o3d
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch_geometric.data.batch import Batch
from torch_geometric.data.batch import Data as GraphData


from model import MergeModel,TactileGCN,ImageCNN,SelectLSTM
from dataset import MergeDataset,TactileDataset,VisionDataset,SelectDataset
from pose_loss import Loss6D


object_name="sugar"


#Printoptions for pytorch
torch.set_printoptions(
    precision=4,    
    threshold=1000,
    edgeitems=3,
    linewidth=150,  
    profile=None,
    sci_mode=False  
)

#--------------------------------------------------------------------------------------------------#
################################# Generate log file for training ###################################
Record_Flag=True
Record_Info="Tactile"
Record_File="Baseline-experiments-{}".format(object_name)
#--------------------------------------------------------------------------------------------------#
#region

Abs_Path=os.path.dirname(os.path.abspath(__file__))

def all_logger_info(str):
    if Record_Flag:
        print(str)
        all_logger.info(str)
    else:
        print(str)

def train_logger_info(str):
    if Record_Flag:    
        print(str)
        train_logger.info(str)
    else:
        print(str)

if Record_Flag:
    #1: Generate log file and save_model path
    #1.1: generate save path
    timestr = Record_Info+str(datetime.datetime.now().strftime('--%m-%d_%H-%M'))

    #generate big savepath path
    save_path=Path(Abs_Path+"/log")
    save_path.mkdir(exist_ok=True)
    save_path=save_path.joinpath(Record_File)
    save_path.mkdir(exist_ok=True)
    save_path=save_path.joinpath(timestr)
    save_path.mkdir(exist_ok=True)
    record_txt=open(os.path.join(save_path,"record.txt"),'a')
    record_txt.write(Record_Info)

    save_models_path=save_path.joinpath("models")
    save_models_path.mkdir(exist_ok=True)
    log_path=save_path.joinpath("logs")
    log_path.mkdir(exist_ok=True)

    #1.2: generate log file to save data
    train_logger=logging.getLogger("train_logger")
    train_logger.setLevel(logging.INFO)
    file_handler=logging.FileHandler(os.path.join(log_path,"train_logger.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    train_logger.addHandler(file_handler)
    train_logger_info("train logger begin to record data!!!")

    all_logger=logging.getLogger("all_logger")
    all_logger.setLevel(logging.INFO)
    file_handler=logging.FileHandler(os.path.join(log_path,"all_logger.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    all_logger.addHandler(file_handler)
    all_logger_info("all logger info begin to record data!!!")

    # 1.3: copy trian python file and model file to log data
    shutil.copy(os.path.abspath(__file__), save_path)#copy train file
    shutil.copy(os.path.join(Abs_Path,"model.py"),save_path)
    train_logger_info("copy train file:{} to save_path {}".format(os.path.abspath(__file__),save_path))

    #--------------------------------------------------------------------------------------------------#
    #---------------------------------------Sepcial setting--------------------------------------------#
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--WholeDatasetPath', type=str, default="/home/media/WholeDataset",
                        help='Wholedataset path')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    WholeDatasetPath=args.WholeDatasetPath
#endregion
#--------------------------------------------------------------------------------------------------#1


def train_tactile_data(object_name):
    """
    To test different serial generalization
    """
    #1: Init all data
    dataset_path=os.path.join(WholeDatasetPath,object_name)
    object_name=dataset_path.split('/')[-1]

    NUM_batch_size=128
    NUM_workers=12
    NUM_train_epoch=100
    NUM_lowest_loss=1000
    Flag_shuffle=True

    train_dataset=TactileDataset(dataset_path,split_method='shuffle',split='train',split_rate=0.6) 
    test_dataset=TactileDataset(dataset_path,split_method='shuffle',split='test',split_rate=0.6) 
    train_loader=DataLoader(train_dataset,batch_size=NUM_batch_size,shuffle=Flag_shuffle,num_workers=NUM_workers)
    test_loader=DataLoader(test_dataset,batch_size=NUM_batch_size,shuffle=Flag_shuffle,num_workers=NUM_workers)
    edge_index=torch.tensor([[0,1,2,3,4],[5,5,5,5,5]],dtype=torch.long).cuda()#the last pose is BaseTHand
    
    network=TactileGCN().cuda()
    optimizer=torch.optim.Adam(network.parameters(),lr=0.001)

    object_point_cloud = o3d.io.read_triangle_mesh(os.path.join(Abs_Path,"ycb_models/{}/textured.obj".format(object_name)))
    object_point_cloud = object_point_cloud.sample_points_uniformly(3000)    
    loss6D=Loss6D(o3d_point_cloud=object_point_cloud,num_points=3000)
    

    #2: Begin to train the network
    for epoch in range(0,NUM_train_epoch):

        #2.1 In training process
        sum_train_loss=0
        sum_test_loss=0
        network.train()
        for batch_idx,data in enumerate(train_loader):
            tip_pose_array,tip_data_array,target_pose=data
            tip_pose_array,tip_data_array,target_pose=tip_pose_array.cuda(),tip_data_array.cuda(),target_pose.cuda()

            #Generate batch graph data
            batch_graphdata_list=[]
            for index,tip_pose in enumerate(tip_pose_array):
                tactile_data=tip_data_array[index]
                graphData=GraphData(x=tip_pose,edge_index=edge_index,edge_attr=tactile_data)
                batch_graphdata_list.append(graphData)
            input_tactile_data=Batch.from_data_list(batch_graphdata_list)

            #network infer
            predict_result=network(input_tactile_data)

            #optimize network
            loss=loss6D(predict_result,target_pose)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_train_loss=sum_train_loss+loss.item()

            if batch_idx%10==0:
                batch_size_fmt="{:15}\t".format("Train:"+str(batch_idx*NUM_batch_size))
                all_logger_info(batch_size_fmt+'| Loss: {:.6f}'.format(loss.item()/len(tip_pose_array)))                

        #2.2 In test process
        network.eval()
        for batch_idx,data in enumerate(test_loader):
            tip_pose_array,tip_data_array,target_pose=data
            tip_pose_array,tip_data_array,target_pose=tip_pose_array.cuda(),tip_data_array.cuda(),target_pose.cuda()
            
            #Generate batch graph data
            batch_graphdata_list=[]
            for index,tip_pose in enumerate(tip_pose_array):
                tactile_data=tip_data_array[index]
                graphData=GraphData(x=tip_pose,edge_index=edge_index,edge_attr=tactile_data)
                batch_graphdata_list.append(graphData)
            input_tactile_data=Batch.from_data_list(batch_graphdata_list)

            #network infer
            predict_result=network(input_tactile_data)
            
            #get loss
            # loss=F.mse_loss(predict_result,target_pose)
            loss=loss6D(predict_result,target_pose)
            sum_test_loss=sum_test_loss+loss.item()

            if batch_idx%10==0:
                batch_size_fmt="{:15}\t".format("Test:"+str(batch_idx*NUM_batch_size))
                all_logger_info(batch_size_fmt+'| Loss: {:.6f}'.format(loss.item()/len(tip_pose_array)))


                

        #3: Save result
        average_train_loss=sum_train_loss/(len(train_loader)*NUM_batch_size)
        average_valid_loss=sum_test_loss/(len(test_loader)*NUM_batch_size)

        epoch_fmt="{:10}\t".format("Epoch:"+str(epoch))
        train_logger_info(epoch_fmt+"|Train loss: {:.6f} Test loss: {:.6f}".format(average_train_loss,average_valid_loss))
        all_logger_info(epoch_fmt+"|Train loss: {:.6f} Test loss: {:.6f}".format(average_train_loss,average_valid_loss))

        if Record_Flag:
            if average_valid_loss<NUM_lowest_loss:
                NUM_lowest_loss=average_valid_loss
                all_logger_info("!!!!Find new lowest loss in epoch{},loss is:{}!!!!!".format(epoch,average_valid_loss))
                bestsave_path=os.path.join(save_models_path,'best_model.pth')
                torch.save(network.state_dict(), bestsave_path)
    
            lastsave_path=os.path.join(save_models_path,'last_model.pth')
            torch.save(network.state_dict(), lastsave_path)

def train_vision_data(object_name):
    """
    To test different serial generalization
    """
    #1: Init all data
    dataset_path=os.path.join(WholeDatasetPath,object_name)
    object_name=dataset_path.split('/')[-1]

    NUM_batch_size=128
    NUM_workers=24
    NUM_train_epoch=50
    NUM_lowest_loss=1000
    Flag_shuffle=True

    train_dataset=VisionDataset(dataset_path,split_method='shuffle',split='train',split_rate=0.6) 
    test_dataset=VisionDataset(dataset_path,split_method='shuffle',split='test',split_rate=0.6) 
    train_loader=DataLoader(train_dataset,batch_size=NUM_batch_size,shuffle=Flag_shuffle,num_workers=NUM_workers)
    test_loader=DataLoader(test_dataset,batch_size=NUM_batch_size,shuffle=Flag_shuffle,num_workers=NUM_workers)
    
    
    network=ImageCNN(Flag_Merge=False).cuda()
    optimizer=torch.optim.Adam(network.parameters(),lr=0.001)

    
    object_point_cloud = o3d.io.read_triangle_mesh(os.path.join(Abs_Path,"ycb_models/{}/textured.obj".format(object_name)))
    object_point_cloud = object_point_cloud.sample_points_uniformly(3000)
    loss6D=Loss6D(o3d_point_cloud=object_point_cloud,num_points=3000)

    #2: Begin to train the network
    for epoch in range(0,NUM_train_epoch):

        #2.1 In training process
        sum_train_loss=0
        sum_test_loss=0
        network.train()
        for batch_idx,data in enumerate(train_loader):
            rgbd_data,target_pose=data
            rgbd_data,target_pose=rgbd_data.cuda(),target_pose.cuda()
            predict_result=network(rgbd_data.float())
            

            #optimize network
            loss=loss6D(predict_result,target_pose)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_train_loss=sum_train_loss+loss.item()

            if batch_idx%10==0:
                batch_size_fmt="{:15}\t".format("Train:"+str(batch_idx*NUM_batch_size))
                all_logger_info(batch_size_fmt+'| Loss: {:.6f}'.format(loss.item()/len(rgbd_data)))                

        #2.2 In test process
        network.eval()
        for batch_idx,data in enumerate(test_loader):
            rgbd_data,target_pose=data
            rgbd_data,target_pose=rgbd_data.cuda(),target_pose.cuda()
            predict_result=network(rgbd_data.float())
            loss=loss6D(predict_result,target_pose)
            sum_test_loss=sum_test_loss+loss.item()

            if batch_idx%10==0:
                batch_size_fmt="{:15}\t".format("Test:"+str(batch_idx*NUM_batch_size))
                all_logger_info(batch_size_fmt+'| Loss: {:.6f}'.format(loss.item()/len(rgbd_data)))

        #3: Save result
        average_train_loss=sum_train_loss/(len(train_loader)*NUM_batch_size)
        average_valid_loss=sum_test_loss/(len(test_loader)*NUM_batch_size)

        epoch_fmt="{:10}\t".format("Epoch:"+str(epoch))
        train_logger_info(epoch_fmt+"|Train loss: {:.6f} Test loss: {:.6f}".format(average_train_loss,average_valid_loss))
        all_logger_info(epoch_fmt+"|Train loss: {:.6f} Test loss: {:.6f}".format(average_train_loss,average_valid_loss))

        if Record_Flag:
            if average_valid_loss<NUM_lowest_loss:
                NUM_lowest_loss=average_valid_loss
                all_logger_info("!!!!Find new lowest loss in epoch{},loss is:{}!!!!!".format(epoch,average_valid_loss))
                bestsave_path=os.path.join(save_models_path,'best_model.pth')
                torch.save(network.state_dict(), bestsave_path)
    
            lastsave_path=os.path.join(save_models_path,'last_model.pth')
            torch.save(network.state_dict(), lastsave_path)

def train_merge_data(object_name):
    """
    To test different serial generalization
    """
    #1: Init all data
    dataset_path=os.path.join(WholeDatasetPath,object_name)
    object_name=dataset_path.split('/')[-1]

    NUM_batch_size=128
    NUM_workers=24
    NUM_train_epoch=50
    NUM_lowest_loss=1000
    Flag_shuffle=True

    train_dataset=MergeDataset(dataset_path,split_method='shuffle',split='train',split_rate=0.6)
    test_dataset=MergeDataset(dataset_path,split_method='shuffle',split='test',split_rate=0.6)
    train_loader=DataLoader(train_dataset,batch_size=NUM_batch_size,shuffle=Flag_shuffle,num_workers=NUM_workers)
    test_loader=DataLoader(test_dataset,batch_size=NUM_batch_size,shuffle=Flag_shuffle,num_workers=NUM_workers)
    
    
    network=MergeModel().cuda()
    optimizer=torch.optim.Adam(network.parameters(),lr=0.001)

    
    object_point_cloud = o3d.io.read_triangle_mesh(os.path.join(Abs_Path,"ycb_models/{}/textured.obj".format(object_name)))
    object_point_cloud = object_point_cloud.sample_points_uniformly(3000)
    loss6D=Loss6D(o3d_point_cloud=object_point_cloud,num_points=3000)
    
    edge_index=torch.tensor([[0,1,2,3,4],[5,5,5,5,5]],dtype=torch.long).cuda()#the last pose is BaseTHand

    #2: Begin to train the network
    for epoch in range(0,NUM_train_epoch):

        #2.1 In training process
        sum_train_loss=0
        sum_test_loss=0
        network.train()
        for batch_idx,data in enumerate(train_loader):
            tip_pose_array,tip_data_array,rgbd_data,target_pose=data
            tip_pose_array,tip_data_array,rgbd_data,target_pose=tip_pose_array.cuda(),tip_data_array.cuda(),rgbd_data.cuda(),target_pose.cuda()

            #Generate batch graph data
            batch_graphdata_list=[]
            for index,tip_pose in enumerate(tip_pose_array):
                tactile_data=tip_data_array[index]
                graphData=GraphData(x=tip_pose,edge_index=edge_index,edge_attr=tactile_data)
                batch_graphdata_list.append(graphData)
            input_tactile_data=Batch.from_data_list(batch_graphdata_list)
            
            predict_result=network(rgbd_data,input_tactile_data)
            

            #optimize network
            loss=loss6D(predict_result,target_pose)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_train_loss=sum_train_loss+loss.item()

            if batch_idx%10==0:
                batch_size_fmt="{:15}\t".format("Train:"+str(batch_idx*NUM_batch_size))
                all_logger_info(batch_size_fmt+'| Loss: {:.6f}'.format(loss.item()/len(rgbd_data)))                

        #2.2 In test process
        network.eval()
        for batch_idx,data in enumerate(test_loader):
            tip_pose_array,tip_data_array,rgbd_data,target_pose=data
            tip_pose_array,tip_data_array,rgbd_data,target_pose=tip_pose_array.cuda(),tip_data_array.cuda(),rgbd_data.cuda(),target_pose.cuda()

            #Generate batch graph data
            batch_graphdata_list=[]
            for index,tip_pose in enumerate(tip_pose_array):
                tactile_data=tip_data_array[index]
                graphData=GraphData(x=tip_pose,edge_index=edge_index,edge_attr=tactile_data)
                batch_graphdata_list.append(graphData)
            input_tactile_data=Batch.from_data_list(batch_graphdata_list)
            
            predict_result=network(rgbd_data,input_tactile_data)
            loss=loss6D(predict_result,target_pose)
            sum_test_loss=sum_test_loss+loss.item()

            if batch_idx%10==0:
                batch_size_fmt="{:15}\t".format("Test:"+str(batch_idx*NUM_batch_size))
                all_logger_info(batch_size_fmt+'| Loss: {:.6f}'.format(loss.item()/len(rgbd_data)))


        #3: Save result
        average_train_loss=sum_train_loss/(len(train_loader)*NUM_batch_size)
        average_valid_loss=sum_test_loss/(len(test_loader)*NUM_batch_size)

        epoch_fmt="{:10}\t".format("Epoch:"+str(epoch))
        train_logger_info(epoch_fmt+"|Train loss: {:.6f} Test loss: {:.6f}".format(average_train_loss,average_valid_loss))
        all_logger_info(epoch_fmt+"|Train loss: {:.6f} Test loss: {:.6f}".format(average_train_loss,average_valid_loss))

        if Record_Flag:
            if average_valid_loss<NUM_lowest_loss:
                NUM_lowest_loss=average_valid_loss
                all_logger_info("!!!!Find new lowest loss in epoch{},loss is:{}!!!!!".format(epoch,average_valid_loss))
                bestsave_path=os.path.join(save_models_path,'best_model.pth')
                torch.save(network.state_dict(), bestsave_path)
    
            lastsave_path=os.path.join(save_models_path,'last_model.pth')
            torch.save(network.state_dict(), lastsave_path)

def train_selectLSTM_data(object_name):
    NUM_batch_size=1
    NUM_workers=24
    NUM_train_epoch=100
    # NUM_lowest_loss=1000
    NUM_highest_acc=0
    Flag_shuffle=True
    NUM_train_length=20


    dataset_path=os.path.join(WholeDatasetPath,object_name)
    train_dataset=SelectDataset(dataset_path,split_method="shuffle",split='train',split_rate=0.6,train_length=NUM_train_length)
    test_dataset=SelectDataset(dataset_path,split_method="shuffle",split='test',split_rate=0.6,train_length=NUM_train_length)
    train_loader=DataLoader(train_dataset,batch_size=NUM_batch_size,shuffle=Flag_shuffle,num_workers=NUM_workers)
    test_loader=DataLoader(test_dataset,batch_size=NUM_batch_size,shuffle=Flag_shuffle,num_workers=NUM_workers)

    network=SelectLSTM().cuda()
    lr=0.0001
    optimizer=torch.optim.Adam(network.parameters(),lr=lr)
    
    for epoch in range(0,NUM_train_epoch):
        sum_train_loss=0
        sum_test_loss=0
        
        train_acc=0
        test_acc=0
        network.train()

        for batch_idx,data in enumerate(train_loader):
            poses_array,feature_array,label_array=data
            poses_array,feature_array,label_array=poses_array.cuda(),feature_array.cuda(),label_array.cuda()            
            optimizer.zero_grad()            
            predict_data=network([feature_array,poses_array])
            loss=F.cross_entropy(predict_data,label_array[:,-1])#Label choose last one
            loss.backward()
            optimizer.step()

            sum_train_loss=sum_train_loss+loss.item()
            train_acc=train_acc+(torch.argmax(predict_data)==label_array[:,-1]).sum()
            
            if batch_idx%200==0:
                batch_size_fmt="{:15}\t".format("Train:"+str(batch_idx*NUM_batch_size))
                all_logger_info(batch_size_fmt+'| Loss: {:.6f}'.format(loss.item()))
            

        network.eval()
        for batch_idx,data in enumerate(test_loader):
            poses_array,feature_array,label_array=data
            poses_array,feature_array,label_array=poses_array.cuda(),feature_array.cuda(),label_array.cuda()
            
            predict_data=network([feature_array,poses_array])
            loss=F.cross_entropy(predict_data,label_array[:,-1])#Label choose last one
            
            sum_test_loss=sum_test_loss+loss.item()
            test_acc=test_acc+(torch.argmax(predict_data)==label_array[:,-1]).sum()


            if batch_idx%200==0:
                batch_size_fmt="{:15}\t".format("Test:"+str(batch_idx*NUM_batch_size))
                all_logger_info(batch_size_fmt+'| Loss: {:.6f}'.format(loss.item()))
                
        #Save result
        average_train_loss=sum_train_loss/(len(train_loader)*NUM_batch_size)
        average_valid_loss=sum_test_loss/(len(test_loader)*NUM_batch_size)
        average_train_acc=train_acc/(len(train_loader)*NUM_batch_size)
        average_test_acc=test_acc/(len(test_loader)*NUM_batch_size)

        epoch_fmt="{:10}\t".format("Epoch:"+str(epoch))
        train_logger_info(epoch_fmt+"|Train loss acc: {:.6f} {:.4f} Test loss acc: {:.6f} {:.4f}".format(average_train_loss,average_train_acc,average_valid_loss,average_test_acc))
        all_logger_info(epoch_fmt+"|Train loss acc: {:.6f} {:.4f} Test loss acc: {:.6f} {:.4f}".format(average_train_loss,average_train_acc,average_valid_loss,average_test_acc))


        if Record_Flag:
            if average_test_acc>NUM_highest_acc:
                NUM_highest_acc=average_test_acc
                all_logger_info("!!!!Find new lowest loss in epoch{},loss is:{} acc is: {:.4f}!!!!!".format(epoch,average_valid_loss,average_test_acc))
                bestsave_path=os.path.join(save_models_path,'best_model.pth')
                torch.save(network.state_dict(), bestsave_path)

            
            lastsave_path=os.path.join(save_models_path,'last_model.pth')
            torch.save(network.state_dict(), lastsave_path)


if __name__ == "__main__":
    train_tactile_data(object_name)
    # train_vision_data(object_name)
    # train_merge_data(object_name)
    
    #########generate select data by test_network.py########### 
    ###           function:generate_select_data()           ###
    #########generate select data by test_network.py########### 
    # train_selectLSTM_data(object_name)
    
    
