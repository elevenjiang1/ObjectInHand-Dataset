import os
import copy
import glob
import yaml
import cv2 as cv
import numpy as np
import open3d as o3d

import torch
from torch.utils.data import DataLoader
from torch_geometric.data.batch import Batch
from torch_geometric.data.batch import Data as GraphData

from model import TactileGCN,MergeModel,ImageCNN,SelectLSTM
from pose_loss import Loss6D
from dataset import VisionDataset,MergeDataset,TactileDataset,SelectDataset


Abs_Path=os.path.dirname(os.path.abspath(__file__))

torch.set_printoptions(
    precision=2,    
    threshold=1000,
    edgeitems=3,
    linewidth=150,  
    profile=None,
    sci_mode=False  
)

np.set_printoptions(precision=3,suppress=True)

###################################Basic function###################################
def matrix_from_quaternion(quaternion, pos=None):
    """
    Return homogeneous rotation matrix from quaternion.
    Input is qx qy qz qw
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    q = q[[3, 0, 1, 2]]   # to w x y z

    if pos is None: pos = np.zeros(3)

    n = np.dot(q, q)
    if n < 1e-20:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], pos[0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], pos[1]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], pos[2]],
        [0.0, 0.0, 0.0, 1.0]])

def quaternion_from_matrix(matrix, isprecise=False):
	"""Return quaternion from rotation matrix.

	LAYOUT : [X, Y, Z, W]

	If isprecise is True, the input matrix is assumed to be a precise rotation
	matrix and a faster algorithm is used.
	"""
	M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
	if isprecise:
		q = np.empty((4,))
		t = np.trace(M)
		if t > M[3, 3]:
			q[0] = t
			q[3] = M[1, 0] - M[0, 1]
			q[2] = M[0, 2] - M[2, 0]
			q[1] = M[2, 1] - M[1, 2]
		else:
			i, j, k = 1, 2, 3
			if M[1, 1] > M[0, 0]:
				i, j, k = 2, 3, 1
			if M[2, 2] > M[i, i]:
				i, j, k = 3, 1, 2
			t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
			q[i] = t
			q[j] = M[i, j] + M[j, i]
			q[k] = M[k, i] + M[i, k]
			q[3] = M[k, j] - M[j, k]
		q *= 0.5 / np.sqrt(t * M[3, 3])
	else:
		m00 = M[0, 0]
		m01 = M[0, 1]
		m02 = M[0, 2]
		m10 = M[1, 0]
		m11 = M[1, 1]
		m12 = M[1, 2]
		m20 = M[2, 0]
		m21 = M[2, 1]
		m22 = M[2, 2]
		# symmetric matrix K
		K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
					  [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
					  [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
					  [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
		K /= 3.0
		# quaternion is eigenvector of K that corresponds to largest eigenvalue
		w, V = np.linalg.eigh(K)
		q = V[:, np.argmax(w)]
		if q[0] < 0.0:
			np.negative(q, q)
	return q

def project_points(points, K):
    us = np.divide(points[:, 0]*K[0, 0], points[:, 2]) + K[0, 2]
    vs = np.divide(points[:, 1]*K[1, 1], points[:, 2]) + K[1, 2]
    us = np.round(us).astype(np.int32).reshape(-1, 1)
    vs = np.round(vs).astype(np.int32).reshape(-1, 1)
    return np.hstack((us, vs))

def check_noralize(quaternions):
    #检查四元数是否正确归一化
    temp = np.array([quaternions[0], quaternions[1], quaternions[2], quaternions[3]])
    if abs(np.sum(np.square(temp)) -1) < 0.01:
        print("***correct is: {}***".format(np.sum(np.square(temp)) -1))
        return True
    print("sum of quaterion is:{}".format(np.sum(np.square(temp))))
    return False

def tactile_np2tensor(tip_data_array,tip_pose_array):
    tensor_tip_data_array=torch.from_numpy(tip_data_array[np.newaxis,:]).cuda()
    tensor_tip_pose_array=torch.from_numpy(tip_pose_array[np.newaxis,:]).cuda()

    return [tensor_tip_pose_array,tensor_tip_data_array]

def tensor2np(tensor):
    return tensor.detach().cpu().numpy()

def pose_matrix2pose_vector(pose_matrix):
    quaternion=quaternion_from_matrix(pose_matrix)
    xyz=pose_matrix[:3,3]
    pose_vector=np.concatenate([xyz,quaternion])
    return pose_vector

def get_pose_from_yaml(yaml_path):
    """
    The yaml file should be like:

    pose:
    position:
        x: -0.4114404581059961
        y: 0.03589671794642055
        z: 0.5614216934472097
    orientation:
        x: -0.6282095478388904
        y: 0.007425104192635422
        z: 0.03591101479832567
        w: 0.7771795357881858
    """
    data_dict=yaml.load(open(yaml_path))['pose']
    pos=np.array([data_dict['position']['x'],data_dict['position']['y'],data_dict['position']['z']])
    ori=np.array([data_dict['orientation']['x'],data_dict['orientation']['y'],data_dict['orientation']['z'],data_dict['orientation']['w']])
    return np.concatenate([pos,ori],axis=0)

def compute_bbox(pose, K, scale_size=285.37860394994397, scale=(1, 1, 1)):
    obj_x = pose[0, 3] * scale[0]
    obj_y = pose[1, 3] * scale[1]
    obj_z = pose[2, 3] * scale[2]
    offset = scale_size / 2
    points = np.ndarray((4, 3), dtype=np.float32)
    points[0] = [obj_x - offset, obj_y - offset, obj_z]     # top left
    points[1] = [obj_x - offset, obj_y + offset, obj_z]     # top right
    points[2] = [obj_x + offset, obj_y - offset, obj_z]     # bottom left
    points[3] = [obj_x + offset, obj_y + offset, obj_z]     # bottom right
    projected_vus = np.zeros((points.shape[0], 2))
    projected_vus[:, 1] = points[:, 0] * K[0, 0] / points[:, 2] + K[0, 2]
    projected_vus[:, 0] = points[:, 1] * K[1, 1] / points[:, 2] + K[1, 2]
    projected_vus = np.round(projected_vus).astype(np.int32)
    return projected_vus

def crop_bbox(color, depth, boundingbox, output_size=(128, 128), seg=None):
    left = np.min(boundingbox[:, 1])
    right = np.max(boundingbox[:, 1])
    top = np.min(boundingbox[:, 0])
    bottom = np.max(boundingbox[:, 0])

    h, w, c = color.shape
    crop_w = right - left
    crop_h = bottom - top
    color_crop = np.zeros((crop_h, crop_w, 3), dtype=color.dtype)
    depth_crop = np.zeros((crop_h, crop_w), dtype=np.float32)
    seg_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
    top_offset = abs(min(top, 0))
    bottom_offset = min(crop_h - (bottom - h), crop_h)
    right_offset = min(crop_w - (right - w), crop_w)
    left_offset = abs(min(left, 0))

    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, h)
    right = min(right, w)
    color_crop[top_offset:bottom_offset, left_offset:right_offset,
               :] = color[top:bottom, left:right, :]
    depth_crop[top_offset:bottom_offset,
               left_offset:right_offset] = depth[top:bottom, left:right]
    resized_rgb = cv.resize(color_crop, output_size,
                             interpolation=cv.INTER_NEAREST)
    resized_depth = cv.resize(
        depth_crop, output_size, interpolation=cv.INTER_NEAREST)

    if seg is not None:
        seg_crop[top_offset:bottom_offset,
                 left_offset:right_offset] = seg[top:bottom, left:right]
        resized_seg = cv.resize(
            seg_crop, output_size, interpolation=cv.INTER_NEAREST)
        final_seg = resized_seg.copy()

    mask_rgb = resized_rgb != 0
    mask_depth = resized_depth != 0
    resized_depth = resized_depth.astype(np.uint16)
    final_rgb = resized_rgb * mask_rgb
    final_depth = resized_depth * mask_depth
    if seg is not None:
        return final_rgb, final_depth, final_seg
    else:
        return final_rgb, final_depth

def get_colormap_from_depth(depth):
    cv_image=depth.copy()
    # ROI=cv2.inRange(cv_image, 0, 1500)
    # cv_image=cv_image*ROI/255
    cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
    cv_image=cv_image.astype(np.uint8)
    color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)
    return color_map

def pose_matrix2pose_vector(pose_matrix):
    quaternion=quaternion_from_matrix(pose_matrix)
    xyz=pose_matrix[:3,3]
    pose_vector=np.concatenate([xyz,quaternion])
    return pose_vector

def clean_dataset():
    """
    To see different serial
    0-349
    350-864
    865-2107
    2108-2266
    2267-4457
    """
    dataset_path="/home/Project/Code/code/workspace/src/seedata/scripts/data/WholeDataset/BigDataset"
    all_index_data=glob.glob(dataset_path+"/color*")
    # print(all_index_data)
    
    all_count=0
    while all_count<len(all_index_data):
        print("Now is index {} data".format(all_count))
        image=cv.imread(os.path.join(dataset_path,"color_{}.png".format(all_count)))
        cv.imshow("image",image)
        temp_input=cv.waitKey(0)
        if temp_input==ord('b'):
            all_count=all_count-5
        all_count=all_count+1

####################################Calculate error metric####################################
def generate_select_data():
    #1: Load imageCNN, TactileGCN, MergeNetwork
    #Load network
    tactileGCN=TactileGCN(Flag_Merge=True).cuda()
    imageCNN=ImageCNN(Flag_Merge=True).cuda()
    mergeModel=MergeModel(Flag_Merge=True).cuda()

    #Load trained result
    tactileGCN.load_state_dict(torch.load("/home/Project/Code/ObjectInHand/fast_merge/log/Baseline-experiments-sugar/Tactile--02-27_01-51/models/best_model.pth"))
    tactileGCN.eval()
    imageCNN.load_state_dict(torch.load("/home/Project/Code/ObjectInHand/fast_merge/log/Baseline-experiments-sugar/Vision--02-27_01-55/models/best_model.pth"))
    imageCNN.eval()
    mergeModel.load_state_dict(torch.load("/home/Project/Code/ObjectInHand/fast_merge/log/Baseline-experiments-sugar/Merge--02-27_10-53/models/best_model.pth"))
    mergeModel.eval()

    #2: Load all data, and infer all result
    object_name="sugar"
    dataset_path="/home/media/WholeDataset/{}".format(object_name)
    
    
    object_point_cloud = o3d.io.read_triangle_mesh(os.path.join(Abs_Path,"ycb_models/{}/textured.obj".format(object_name)))
    object_point_cloud = object_point_cloud.sample_points_uniformly(3000)
    loss6D=Loss6D(o3d_point_cloud=object_point_cloud,num_points=3000)

    #Load all data by split_index
    all_dataset=MergeDataset(dataset_path,split_method='index',split='train',split_index_list=[0,10000]) 
    all_loader=DataLoader(all_dataset,batch_size=1,shuffle=False)
    edge_index=torch.tensor([[0,1,2,3,4],[5,5,5,5,5]],dtype=torch.long).cuda()#the last pose is BaseTHand


    #3: Save 3 pose, 3 feature vector, and classify result
    save_all_data_dict={}
    tactile_pose_list=[]
    tactile_feature_list=[]
    image_pose_list=[]
    image_feature_list=[]
    merge_pose_list=[]
    merge_feature_list=[]
    target_pose_list=[]
    all_label_list=[]
    

    for batch_idx,data in enumerate(all_loader):
        tip_pose_array,tip_data_array,rgbd_data,target_pose=data
        tip_pose_array,tip_data_array,rgbd_data,target_pose=tip_pose_array.cuda(),tip_data_array.cuda(),rgbd_data.cuda(),target_pose.cuda()

        #Generate batch graph data
        batch_graphdata_list=[]
        for index,tip_pose in enumerate(tip_pose_array):
            tactile_data=tip_data_array[index]
            graphData=GraphData(x=tip_pose,edge_index=edge_index,edge_attr=tactile_data)
            batch_graphdata_list.append(graphData)
        input_tactile_data=Batch.from_data_list(batch_graphdata_list)

        #three network infer
        tactile_pose,tactile_feature=tactileGCN(input_tactile_data)
        image_pose,image_feature=imageCNN(rgbd_data)
        merge_pose,merge_feature=mergeModel(rgbd_data,input_tactile_data)
        tactile_loss=loss6D(tactile_pose,target_pose)
        image_loss=loss6D(image_pose,target_pose)
        merge_loss=loss6D(merge_pose,target_pose)
        
        min_loss=torch.min(torch.Tensor([tactile_loss,image_loss,merge_loss]))
        
        #Find mini loss between target_pose and predict_pose
        if min_loss==tactile_loss:
            all_label_list.append(0)
        elif min_loss==image_loss:
            all_label_list.append(1)
        elif min_loss==merge_loss:
            all_label_list.append(2)
            
        #change back to numpy and save all data
        tactile_pose=tactile_pose.cpu().detach().numpy()
        tactile_feature=tactile_feature.cpu().detach().numpy()
        image_pose=image_pose.cpu().detach().numpy()
        image_feature=image_feature.cpu().detach().numpy()
        merge_pose=merge_pose.cpu().detach().numpy()
        merge_feature=merge_feature.cpu().detach().numpy()
        target_pose=target_pose.cpu().detach().numpy()

        #save all data to list
        tactile_pose_list.append(tactile_pose)
        tactile_feature_list.append(tactile_feature)
        image_pose_list.append(image_pose)
        image_feature_list.append(image_feature)
        merge_pose_list.append(merge_pose)
        merge_feature_list.append(merge_feature)
        target_pose_list.append(target_pose)


    all_tactile_pose=np.concatenate(tactile_pose_list)
    all_tactile_feature=np.concatenate(tactile_feature_list)
    all_image_pose=np.concatenate(image_pose_list)
    all_image_feature=np.concatenate(image_feature_list)
    all_merge_pose=np.concatenate(merge_pose_list)
    all_merge_feature=np.concatenate(merge_feature_list)
    all_target_pose=np.concatenate(target_pose_list)
    all_label=np.array(all_label_list)

    save_all_data_dict['tactile_pose']=all_tactile_pose
    save_all_data_dict['tactile_feature']=all_tactile_feature
    save_all_data_dict['image_pose']=all_image_pose
    save_all_data_dict['image_feature']=all_image_feature
    save_all_data_dict['merge_pose']=all_merge_pose
    save_all_data_dict['merge_feature']=all_merge_feature
    save_all_data_dict['target_pose']=all_target_pose
    save_all_data_dict['all_label']=all_label
    
    
    count_0=all_label[all_label==0]
    count_1=all_label[all_label==1]
    count_2=all_label[all_label==2]
    print("Three type labels are:")
    print(count_0.shape)
    print(count_1.shape)
    print(count_2.shape)

    np.save(os.path.join(Abs_Path,"selectlstm_data/select_data_{}.npy".format(object_name)),save_all_data_dict)


####################################Calculate error metric####################################
def normal_quaternion(quaternion):
    return quaternion/(np.sqrt(np.sum(np.square(quaternion))))

def calculate_diff(pose1,pose2):
    """
    Reference metric in "VisuoTactile 6D Pose Estimation of an In-Hand Object Using Vision and Tactile Sensor Data"
    #return in m,rad
    """
    xyz_diff=np.linalg.norm(pose1[:3]-pose2[:3])
    quaternion_diff=np.arccos(2*np.square(np.dot(normal_quaternion(pose1[3:]),normal_quaternion(pose2[3:])))-1)
    
    return xyz_diff,quaternion_diff


###################################See data Distribution###################################
def see_error_rate_result():
    pass


###################################To test network generalization###################################
def test_tactile_data():
    object_name="pear"
    dataset_path="/home/media/WholeDataset/{}".format(object_name)

    NUM_batch_size=1
    NUM_workers=12
    Flag_shuffle=False

    all_dataset=TactileDataset(dataset_path,split_method='index',split='train',split_index_list=[0,100000]) 
    all_loader=DataLoader(all_dataset,batch_size=NUM_batch_size,shuffle=Flag_shuffle,num_workers=NUM_workers)
    edge_index=torch.tensor([[0,1,2,3,4],[5,5,5,5,5]],dtype=torch.long).cuda()#the last pose is BaseTHand
    
    network=TactileGCN().cuda()
    network.load_state_dict(torch.load("/home/Project/Code/ObjectInHand/fast_merge/log/Baseline-experiments-pear/2023-02-25_02-10/models/best_model.pth"))
    network.eval()
        
    all_predict_result_list=[]
    all_diff_result_list=[]
    
    for index,data in enumerate(all_loader):
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

        #get all results
        np_predict_result=predict_result.cpu().detach().numpy().squeeze()
        np_target_pose=target_pose.cpu().detach().numpy().squeeze()
        all_predict_result_list.append(np_predict_result)
        all_diff_result_list.append(calculate_diff(np_predict_result,np_target_pose))
        
    all_predict_array=np.array(all_predict_result_list)
    all_diff_result=np.array(all_diff_result_list)
    
    train_dataset=TactileDataset(dataset_path,split_method='shuffle',split='train',split_rate=0.8) 
    train_index=train_dataset.index_list
    test_dataset=TactileDataset(dataset_path,split_method='shuffle',split='test',split_rate=0.8) 
    test_index=test_dataset.index_list
    
    print("***********Object:{} all results are:***********".format(object_name))
    print("All average result are:")
    print(np.mean(all_diff_result,axis=0))
    
    print("train average result are:")
    train_result=all_diff_result[train_index]
    print(np.mean(train_result,axis=0))
    
    print("test average result are:")
    test_result=all_diff_result[test_index]
    print(test_result.shape)
    print(np.mean(test_result,axis=0))
    
    
    np.savez(os.path.join(Abs_Path,"tactile_result.npz"),all_predict_array=all_predict_array,all_diff_result=all_diff_result)

def test_vision_data():
    #1: Init the network
    # dataset_path="/home/media/WholeDataset/cup"
    dataset_path="/home/media/WholeDataset/pear"
    object_name=dataset_path.split('/')[-1]

    network=ImageCNN().cuda()
    network.load_state_dict(torch.load("/home/Project/Code/ObjectInHand/fast_merge/log/Baseline-experiments-pear/Vision--02-25_15-46/models/best_model.pth"))
    network.eval()

    all_predict_result_list=[]
    all_diff_result_list=[]
    
    #2: load index data
    all_dataset=VisionDataset(dataset_path,split_method='index',split='train',split_index_list=[0,10000]) 
    all_loader=DataLoader(all_dataset,batch_size=1,shuffle=False)
    for batch_idx,data in enumerate(all_loader):
        rgbd_data,target_pose=data
        rgbd_data,target_pose=rgbd_data.cuda(),target_pose.cuda()
        predict_result=network(rgbd_data.float())
        

        #get all results
        np_predict_result=predict_result.cpu().detach().numpy().squeeze()
        np_target_pose=target_pose.cpu().detach().numpy().squeeze()
        all_predict_result_list.append(np_predict_result)
        all_diff_result_list.append(calculate_diff(np_predict_result,np_target_pose))
    
    all_predict_array=np.array(all_predict_result_list)
    all_diff_result=np.array(all_diff_result_list)


    train_dataset=VisionDataset(dataset_path,split_method='shuffle',split='train',split_rate=0.6) 
    train_index=train_dataset.index_list
    test_dataset=VisionDataset(dataset_path,split_method='shuffle',split='test',split_rate=0.6) 
    test_index=test_dataset.index_list
    
    print("***********Object:{} all results are:***********".format(object_name))
    print("All average result are:")
    print(np.mean(all_diff_result,axis=0))
    
    print("train average result are:")
    train_result=all_diff_result[train_index]
    print(np.mean(train_result,axis=0))
    
    print("test average result are:")
    test_result=all_diff_result[test_index]
    print(test_result.shape)
    print(np.mean(test_result,axis=0))
    
    
    np.savez(os.path.join(Abs_Path,"vision_result.npz"),all_predict_array=all_predict_array,all_diff_result=all_diff_result)

def test_merge_data():
    """
    To test different serial generalization
    """
    #1: Init all data

    dataset_path="/home/media/WholeDataset/pear"
    object_name=dataset_path.split('/')[-1]

    network=MergeModel().cuda()
    network.load_state_dict(torch.load("/home/Project/Code/ObjectInHand/fast_merge/log/Baseline-experiments-pear/Merge--02-25_16-29/models/best_model.pth"))
    network.eval()

    all_predict_result_list=[]
    all_diff_result_list=[]
    
    #2: load index data
    all_dataset=MergeDataset(dataset_path,split_method='index',split='train',split_index_list=[0,10000]) 
    all_loader=DataLoader(all_dataset,batch_size=1,shuffle=False)
    
    
    edge_index=torch.tensor([[0,1,2,3,4],[5,5,5,5,5]],dtype=torch.long).cuda()#the last pose is BaseTHand
    
    all_diff_result_list=[]
    all_predict_result_list=[]
        
    for batch_idx,data in enumerate(all_loader):
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
    
        np_predict_result=predict_result.squeeze().detach().cpu().numpy()
        np_target_pose=target_pose.squeeze().detach().cpu().numpy()
        
        
        #error data is:
        diff_result=calculate_diff(np_predict_result,np_target_pose)
        all_diff_result_list.append(diff_result)
        all_predict_result_list.append(np_predict_result)
        
    all_predict_array=np.array(all_predict_result_list)
    all_diff_result=np.array(all_diff_result_list)
        
    np.savez(os.path.join(Abs_Path,"merge.npz"),all_predict_array=all_predict_array,all_diff_result=all_diff_result)
    
    
    train_dataset=MergeDataset(dataset_path,split_method='shuffle',split='train',split_rate=0.6) 
    test_dataset=MergeDataset(dataset_path,split_method='shuffle',split='test',split_rate=0.6) 
    train_index=train_dataset.index_list
    test_index=test_dataset.index_list
    
    print("All result array shape is:")
    print(all_diff_result.shape)
    
    all_diff_result=all_diff_result
    print("All average result are:")
    print(np.mean(all_diff_result,axis=0))
    
    print("train average result are:")
    train_result=all_diff_result[train_index]
    print(np.mean(train_result,axis=0))
    
    print("test average result are:")
    test_result=all_diff_result[test_index]
    print(np.mean(test_result,axis=0))
    
def test_select_LSTM():
    #1: Load network and dataset
    object_name="sugar"
    dataset_path="/home/media/WholeDataset/{}".format(object_name)
    npy_path="/home/Project/Code/ObjectInHand/fast_merge/selectlstm_data/select_data_{}.npy".format(object_name)
    save_all_data_dict=np.load(npy_path,allow_pickle=True).item()

    NUM_train_length=20
    train_dataset=SelectDataset(dataset_path,split_method="shuffle",split='train',split_rate=0.6,train_length=NUM_train_length)
    test_dataset=SelectDataset(dataset_path,split_method="shuffle",split='test',split_rate=0.6,train_length=NUM_train_length)
    train_index_list=copy.deepcopy(train_dataset.index_list)
    test_index_list=copy.deepcopy(test_dataset.index_list)

    network=SelectLSTM().cuda()
    network.load_state_dict(torch.load("/home/Project/Code/ObjectInHand/fast_merge/log/SelectRegression-experiments-sugar/Select--02-27_16-05/models/best_model.pth"))
    network.eval()
    
    all_acc_list=[]
    all_predict_class_list=[]
    
    #2.3: test all index list
    print("Processing test index list....")
    for idx,index_data in enumerate(range(train_dataset.all_merge_feature.shape[0])):
        if index_data<NUM_train_length:
            merge_feature=train_dataset.all_merge_feature[:index_data+1].astype(np.float32)
            tactile_feature=train_dataset.all_tactile_feature[:index_data+1].astype(np.float32)
            image_feature=train_dataset.all_image_feature[:index_data+1].astype(np.float32)

            merge_pose=train_dataset.all_merge_pose[:index_data+1].astype(np.float32)
            tactile_pose=train_dataset.all_tactile_pose[:index_data+1].astype(np.float32)
            densefusion_pose=train_dataset.all_image_pose[:index_data+1].astype(np.float32)
            return_label=train_dataset.all_label[index_data]
        else:
            merge_feature=train_dataset.all_merge_feature[index_data-NUM_train_length+1:index_data+1].astype(np.float32)
            tactile_feature=train_dataset.all_tactile_feature[index_data-NUM_train_length+1:index_data+1].astype(np.float32)
            image_feature=train_dataset.all_image_feature[index_data-NUM_train_length+1:index_data+1].astype(np.float32)

            merge_pose=train_dataset.all_merge_pose[index_data-NUM_train_length+1:index_data+1].astype(np.float32)
            tactile_pose=train_dataset.all_tactile_pose[index_data-NUM_train_length+1:index_data+1].astype(np.float32)
            densefusion_pose=train_dataset.all_image_pose[index_data-NUM_train_length+1:index_data+1].astype(np.float32)
            return_label=train_dataset.all_label[index_data]
    

        input_pose_data=np.concatenate([merge_pose,tactile_pose,densefusion_pose],axis=1)
        input_feature_data=np.concatenate([merge_feature,tactile_feature,image_feature],axis=1)

        input_tensor_pose_data=torch.from_numpy(input_pose_data).cuda().unsqueeze(0)
        input_tensor_feature_data=torch.from_numpy(input_feature_data).cuda().unsqueeze(0)
        input_tensor_label=torch.from_numpy(np.array(return_label)).cuda()
        

        predict_data=network([input_tensor_feature_data,input_tensor_pose_data])
        predict_class=torch.argmax(predict_data)
        if input_tensor_label==predict_class:
            all_acc_list.append(1)
        else:
            all_acc_list.append(0)

        np_predict_class=predict_class.cpu().detach().numpy()
        all_predict_class_list.append(np_predict_class)
        
    all_acc_array=np.array(all_acc_list)
    
    print("All acc is:",np.sum(all_acc_array)/len(all_acc_list))
    print("All acc array is:")
    print(all_acc_array.shape)
    
    
    #2.4: Get train, test acc base on 
    all_predict_class_array=np.array(all_predict_class_list)
    
            
    #3: begin to test all result metric
    #3.1 load all data
    all_tactile_pose=save_all_data_dict['tactile_pose']
    all_image_pose=save_all_data_dict['image_pose']
    all_merge_pose=save_all_data_dict['merge_pose']
    all_target_pose=save_all_data_dict['target_pose']
    all_selectlstm_predict_result=all_predict_class_array
    
    
    tactile_result_list=[]
    image_result_list=[]
    merge_result_list=[]
    select_lstm_result_list=[]

    for index,true_pose in enumerate(all_target_pose):
        tactile_pose=all_tactile_pose[index]
        image_pose=all_image_pose[index]
        merge_pose=all_merge_pose[index]

        tactile_result_list.append(calculate_diff(tactile_pose,true_pose))
        image_result_list.append(calculate_diff(image_pose,true_pose))
        merge_result_list.append(calculate_diff(merge_pose,true_pose))

        selectlstm_result=all_selectlstm_predict_result[index]
        if selectlstm_result==0:
            select_lstm_result_list.append(calculate_diff(tactile_pose,true_pose))
        if selectlstm_result==1:
            select_lstm_result_list.append(calculate_diff(image_pose,true_pose))
        if selectlstm_result==2:
            select_lstm_result_list.append(calculate_diff(merge_pose,true_pose))

    #3: see all average result
    all_tactile_result=np.array(tactile_result_list)
    all_image_result=np.array(image_result_list)
    all_merge_result=np.array(merge_result_list)
    all_selectlstm_result=np.array(select_lstm_result_list)
    save_all_data_dict['tactile_result']=all_tactile_result
    save_all_data_dict['image_result']=all_image_result
    save_all_data_dict['merge_result']=all_merge_result
    save_all_data_dict['selectlstm_result']=all_selectlstm_result
    
    
    
    #4: check_all data result
    all_tactile_result=save_all_data_dict['tactile_result']
    all_image_result=save_all_data_dict['image_result']
    all_merge_result=save_all_data_dict['merge_result']
    all_selectlstm_result=save_all_data_dict['selectlstm_result']
    print("**********Object:{} all results:***********************".format(object_name))
    print("all average result are:")
    print(np.mean(all_tactile_result,axis=0))
    print(np.mean(all_image_result,axis=0))
    print(np.mean(all_merge_result,axis=0))
    print(np.mean(all_selectlstm_result,axis=0))


    #4: see train and test split result
    train_tactile_result=all_tactile_result[train_index_list]
    train_image_result=all_image_result[train_index_list]
    train_merge_result=all_merge_result[train_index_list]    
    train_selectlstm_result=all_selectlstm_result[train_index_list]
    print("all train average result are:")
    print(np.mean(train_tactile_result,axis=0))
    print(np.mean(train_image_result,axis=0))
    print(np.mean(train_merge_result,axis=0))
    print(np.mean(train_selectlstm_result,axis=0))
    

    test_tactile_result=all_tactile_result[test_index_list]
    test_image_result=all_image_result[test_index_list]
    test_merge_result=all_merge_result[test_index_list]
    test_selectlstm_result=all_selectlstm_result[test_index_list]
    print("all test average result are:")
    print(np.mean(test_tactile_result,axis=0))
    print(np.mean(test_image_result,axis=0))
    print(np.mean(test_merge_result,axis=0))
    print(np.mean(test_selectlstm_result,axis=0))


###################################To generate analysis data###################################
def get_analysis_data():
    #1: Load network and dataset
    object_name="spatula"
    dataset_path="/home/media/WholeDataset/{}".format(object_name)
    npy_path="/home/Project/Code/ObjectInHand/fast_merge/selectlstm_data/select_data_{}.npy".format(object_name)
    save_all_data_dict=np.load(npy_path,allow_pickle=True).item()
    

    NUM_train_length=20
    train_dataset=SelectDataset(dataset_path,split_method="shuffle",split='train',split_rate=0.6,train_length=NUM_train_length)
    test_dataset=SelectDataset(dataset_path,split_method="shuffle",split='test',split_rate=0.6,train_length=NUM_train_length)
    train_index_list=copy.deepcopy(train_dataset.index_list)
    test_index_list=copy.deepcopy(test_dataset.index_list)
    save_all_data_dict['train_index']=train_index_list
    save_all_data_dict['test_index']=test_index_list

    network=SelectLSTM().cuda()
    network.load_state_dict(torch.load("/home/Project/Code/ObjectInHand/fast_merge/log/Baseline-experiments-spatula/Select--02-27_01-50/models/best_model.pth"))
    network.eval()
    
    all_select_predict_class_list=[]
    all_select_predict_pose_list=[]
    all_contact_list=[]
    all_occlusion_list=[]
    
    
    tactile_result_list=[]
    image_result_list=[]
    merge_result_list=[]
    select_lstm_result_list=[]
    
    
    #2: test all index list
    all_tactile_pose=save_all_data_dict['tactile_pose']
    all_image_pose=save_all_data_dict['image_pose']
    all_merge_pose=save_all_data_dict['merge_pose']
    all_target_pose=save_all_data_dict['target_pose']
    print("Processing test index list....")
    for idx,index_data in enumerate(range(train_dataset.all_merge_feature.shape[0])):
        #2.1: Load select predict data   
        if index_data<NUM_train_length:
            merge_feature=train_dataset.all_merge_feature[:index_data+1].astype(np.float32)
            tactile_feature=train_dataset.all_tactile_feature[:index_data+1].astype(np.float32)
            image_feature=train_dataset.all_image_feature[:index_data+1].astype(np.float32)

            merge_pose=train_dataset.all_merge_pose[:index_data+1].astype(np.float32)
            tactile_pose=train_dataset.all_tactile_pose[:index_data+1].astype(np.float32)
            densefusion_pose=train_dataset.all_image_pose[:index_data+1].astype(np.float32)
            return_label=train_dataset.all_label[index_data]
        else:
            merge_feature=train_dataset.all_merge_feature[index_data-NUM_train_length+1:index_data+1].astype(np.float32)
            tactile_feature=train_dataset.all_tactile_feature[index_data-NUM_train_length+1:index_data+1].astype(np.float32)
            image_feature=train_dataset.all_image_feature[index_data-NUM_train_length+1:index_data+1].astype(np.float32)

            merge_pose=train_dataset.all_merge_pose[index_data-NUM_train_length+1:index_data+1].astype(np.float32)
            tactile_pose=train_dataset.all_tactile_pose[index_data-NUM_train_length+1:index_data+1].astype(np.float32)
            densefusion_pose=train_dataset.all_image_pose[index_data-NUM_train_length+1:index_data+1].astype(np.float32)
            return_label=train_dataset.all_label[index_data]
    
        input_pose_data=np.concatenate([merge_pose,tactile_pose,densefusion_pose],axis=1)
        input_feature_data=np.concatenate([merge_feature,tactile_feature,image_feature],axis=1)
        input_tensor_pose_data=torch.from_numpy(input_pose_data).cuda().unsqueeze(0)
        input_tensor_feature_data=torch.from_numpy(input_feature_data).cuda().unsqueeze(0)
        predict_data=network([input_tensor_feature_data,input_tensor_pose_data])
        predict_class=torch.argmax(predict_data)
        np_predict_class=predict_class.cpu().detach().numpy()
        all_select_predict_class_list.append(np_predict_class)
        
        
        #2.2 Load contact data and occlusion rate
        meta_data=np.load(os.path.join(dataset_path,"meta_{}.npz".format(index_data)))
        occlusion_rate=meta_data['occlusion_rate']
        contact_sum=meta_data['tip_contact_array']
        all_occlusion_list.append(occlusion_rate)
        all_contact_list.append(contact_sum)
        
        
        #2.3: calculate all pose error
        tactile_pose=all_tactile_pose[index_data]
        image_pose=all_image_pose[index_data]
        merge_pose=all_merge_pose[index_data]
        true_pose=all_target_pose[index_data]
        tactile_result_list.append(calculate_diff(tactile_pose,true_pose))
        image_result_list.append(calculate_diff(image_pose,true_pose))
        merge_result_list.append(calculate_diff(merge_pose,true_pose))
        
        if np_predict_class==0:
            select_lstm_result_list.append(calculate_diff(tactile_pose,true_pose))
            all_select_predict_pose_list.append(tactile_pose)
        if np_predict_class==1:
            select_lstm_result_list.append(calculate_diff(image_pose,true_pose))
            all_select_predict_pose_list.append(image_pose)
        if np_predict_class==2:
            select_lstm_result_list.append(calculate_diff(merge_pose,true_pose))
            all_select_predict_pose_list.append(merge_pose)
        
        
    #2.2 save occlusion and contact data
    all_predict_class_array=np.array(all_select_predict_class_list)
    all_select_pose=np.array(all_select_predict_pose_list)
    all_occlusion_array=np.array(all_occlusion_list)
    all_contact_array=np.array(all_contact_list)
    
    pre_saved_data_dict={}
    pre_saved_data_dict['all_occlusion_array']=all_occlusion_array
    pre_saved_data_dict['all_contact_array']=all_contact_array
    
    pre_saved_data_dict['all_tactile_pose']=save_all_data_dict['tactile_pose']
    pre_saved_data_dict['all_image_pose']=save_all_data_dict['image_pose']
    pre_saved_data_dict['all_merge_pose']=save_all_data_dict['merge_pose']
    pre_saved_data_dict['all_select_pose']=all_select_pose
    pre_saved_data_dict['all_target_pose']=save_all_data_dict['target_pose']
    pre_saved_data_dict['all_target_label']=save_all_data_dict['all_label']
    
    pre_saved_data_dict['all_tactile_result']=np.array(tactile_result_list)
    pre_saved_data_dict['all_image_result']=np.array(image_result_list)
    pre_saved_data_dict['all_merge_result']=np.array(merge_result_list)
    pre_saved_data_dict['all_select_result']=np.array(select_lstm_result_list)
    pre_saved_data_dict['all_select_label']=all_predict_class_array
    
    pre_saved_data_dict['train_index']=train_index_list
    pre_saved_data_dict['test_index']=test_index_list
    
    
    if True:
        save_target_path="/home/Project/Code/ObjectInHand/fast_merge/all_result_data/result_data_{}.npy".format(object_name)
        np.save(save_target_path,pre_saved_data_dict)
    
    
    
if __name__ == "__main__":
    generate_select_data()
    
    # test_tactile_data()
    # test_vision_data()
    # test_merge_data()
    # test_select_LSTM()
    
    # get_analysis_data()


