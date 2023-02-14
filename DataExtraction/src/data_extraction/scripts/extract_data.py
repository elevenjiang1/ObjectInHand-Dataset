#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import png
import math
import yaml
import copy
import time
import glob
import shutil
import cv2 as cv
import numpy as np
import open3d as o3d


import tf
import rospy
from cv_bridge import CvBridge
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseArray
from sr_robot_msgs.msg import BiotacAll
from sensor_msgs.msg import Image,JointState
from tams_biotac.msg import Contact,ContactArray

Abs_Path=os.path.dirname(os.path.abspath(__file__))


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


###############################Class for extract data###############################
class ExtractData:
    def __init__(self,object_name,Flag_save_data=False,init_node=True):
        """
        Class for check and extract data
        :param: object_name: object class in data file; !!!IF dataset wrong, pleach check the path!!!
        :param: Flag_save_data: Flag for save data
        :param: init_node: whether for init ros node
        """
        if init_node:
            rospy.init_node("ExtractData")

        #Update Fix transform
        self.object_name=object_name
        self.init_config_data(self.object_name)

        #Set save flag
        self.Flag_save_data=Flag_save_data
        self.save_lock=False#to avoid conflit in self.big_save_list update

        #For image
        self.bridge=CvBridge()

        #For tf
        self.tf_linstener=tf.TransformListener()
        self.tf_broadcaster=tf.TransformBroadcaster()
        self.five_tip_name=['rh_thdistal_J1_dummy','rh_ff_J1_dummy','rh_mf_J1_dummy','rh_rf_J1_dummy','rh_lf_J1_dummy']
        self.five_contact_name={"rh_th_biotac_link":0,"rh_ff_biotac_link":1, "rh_mf_biotac_link":2,"rh_rf_biotac_link":3, "rh_lf_biotac_link":4}
        self.shadowhand_base_link_name="rh_forearm"
        self.basefootprint_link_name="base_footprint"
        
        #Pre save data
        self._save_lock=False
        self._update_color_image=None
        self._update_depth_image=None
        self._update_tip_data=None
        self._update_tip_pose_array=None
        self._update_tip_contact_array=None
        self._update_joints_state=None
        self._update_pose_array=None
        self.big_save_list=[]

        #For tactile
        while not rospy.is_shutdown():
            print("Please play rosbag in 10s, waiting the transform...")
            self.tf_linstener.waitForTransform('rh_thdistal_J1_dummy','rh_ff_J1_dummy',rospy.Time(0),rospy.Duration(10.0))
            
            #Update base_T_shandohand
            trans,rot=self.tf_linstener.lookupTransform(self.basefootprint_link_name,self.shadowhand_base_link_name,rospy.Time(0))
            self.basefootprint_T_hand_pose=np.array([trans[0],trans[1],trans[2],rot[0],rot[1],rot[2],rot[3]])#in xyz qx,qy,qz,qw
            self.matrix_basefootprint_T_hand=matrix_from_quaternion(self.basefootprint_T_hand_pose[3:],pos=self.basefootprint_T_hand_pose[:3])
            self.FIX_camera_T_hand=np.matmul(np.linalg.inv(self.FIX_basefootprint_T_camera),self.matrix_basefootprint_T_hand)
            
            break

        #Update RGBD data
        rospy.Subscriber("/camera/color/image_raw",Image,self.update_color_image_cb,queue_size=1)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw',Image,self.update_depth_image_cb,queue_size=1)
        
        #Update Pose array
        rospy.Subscriber("/pose_array",PoseArray,self.update_pose_array_cb,queue_size=1)
        
        #Update tip data
        rospy.Subscriber("/hand/rh/tactile_filtered",BiotacAll,self.update_tip_data_cb,queue_size=1)

        #Update tip_contact data
        rospy.Subscriber("/hand/rh/contacts",ContactArray,self.update_tip_contact_cb,queue_size=1)

        #Update tf transform
        rospy.Subscriber("/tf",TFMessage,self.update_tip_pose_cb,queue_size=1)
        
        #Update joints_states
        rospy.Subscriber("/joint_states",JointState,self.update_joints_state_cb,queue_size=1)
        


    def beautiful_print(self,data):
        print("!"*(len(data)+6))
        print("!!!"+data+"!!!")
        print("!"*(len(data)+6))

    def get_pose_from_yaml(self,yaml_path):
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
    
    def get_colormap_from_depth(self,depth):
        cv_image=depth.copy()
        ROI=cv.inRange(cv_image, 0, 1500)
        cv_image=cv_image*ROI/255
        cv.normalize(cv_image,cv_image,255,0,cv.NORM_MINMAX)
        cv_image=cv_image.astype(np.uint8)
        color_map=cv.applyColorMap(cv_image,cv.COLORMAP_JET)
        return color_map

    def init_config_data(self,object_name,dataset_path=None):
        """
        Init object pointcloud and calibration transform
        dataset_path should be like:
        2022-0827/
        ├── cleanser
        │   ├── 2022-08-14-15-03-39-processed.bag
        │   └── config
        │       ├── camera_transmitter.yaml
        │       ├── merged_cloud_cleanser.ply
        │       └── sensor1_object_cleanser.yaml
        ├── cracker
        │   ├── 2022-08-14-16-00-16-processed.bag
        │   ├── config
        │   │   ├── camera_transmitter.yaml
        │   │   ├── merged_cloud_cracker.ply
        │   │   └── sensor1_object_cracker.yaml
        ├── cup
        │   ├── 2022-08-14-17-07-08-processed.bag
        │   ├── config
        │   │   ├── camera_transmitter.yaml
        │   │   ├── merged_cloud_cup.ply
        │   │   └── sensor1_object_cup.yaml
        ...
        *.bag file is rosbag
        config file include *.ply for object pointcloud, *.yaml for sensor calibration transform
        """
        if dataset_path is None:
            dataset_path="/home/media/2022-0827"
        config_file_path="{}/{}/config".format(dataset_path,object_name)
        if not os.path.exists(config_file_path):
            self.beautiful_print("!!!Config path: {} not exist!!!".format(config_file_path))
            sys.exit()

        #Init sensorTobject
        camera_transmitter=self.get_pose_from_yaml("{}/{}/config/camera_transmitter.yaml".format(dataset_path,object_name))
        self.FIX_camera_T_transmitter=matrix_from_quaternion(quaternion=camera_transmitter[3:],pos=camera_transmitter[:3])
        sensor_object=self.get_pose_from_yaml("{}/{}/config/sensor1_object_{}.yaml".format(dataset_path,object_name,object_name))
        self.FIX_sensor_T_object=matrix_from_quaternion(quaternion=sensor_object[3:],pos=sensor_object[:3])

        self.object_point_cloud = o3d.io.read_point_cloud("{}/{}/config/merged_cloud_{}.ply".format(dataset_path,object_name,object_name))
        self.object_point_cloud = self.object_point_cloud.voxel_down_sample(voxel_size=0.005)
        self.object_point_cloud.transform(self.FIX_sensor_T_object)
        
        #Init basefootprint_T_camera
        basefootprint_T_camera_data=yaml.load(open(os.path.join(Abs_Path,"files/basefootprint_camera.yaml")))['transformation']
        pos=np.array([basefootprint_T_camera_data['x'],basefootprint_T_camera_data['y'],basefootprint_T_camera_data['z']])
        ori=np.array([basefootprint_T_camera_data['qx'],basefootprint_T_camera_data['qy'],basefootprint_T_camera_data['qz'],basefootprint_T_camera_data['qw']])
        self.FIX_basefootprint_T_camera=matrix_from_quaternion(quaternion=ori,pos=pos)
        
        
        self.beautiful_print("Load {} object data".format(object_name))

    def update_color_image_cb(self,msg):
        """
        Update color_image;
        if Flag_save_data is ture, then will save base on color_image, sample other type of data when color_image is update
        """
        self._update_color_image=copy.deepcopy(self.bridge.imgmsg_to_cv2(msg,"bgr8"))

        if self.Flag_save_data:
            #Not allow other data update when saving the image
            self.save_lock=True
            
            #avoid save nothing
            if self._update_depth_image is None or self._update_pose_array is None or self._update_tip_pose_array is None or self._update_tip_data is None or self._update_tip_contact_array is None:
                print("one data is None,stop to record!!")
                self.save_lock=True
                return

            #data in big_save_list will save later in main function
            self.big_save_list.append([self._update_color_image,self._update_depth_image,self._update_pose_array,self._update_tip_data,self._update_tip_pose_array,self._update_tip_contact_array,self._update_joints_state])
            
            #Release self.save_lock
            self.save_lock=False
        
    def update_depth_image_cb(self,msg):
        depth_image=self.bridge.imgmsg_to_cv2(msg,"32FC1")
        if not self.save_lock:
            self._update_depth_image=copy.deepcopy(depth_image)

    def update_pose_array_cb(self,msg):
        """
        To update pose array; get object pose_array(4*4 shape) in camera_frame
        """
        position=msg.poses[0].position
        orientation=msg.poses[0].orientation

        record_data=np.array([position.x,position.y,position.z,orientation.x,orientation.y,orientation.z,orientation.w])

        pose_array_matrix=matrix_from_quaternion(record_data[3:],pos=record_data[:3])
        object_in_cam=np.matmul(self.FIX_camera_T_transmitter,pose_array_matrix)#4*4 matrix

        #Update Pose array
        if not self.save_lock:
            self._update_pose_array=copy.deepcopy(object_in_cam)

    def update_tip_pose_cb(self,msg):
        if len(msg.transforms)>2:#just record big tf_link update times
            #For record each tip to palm transform
            trans,rot=self.tf_linstener.lookupTransform(self.shadowhand_base_link_name ,self.five_tip_name[0] ,rospy.Time(0))
            pose1=np.array([trans[0],trans[1],trans[2],rot[0],rot[1],rot[2],rot[3]])
            trans,rot=self.tf_linstener.lookupTransform(self.shadowhand_base_link_name ,self.five_tip_name[1] ,rospy.Time(0))
            pose2=np.array([trans[0],trans[1],trans[2],rot[0],rot[1],rot[2],rot[3]])
            trans,rot=self.tf_linstener.lookupTransform(self.shadowhand_base_link_name ,self.five_tip_name[2] ,rospy.Time(0))
            pose3=np.array([trans[0],trans[1],trans[2],rot[0],rot[1],rot[2],rot[3]])
            trans,rot=self.tf_linstener.lookupTransform(self.shadowhand_base_link_name ,self.five_tip_name[3] ,rospy.Time(0))
            pose4=np.array([trans[0],trans[1],trans[2],rot[0],rot[1],rot[2],rot[3]])
            trans,rot=self.tf_linstener.lookupTransform(self.shadowhand_base_link_name ,self.five_tip_name[4] ,rospy.Time(0))
            pose5=np.array([trans[0],trans[1],trans[2],rot[0],rot[1],rot[2],rot[3]])

            #change matrix_basefootprint_T_hand to pose
            trans=self.matrix_basefootprint_T_hand[:3,3]
            rot=quaternion_from_matrix(self.matrix_basefootprint_T_hand)
            pose6=np.array([trans[0],trans[1],trans[2],rot[0],rot[1],rot[2],rot[3]])
            
            #Update basefootprintTcamera
            trans,rot=self.tf_linstener.lookupTransform(self.shadowhand_base_link_name ,self.five_tip_name[4] ,rospy.Time(0))
            pose5=np.array([trans[0],trans[1],trans[2],rot[0],rot[1],rot[2],rot[3]])

            #udpate array
            if not self.save_lock:
                self._update_tip_pose_array=copy.deepcopy(np.array([pose1,pose2,pose3,pose4,pose5,pose6]))#6,7 array
                
    def update_tip_data_cb(self,msg):
        """
        Update tactile data
        (Now ring finger all data are zeros)
        """
        once_data=[]
        for tactile in msg.tactiles:
            once_data.append(tactile.electrodes)

        mix_tip_data=np.array([once_data[4],once_data[0],once_data[1],once_data[2],once_data[3]])#Special tip order
        if not self.save_lock:
            self._update_tip_data=copy.deepcopy(mix_tip_data)#5,19 array,int64 type
        
    def update_tip_contact_cb(self,msg):
        contact_array=np.zeros(5)
        for contact in msg.contacts:
            contact_index=self.five_contact_name[contact.header.frame_id]
            contact_array[contact_index]=1

        if not self.save_lock:
            self._update_tip_contact_array=copy.deepcopy(contact_array)

    def update_joints_state_cb(self,msg):
        """
        Order is:
        [rh_FFJ1, rh_FFJ2, rh_FFJ3, rh_FFJ4, rh_LFJ1, rh_LFJ2, rh_LFJ3, rh_LFJ4, rh_LFJ5,
        rh_MFJ1, rh_MFJ2, rh_MFJ3, rh_MFJ4, rh_RFJ1, rh_RFJ2, rh_RFJ3, rh_RFJ4, rh_THJ1,
        rh_THJ2, rh_THJ3, rh_THJ4, rh_THJ5, rh_WRJ1, rh_WRJ2]
        """
        #Include two joints state, one is shadowhand joints,33 joints
        #another is PR2 joints, 24 joints
        if len(msg.position)==24:
            joints_states=np.array(list(msg.position))
            if not self.save_lock:
                self._update_joints_state=copy.deepcopy(joints_states)
        

############################Example for extract and show data################################
def example_check_data():
    #1: Init ExtractData class
    extractData=ExtractData(object_name="cracker",Flag_save_data=False,init_node=True)
    print("Begin to see images...")

    #2: load camera_matrix and point_cloud
    K = np.array([611.666, 0, 325.213, 
                0, 610.092,253.658, 
                0, 0, 1]).reshape(3, 3)

    object_point_cloud=extractData.object_point_cloud
        
    while not rospy.is_shutdown():

        if extractData._update_tip_pose_array is None:
            print("Wait for data input...")
            time.sleep(1)
            continue

        extractData.save_lock=True
        image=extractData._update_color_image

        # #To see depth data
        # depth_image=extractData._update_depth_image
        # color_map=extractData.get_colormap_from_depth(depth_image)
        # cv.imshow("color_map",color_map)

        # To see tactile data
        tip_data=extractData._update_tip_data
        print("tipdata is:")
        print(tip_data)

        
        #To see pose array
        pose_array=extractData._update_pose_array
        copy_point_cloud=copy.deepcopy(object_point_cloud)
        copy_point_cloud.transform(pose_array)
        uvs = project_points(np.asarray(copy_point_cloud.points), K)    
        for ii in range(len(uvs)):
            cv.circle(image,(uvs[ii,0],uvs[ii,1]),radius=1,color=(0,0,255),thickness=-1)


        # #To see tip pose array
        # tip_pose_array=extractData._update_tip_pose_array
        # print(tip_pose_array.shape)

        extractData.save_lock=False

        cv.imshow("images",image)
        if cv.waitKey(30)==ord('q'):
            break

def example_save_data():
    target_path="/home/media/WholeDataset/example"
    if not os.path.exists(target_path):
        temp=raw_input("target path not exist!!,create one? y for creation")
        if temp=='y':
            os.mkdir(target_path)

    #1: define ExtractData and check whether update data is new
    extractData=ExtractData(object_name="cracker",Flag_save_data=True,init_node=True)
    print("Please use 'rosbag play xxx.bag' to send data")
    print("Begin to wait incoming data...")
    num_last_save_time=0
    count=0
    while not rospy.is_shutdown():
        num_update_save=copy.deepcopy(len(extractData.big_save_list))
        if num_update_save==num_last_save_time:
            print("last data not update!!!")
            count=count+1
            if count>3:
                break

        else:
            print("Extract data Have save {} data".format(len(extractData.big_save_list)))
            count=0
            num_last_save_time=num_update_save
        rospy.sleep(1)

    #2: save all data
    #2.1: Load camera_matrix and pointcloud to  generate new shapes
    K = np.array([611.666, 0, 325.213, 
                0, 610.092,253.658, 
                0, 0, 1]).reshape(3, 3)

    object_point_cloud=extractData.object_point_cloud


    save_flag=True
    if save_flag:
        if not os.path.exists(target_path):
            print("!!!!target path not exist,please check!!!!")
            return

        #update save count number
        save_count_number=0
        if len(os.listdir(target_path))!=0:
            if len(os.listdir(target_path))//4!=0:
                print("Contains {} data,but can not group each 4 data".format(len(os.listdir(target_path))))            
            save_count_number=len(glob.glob(os.path.join(target_path,"color_*.png")))#Save base on color_image
            print("!!!target_path existing data,Begin number will be:{}!!!".format(save_count_number))

        print("Begin to save data...")
        begin_number=save_count_number
        for index,data in enumerate(extractData.big_save_list):
            print("Saveing {}/{} data...".format(index+begin_number,len(extractData.big_save_list)+begin_number))
            #2.2: save data in big list
            update_color_image,update_depth_image,update_pose_array,update_tip_data,update_tip_pose_array,update_contact_array,update_joints_state=data

            #2.3: plot_mask data
            plot_image=np.zeros(update_color_image.shape[:2],dtype=np.uint8)
            temp_plot_cloud=copy.deepcopy(object_point_cloud)
            temp_plot_cloud.transform(update_pose_array)
            uvs = project_points(np.asarray(temp_plot_cloud.points), K)
            for ii in range(len(uvs)):
                cv.circle(plot_image,(uvs[ii,0],uvs[ii,1]),radius=5,color=255,thickness=-1)
            #generate contours
            thresh = cv.threshold(plot_image, 30, 255, cv.THRESH_BINARY)[1]
            _, contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
            mask_image = np.zeros(plot_image.shape[:2],dtype = np.uint8)

            #plot mask
            if not len(contours)==0:
                cnt = max(contours, key=cv.contourArea)
                mask_image=cv.drawContours(mask_image, [cnt], -1, 255, -1)
            else:
                print("index {} has no mask!!!".format(save_count_number))

            #2.4: save all data
            #color_image
            cv.imwrite(os.path.join(target_path,"color_{}.png".format(save_count_number)),update_color_image)
            #depth_image
            depth_path=os.path.join(target_path,"depth_{}.png".format(save_count_number))
            with open(depth_path,'wb') as f:
                    writer=png.Writer(width=update_depth_image.shape[1],height=update_depth_image.shape[0],bitdepth=16)
                    zgray2list=update_depth_image.tolist()
                    writer.write(f,zgray2list)
            #meta data
            np.savez(os.path.join(target_path,"meta_{}".format(save_count_number)),
                    tip_data_array=update_tip_data,pose_array=update_pose_array,tip_pose_array=update_tip_pose_array,tip_contact_array=update_contact_array,joints_state=update_joints_state,camera_T_shadowhand=extractData.FIX_camera_T_hand)
            #mask_image
            cv.imwrite(os.path.join(target_path,"mask_{}.png".format(save_count_number)),mask_image)

            save_count_number=save_count_number+1

def see_images():
    target_path="/home/media/WholeDataset/{}".format("cleanser")
    begin_to_see=2800

    speed_up=10#To skip the index image

    while True:
        print("See image:{}".format(begin_to_see))
        image=cv.imread(os.path.join(target_path,"color_{}.png".format(begin_to_see)))
        
        mask=cv.imread(os.path.join(target_path,"mask_{}.png".format(begin_to_see)))

        cv.imshow("image",image)
        cv.imshow("mask",mask)

        temp_image=np.zeros(image.shape)
        temp_image[mask==255]=image[mask==255]
        temp_image=temp_image.astype(np.uint8)
        cv.imshow("sement image",temp_image)

        temp=cv.waitKey(0)
        if temp==ord('b'):
            begin_to_see=begin_to_see-2*speed_up
        if temp==ord('q'):
            break

        begin_to_see=begin_to_see+1*speed_up

def remove_data():
    delete_begin=5993
    dataset_path="/home/media/WholeDataset/{}".format("cleanser")
    all_index_file=glob.glob(os.path.join(dataset_path,"color_*.png"))
    print(len(all_index_file))


    # max_index=6459
    # while delete_begin<max_index:
    #     os.remove(os.path.join(dataset_path,"color_{}.png".format(delete_begin)))
    #     os.remove(os.path.join(dataset_path,"depth_{}.png".format(delete_begin)))
    #     os.remove(os.path.join(dataset_path,"mask_{}.png".format(delete_begin)))
    #     os.remove(os.path.join(dataset_path,"meta_{}.npz".format(delete_begin)))
    #     delete_begin=delete_begin+1

    

if __name__ == '__main__':
    # example_check_data()
    example_save_data()

