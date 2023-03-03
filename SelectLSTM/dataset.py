import os
import os.path
import glob
import random
import cv2 as cv
import numpy as np
import torch.utils.data as data


Abs_Path=os.path.dirname(os.path.abspath(__file__))

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

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


def generate_shuffle_order():
    whole_dataset_path="/home/media/WholeDataset"
    object_name_list=['cleanser', 'cracker', 'cup', 'mustard', 'pear', 'spatula', 'sugar']
    txt_folders_path=os.path.join(Abs_Path,"shuffile_files")
    for object_name in object_name_list:
        dataset_path=os.path.join(whole_dataset_path,object_name)
        all_index_files=glob.glob(os.path.join(dataset_path,"color*.png"))
        index_list=[i for i in range(len(all_index_files))]
        random.shuffle(index_list)
        
        # target_file=open(os.path.join(txt_folders_path,"{}.txt".format(object_name)),'w')
        # for index in index_list:
            # target_file.write("{}\n".format(index))


class TactileDataset(data.Dataset):
    def __init__(self,dataset_path,split_method,split,split_rate=None,split_index_list=None):
        """
        dataset_path: the folder save data,including color_n.png;depth_n.png;mask_n.png;meta_n.npz
        split: train or test
        split_index_list: special index from data; [0,100]
        """
        self.dataset_path=dataset_path
        object_name=self.dataset_path.split('/')[-1]

        #1: random split by serial method
        if split_method=="shuffle":
            index_list=list(map(int,open(os.path.join(Abs_Path,"shuffile_files/{}.txt".format(object_name)),'r').readlines()))
            if split=='train':
                self.index_list=index_list[:int(len(index_list)*split_rate)]
            elif split=='test':
                self.index_list=index_list[int(len(index_list)*split_rate):]
            
        elif split_method=="serial":
            #load index data
            record_txt_data=open(os.path.join(dataset_path,"../record_files/{}.txt".format(object_name))).read().splitlines()
            each_serial_index_list=[0]
            for data in record_txt_data:
                end_num=int(data.split(":")[-1])
                each_serial_index_list.append(end_num)

            #generate_index_list
            self.index_list=[]
            for index_number,end_number in enumerate(each_serial_index_list):
                if index_number==0:
                    continue
                begin_number=each_serial_index_list[index_number-1]
                gap=end_number-begin_number
                if split=='train':
                    self.index_list.append(np.arange(begin_number+1,int(begin_number+gap*split_rate)))
                elif split=="test":
                    self.index_list.append(np.arange(int(begin_number+gap*split_rate),end_number))

            self.index_list=np.concatenate(self.index_list)
            
        elif split_method=="index":
            assert len(split_index_list)==2
            all_index_data=glob.glob(dataset_path+"/color*")
            if split_index_list[1]>len(all_index_data):
                self.index_list=np.arange(split_index_list[0],len(all_index_data),step=1)
                print("All data only have {},but input split_index_list is:{}".format(len(all_index_data),split_index_list))
            else:
                self.index_list=np.arange(split_index_list[0],split_index_list[1],step=1)
        else:
            print("!!!!!!Please input split method of 'shuffle,serial,index'!!!!!!")
            
        print("Load {} dataset, contain {} data".format(split,len(self.index_list)))

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        #1: Load index file
        index=self.index_list[index]
                    
        #2: Load target pose
        meta_data=np.load(os.path.join(self.dataset_path,"meta_{}.npz".format(index)))
        camera_T_object=meta_data['ft_pose_array']
        camera_T_hand=meta_data['ft_camera_T_shadowhand']
        pose_matrix=np.matmul(np.linalg.inv(camera_T_hand),camera_T_object)
        
        xyz=pose_matrix[:3,3]
        q=quaternion_from_matrix(pose_matrix)
        target_pose=np.concatenate([xyz,q])
        target_pose=target_pose.astype(np.float32)
        
        #3: Load first frame tip data
        tip_data_array=meta_data['tip_data_array'].astype(np.float32)#5,19
        tip_pose_array=meta_data['tip_pose_array'].astype(np.float32)#6,7


        return tip_pose_array,tip_data_array,target_pose

class VisionDataset(data.Dataset):
    def __init__(self,dataset_path,split_method,split=None,split_rate=None,split_index_list=None):
        """
        dataset_path: the folder save data,including color_n.png;depth_n.png;mask_n.png;meta_n.npz
        split: train or test
        split_index_list: special index from data; [0,100]
        """
        self.dataset_path=dataset_path
        object_name=self.dataset_path.split('/')[-1]

        self.camera_K = np.array([611.666, 0, 325.213, 
                0, 610.092,253.658, 
                0, 0, 1]).reshape(3, 3)


        #1: random split by serial method
        if split_method=="shuffle":
            index_list=list(map(int,open(os.path.join(Abs_Path,"shuffile_files/{}.txt".format(object_name)),'r').readlines()))
            if split=='train':
                self.index_list=index_list[:int(len(index_list)*split_rate)]
            elif split=='test':
                self.index_list=index_list[int(len(index_list)*split_rate):]
            
        elif split_method=="serial":
            #load index data
            record_txt_data=open(os.path.join(dataset_path,"../record_files/{}.txt".format(object_name))).read().splitlines()
            each_serial_index_list=[0]
            for data in record_txt_data:
                end_num=int(data.split(":")[-1])
                each_serial_index_list.append(end_num)

            #generate_index_list
            self.index_list=[]
            for index_number,end_number in enumerate(each_serial_index_list):
                if index_number==0:
                    continue
                begin_number=each_serial_index_list[index_number-1]
                gap=end_number-begin_number
                if split=='train':
                    self.index_list.append(np.arange(begin_number+1,int(begin_number+gap*split_rate)))
                elif split=="test":
                    self.index_list.append(np.arange(int(begin_number+gap*split_rate),end_number))

            self.index_list=np.concatenate(self.index_list)
            
        elif split_method=="index":
            assert len(split_index_list)==2
            all_index_data=glob.glob(dataset_path+"/color*")
            if split_index_list[1]>len(all_index_data):
                self.index_list=np.arange(split_index_list[0],len(all_index_data),step=1)
                print("All data only have {},but input split_index_list is:{}".format(len(all_index_data),split_index_list))
            else:
                self.index_list=np.arange(split_index_list[0],split_index_list[1],step=1)
        else:
            print("!!!!!!Please input split method of 'shuffle,serial,index'!!!!!!")
            
            
        print("Load {} dataset, contain {} data".format(split,len(self.index_list)))

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        #1: Load index file
        index=self.index_list[index]
                    
        #2: Load target pose
        meta_data=np.load(os.path.join(self.dataset_path,"meta_{}.npz".format(index)))
        camera_T_object=meta_data['ft_pose_array']
        camera_T_hand=meta_data['ft_camera_T_shadowhand']
        pose_matrix=np.matmul(np.linalg.inv(camera_T_hand),camera_T_object)
        
        xyz=pose_matrix[:3,3]
        q=quaternion_from_matrix(pose_matrix)
        target_pose=np.concatenate([xyz,q])
        target_pose=target_pose.astype(np.float32)
        
        #3: Load vision data
        rgb_image=cv.imread(os.path.join(self.dataset_path,"color_{}.png".format(index)))
        depth_image=cv.imread(os.path.join(self.dataset_path,"depth_{}.png".format(index)),cv.IMREAD_UNCHANGED)
        bbox=compute_bbox(camera_T_object,self.camera_K,scale=(1000, 1000, 1000))
        rgb,depth=crop_bbox(rgb_image,depth_image,bbox,output_size=(128,128))
        rgb=rgb/255
        depth=depth/1000
        depth=depth[:,:,np.newaxis]

        #concate as vision data
        rgbd_data=np.concatenate([rgb,depth],axis=2)
        rgbd_data=np.transpose(rgbd_data,[2,0,1])
        
        return rgbd_data,target_pose

class MergeDataset(data.Dataset):
    def __init__(self, dataset_path,split_method,split,split_rate=None,split_index_list=None,num_pt=1000):
        #1: Load random index list
        self.dataset_path=dataset_path
        object_name=dataset_path.split('/')[-1]
        self.camera_K = np.array([611.666, 0, 325.213, 
                0, 610.092,253.658, 
                0, 0, 1]).reshape(3, 3)

        if split_method=="shuffle":
            index_list=list(map(int,open(os.path.join(Abs_Path,"shuffile_files/{}.txt".format(object_name)),'r').readlines()))
            if split=='train':
                self.index_list=index_list[:int(len(index_list)*split_rate)]
            elif split=='test':
                self.index_list=index_list[int(len(index_list)*split_rate):]
            
        elif split_method=="serial":
            #load index data
            record_txt_data=open(os.path.join(dataset_path,"../record_files/{}.txt".format(object_name))).read().splitlines()
            each_serial_index_list=[0]
            for data in record_txt_data:
                end_num=int(data.split(":")[-1])
                each_serial_index_list.append(end_num)

            #generate_index_list
            self.index_list=[]
            for index_number,end_number in enumerate(each_serial_index_list):
                if index_number==0:
                    continue
                begin_number=each_serial_index_list[index_number-1]
                gap=end_number-begin_number
                if split=='train':
                    self.index_list.append(np.arange(begin_number+1,int(begin_number+gap*split_rate)))
                elif split=="test":
                    self.index_list.append(np.arange(int(begin_number+gap*split_rate),end_number))

            self.index_list=np.concatenate(self.index_list)
            
        elif split_method=="index":
            assert len(split_index_list)==2
            all_index_data=glob.glob(dataset_path+"/color*")
            if split_index_list[1]>len(all_index_data):
                self.index_list=np.arange(split_index_list[0],len(all_index_data),step=1)
                print("All data only have {},but input split_index_list is:{}".format(len(all_index_data),split_index_list))
            else:
                self.index_list=np.arange(split_index_list[0],split_index_list[1],step=1)
        else:
            print("!!!!!!Please input split method of 'shuffle,serial,index'!!!!!!")
                    
        print("MergeDataset Load {} dataset, contain data:{}".format(split,len(self.index_list)))

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        #1: Load index file
        index=self.index_list[index]
                    
        #2: Load target pose
        meta_data=np.load(os.path.join(self.dataset_path,"meta_{}.npz".format(index)))
        
        #From hand
        camera_T_object=meta_data['ft_pose_array']
        camera_T_hand=meta_data['ft_camera_T_shadowhand']
        pose_matrix=np.matmul(np.linalg.inv(camera_T_hand),camera_T_object)
        xyz=pose_matrix[:3,3]
        q=quaternion_from_matrix(pose_matrix)
        target_pose=np.concatenate([xyz,q])
        target_pose=target_pose.astype(np.float32)
        
        #3: Load vision data
        rgb_image=cv.imread(os.path.join(self.dataset_path,"color_{}.png".format(index)))
        depth_image=cv.imread(os.path.join(self.dataset_path,"depth_{}.png".format(index)),cv.IMREAD_UNCHANGED)
        bbox=compute_bbox(camera_T_object,self.camera_K,scale=(1000, 1000, 1000))
        rgb,depth=crop_bbox(rgb_image,depth_image,bbox,output_size=(128,128))
        rgb=rgb/255
        depth=depth/1000
        depth=depth[:,:,np.newaxis]

        #concate as vision data
        rgbd_data=np.concatenate([rgb,depth],axis=2)
        rgbd_data=np.transpose(rgbd_data,[2,0,1]).astype(np.float32)
        
        #4: Load tactile data
        tip_data_array=meta_data['tip_data_array'].astype(np.float32)#5,19
        tip_pose_array=meta_data['tip_pose_array'].astype(np.float32)#6,7
        return tip_pose_array,tip_data_array,rgbd_data,target_pose

class SelectDataset(data.Dataset):
    def __init__(self,dataset_path,split_method,split='train',split_rate=None,split_index_list=None,train_length=20):
        self.train_length=train_length
        object_name=dataset_path.split('/')[-1]
        
        npy_path=os.path.join(Abs_Path,"../selectlstm_data/select_data_{}.npy".format(object_name))
        save_all_data_dict=np.load(npy_path,allow_pickle=True).item()

        self.all_tactile_pose=save_all_data_dict['tactile_pose']
        self.all_tactile_feature=save_all_data_dict['tactile_feature']
        self.all_image_pose=save_all_data_dict['image_pose']
        self.all_image_feature=save_all_data_dict['image_feature']
        self.all_merge_pose=save_all_data_dict['merge_pose']
        self.all_merge_feature=save_all_data_dict['merge_feature']
        self.all_label=save_all_data_dict['all_label']


        self.all_data_count=self.all_label.shape[0]

        if split_method=="shuffle":
            index_list=list(map(int,open(os.path.join(Abs_Path,"shuffile_files/{}.txt".format(object_name)),'r').readlines()))
            if split=='train':
                self.index_list=index_list[:int(len(index_list)*split_rate)]
            elif split=='test':
                self.index_list=index_list[int(len(index_list)*split_rate):]
            
        elif split_method=="serial":
            #load index data
            record_txt_data=open(os.path.join(dataset_path,"../record_files/{}.txt".format(object_name))).read().splitlines()
            each_serial_index_list=[0]
            for data in record_txt_data:
                end_num=int(data.split(":")[-1])
                each_serial_index_list.append(end_num)

            #generate_index_list
            self.index_list=[]
            for index_number,end_number in enumerate(each_serial_index_list):
                if index_number==0:
                    continue
                begin_number=each_serial_index_list[index_number-1]
                gap=end_number-begin_number
                if split=='train':
                    self.index_list.append(np.arange(begin_number+1,int(begin_number+gap*split_rate)))
                elif split=="test":
                    self.index_list.append(np.arange(int(begin_number+gap*split_rate),end_number))

            self.index_list=np.concatenate(self.index_list)
            
        elif split_method=="index":
            assert len(split_index_list)==2
            all_index_data=glob.glob(dataset_path+"/color*")
            if split_index_list[1]>len(all_index_data):
                self.index_list=np.arange(split_index_list[0],len(all_index_data),step=1)
                print("All data only have {},but input split_index_list is:{}".format(len(all_index_data),split_index_list))
            else:
                self.index_list=np.arange(split_index_list[0],split_index_list[1],step=1)
        else:
            print("!!!!!!Please input split method of 'shuffle,serial,index'!!!!!!")
            
                
        print("Load {} data in {} set".format(len(self.index_list),split))

        ###########To see data distribution###########
        count_0=self.all_label[self.all_label==0]
        count_1=self.all_label[self.all_label==1]
        count_2=self.all_label[self.all_label==2]
        print("Including the kind of data is:")
        print(count_0.shape)
        print(count_1.shape)
        print(count_2.shape)
        
    def __getitem__(self, index):
        index=int(self.index_list[index])
        train_length=self.train_length
        
        if index+train_length>self.all_data_count:
            merge_feature=self.all_merge_feature[index: ,:].astype(np.float32)
            tactile_feature=self.all_tactile_feature[index:,:].astype(np.float32)
            image_feature=self.all_image_feature[index:,:].astype(np.float32)

            merge_pose=self.all_merge_pose[index: ,:].astype(np.float32)
            tactile_pose=self.all_tactile_pose[index: ,:].astype(np.float32)
            densefusion_pose=self.all_image_pose[index: ,:].astype(np.float32)
            return_label=self.all_label[index: ]

        else:
            merge_feature=self.all_merge_feature[index:index+train_length ,:].astype(np.float32)
            tactile_feature=self.all_tactile_feature[index:index+train_length,:].astype(np.float32)
            image_feature=self.all_image_feature[index:index+train_length,:].astype(np.float32)
            
            merge_pose=self.all_merge_pose[index:index+train_length,:].astype(np.float32)
            tactile_pose=self.all_tactile_pose[index:index+train_length,:].astype(np.float32)
            densefusion_pose=self.all_image_pose[index:index+train_length,:].astype(np.float32)
            return_label=self.all_label[index:index+train_length]

        return np.concatenate([merge_pose,tactile_pose,densefusion_pose],axis=1),\
                np.concatenate([merge_feature,tactile_feature,image_feature],axis=1),\
                return_label

    def __len__(self) -> int:
         return len(self.index_list)


if __name__ == '__main__':
    generate_shuffle_order()