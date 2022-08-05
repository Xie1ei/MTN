
from dataclasses import dataclass
import enum
from typing import Dict, List
import json

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import transforms

from PIL import Image
from scipy.spatial.transform import Rotation

class ObjectDataset(Dataset):
    def __init__(self, txt_path, img_trans=None,
                conf = dict(
                    single_cam = True,
                    cam = ['front_middle_camera', 'rear_middle_camera','rr_wide_camera',
                             'rf_wide_camera', 'lf_wide_camera', 'lr_wide_camera' ],
                    tasks = ['Object'], # Object , 2dlane , 3dlane,
                    object_map = dict(
                        car = 1, bus = 2, truck = 3, motorcycle = 4, bicycle = 5  # 其他类别后续增加
                    ),
                    BEV_size = [8,80, 6, -6] # [a, b, c, -d] 代表 ego坐标系下，bev检测范围为 前向 a-b米， 横向左c米， 右d米
                )
    ) -> None:
        super().__init__()
        self.conf = conf
        if self.conf['single_cam']:
            self.conf['cam'] = ['front_middle_camera']

        self.f_list = [line.strip() for line in open(txt_path)]
        if img_trans:
            self.img_trans = img_trans
        else:
            self.img_trans = transforms.ToTensor()

       
    
    def __len__(self):
        return len(self.f_list)
    
    def __getitem__(self, index):
        """
        Haomo ego
                                up z    x 
                                   ^   ^
                                   |  /
                                   | /
                     left y <------ 0

        """
        res = {}
        # info
        with open(self.f_list[index],'r') as f:
            info = json.load(f)
            
        # imgs
        imgs = self._get_image(info['images'])
        imgs = [imgs[x] for x in self.conf['cam']]
        if len(imgs) == 1:
            if  self.img_trans:
                res['img'] = self.img_trans(imgs[0])
            else:
                res['img'] = imgs[0]
        else:   
            if  self.img_trans:
                imgs = [self.img_trans(x) for x in imgs]
            if not isinstance(imgs[0], torch.Tensor):
                imgs = [torch.from_numpy(np.array(x)) for x in imgs]
            res['img'] = torch.stack(imgs)
        
        # TODO tensor matrix
        mat = self._get_mats('/' + info['calibration_file'])
        res['cam2ego'] = [torch.from_numpy(mat['cam'][x]['cam2ego']) for x in self.conf['cam']]
        res['cam_intrins'] = [torch.from_numpy(mat['cam'][x]['cam_intrins']) for x in self.conf['cam']]
        res['dist_coef'] = [torch.from_numpy(mat['cam'][x]['dist_coef']) for x in self.conf['cam']]
        res['lidar'] = torch.from_numpy(mat['lidar'])
        if self.conf['single_cam']:
            res['cam2ego'] = res['cam2ego'][0]
            res['cam_intrins'] = res['cam_intrins'][0]
            res['dist_coef'] = res['dist_coef'][0]
        else:
            res['cam2ego'] = torch.stack(res['cam2ego'], 0)
            res['cam_intrins'] = torch.stack(res['cam_intrins'], 0)
            res['dist_coef'] = torch.stack(res['dist_coeffs'],0)
        # gt for each tasks
        for task in self.conf['tasks']:
            if task == 'Object':
                res['object'] = self._get_object(info['objects'], mat['lidar'])  
            # TODO lane task             
        
        return res




    def _get_object(self,objects, lidar_matrix):
        """
        return: nums_targets x 6. [batch_idx class x y h w] ego
               ^ X
               |   
               |
       Y <------
        """
        res = []
        for object in objects:
            
            if object['className'] in self.conf['object_map'].keys():
                tmp = np.zeros(6)
                tmp[1] = self.conf['object_map'][object['className']]
                position = np.array([x for x in object['3D_attributes']['position'].values()]).reshape(1,3)
                position = lidar_matrix[None,:3,:3] @ position[:,:,None] + lidar_matrix[None, :3, 3:]
                tmp[2:4] = position[0,:2,0] 
                
                tmp[4:] = np.array([object['3D_attributes']['dimension'][x] for x in ['length', 'width']])
                # TODO ego box to xywh
                if self.conf['BEV_size'][0]<tmp[2]<self.conf['BEV_size'][1] and self.conf['BEV_size'][3]<tmp[3]<self.conf['BEV_size'][2]:
                    new_tmp = tmp.copy()
                    w , h = abs(self.conf['BEV_size'][2]-self.conf['BEV_size'][3]), abs(self.conf['BEV_size'][0]-self.conf['BEV_size'][1])
                    new_tmp[2] = (w/2-tmp[3])/w
                    new_tmp[3] = (self.conf['BEV_size'][1]-tmp[2])/h
                    new_tmp[4] = (tmp[5])/w
                    new_tmp[5] = (tmp[4])/h
                    new_tmp = torch.from_numpy(new_tmp)
                    res.append(new_tmp)
        return res

    def _get_image(self,images):
        imgs  = dict()
        for image in images:
            if image['image_orientation'] in self.conf['cam']:
                img = Image.open('/' + image['image'])
                imgs[image['image_orientation']] = img
        return imgs

    def _get_mats(self, calibration_file:str):
        """
        return
            mat = {
                cam = {
                    cam_name = {
                        cam2ego: 4x4
                        cam_intrins: 3x3
                        dist_coef: 5x1
                    }
                }
                lidar = {lidar2ego: 4x4 }
            }
        
        """

        mat = dict()
        cam = {}
        f = open('/'+ calibration_file)
        cal = json.load(f)
        # cam
        # print(cal['sensor_config'].keys())
        # exit()
        for camera in cal['sensor_config']['cam_param']:
            if camera.get('name', 'xxx') in self.conf['cam']:
                cam_matrix = np.eye(3,dtype=np.float64)
                cam_matrix[0, 0] = camera['fx']
                cam_matrix[1, 1] = camera['fy']
                cam_matrix[0, 2] = camera['cx']
                cam_matrix[1, 2] = camera['cy']

                quaternion = [camera['pose']['attitude'][k] for k in ['x', 'y', 'z', 'w']]
                rotation = Rotation.from_quat(quaternion).as_matrix()
                translation = np.array(
                        [camera['pose']['translation'][k] for k in ['x', 'y', 'z']],
                                        dtype=np.float64).reshape(3, 1)
                extrinsics = np.eye(4,dtype=np.float64)
                extrinsics[:3,:3] = rotation
                extrinsics[:3, -1] = translation.reshape(3,)

                dist_coeffs = np.zeros((5,1), dtype=np.float64)
                dist_coeffs[0, 0] = camera['distortion'][0]
                dist_coeffs[1, 0] = camera['distortion'][1]
                dist_coeffs[4, 0] = camera['distortion'][2]
                dist_coeffs[2, 0] = camera['distortion'][3]

                cam[camera['name']] = { 'cam2ego' : extrinsics,
                                        'cam_intrins' : cam_matrix,
                                        'dist_coef' : dist_coeffs
                }
        # 
        
        mat['cam'] = cam
        # TODO Lidar matrix
        lidar_param = cal["sensor_config"]["lidar_param"]
        lidar_middle = None
        for lidar in lidar_param:
            if lidar['name'] == 'MIDDLE_LIDAR':
                quaternion = [lidar['pose']['attitude'][k] for k in ['x', 'y', 'z', 'w']]
                rotation = Rotation.from_quat(quaternion).as_matrix()

                translation = np.array(
                            [lidar['pose']['translation'][k] for k in ['x', 'y', 'z']], 
                                            dtype=np.float64).reshape(3, 1)
                extrinsics = np.eye(4,dtype=np.float64)
                extrinsics[:3,:3] = rotation
                extrinsics[:3, -1] = translation.reshape(3,)
                lidar_middle = extrinsics
                break
        mat['lidar'] = lidar_middle

        f.close()
        return mat

    @staticmethod
    def collate_fn(batch):
        """

        """
        res = {}
        res['img'] = torch.stack([x['img'] for x in batch], 0)
        res['cam2ego'] = torch.stack([x['cam2ego'] for x in batch],0)
        res['cam_intrins'] = torch.stack([x['cam_intrins'] for x in batch], 0)
        res['dist_coef'] = torch.stack([x['dist_coef'] for x in batch], 0)
        res['lidar'] = torch.stack([x['lidar'] for x in batch], 0)

        objects = []
        for i ,x in enumerate(batch):
            for ob in x['object']:
                ob[0] = i
                objects.append(ob)
        
        res['object'] = torch.stack(objects,0)
        return res

        