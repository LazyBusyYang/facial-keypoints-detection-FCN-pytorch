import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import scipy.misc
import os, sys
import pandas as pd
import imageio
from PIL import Image
try:
    from augmentations import FCN_train_aug, FCN_val_aug
except:
    from .augmentations import FCN_train_aug, FCN_val_aug
import cv2

points_classes = ['left_eye_center_x',
    'left_eye_center_y',
    'right_eye_center_x',
    'right_eye_center_y',
    'left_eye_inner_corner_x',
    'left_eye_inner_corner_y',
    'left_eye_outer_corner_x',
    'left_eye_outer_corner_y',
    'right_eye_inner_corner_x',
    'right_eye_inner_corner_y',
    'right_eye_outer_corner_x',
    'right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x',
    'left_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x',
    'left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x',
    'right_eyebrow_inner_end_y',
    'right_eyebrow_outer_end_x',
    'right_eyebrow_outer_end_y',
    'nose_tip_x',
    'nose_tip_y',
    'mouth_left_corner_x',
    'mouth_left_corner_y',
    'mouth_right_corner_x',
    'mouth_right_corner_y',
    'mouth_center_top_lip_x',
    'mouth_center_top_lip_y',
    'mouth_center_bottom_lip_x',
    'mouth_center_bottom_lip_y']

class FLandmarksDataset(Dataset):

    def __init__(self, csv_file, mode="train" , replace_nan = "drop", aug_transform=FCN_val_aug, classes=15,  sigma=1):
        super(FLandmarksDataset, self).__init__()

        self.height = 96
        self.width = 96


        self.classes = classes  
        self.sigma = sigma  # gaussian center heat map sigma

        self.key_pts_frame = pd.read_csv(csv_file)
        self.replace_nan = replace_nan
        
        self.not_null_frame = self.key_pts_frame.dropna()
        
        if replace_nan == "weighted":
        # if True:
            not_null_matrix = self.key_pts_frame.notnull().values
            for i in range(not_null_matrix.shape[0]):
                for j in range(1,not_null_matrix.shape[1], 1):
                    not_null_matrix[i, 0] = np.logical_and(not_null_matrix[i, 0], not_null_matrix[i, j])
            not_null_vector = not_null_matrix[:, 0]
            null_vector = np.logical_not(not_null_vector)
            true_count = 0
            for bool_val in null_vector:
                if bool_val:
                    true_count += 1
            # print("Found %d lines with nan"%true_count)
            nan_rows = self.key_pts_frame[null_vector]

        if mode=='train':
            if replace_nan == "drop":
                self.key_pts_frame =  self.not_null_frame[0:int(0.7*len(self.not_null_frame))]
            elif replace_nan == "weighted":                
                self.key_pts_frame =  pd.concat([self.not_null_frame[0:int(0.7*len(self.not_null_frame))],
                                                nan_rows])
                # print(len(self.key_pts_frame))      
            else:
                self.key_pts_frame =  self.key_pts_frame[0:int(0.7*len(self.key_pts_frame))]
        else: # val
            self.key_pts_frame =  self.not_null_frame[int(0.7*len(self.not_null_frame)): -1]
        self.aug_transform = aug_transform(size=self.height, mean=(0,0,0))
        self.norm = transforms.Compose([
                transforms.Normalize(mean=[0.485,],std=[0.229, ]),
                ])

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):

        label_size = self.width
        # print("label_size:", label_size)

        # load image as numpy(rgb image of cv2)
        img_str = self.key_pts_frame.iloc[idx, -1]
        pixels = img_str.split(" ")
        img_list = []
        for i in range(0, 96, 1):
            img_list.append([int(pixel) for pixel in pixels[i*96: (i+1)*96]])
        img_np = np.array(img_list).astype(np.uint8)
        img_np = img_np[..., np.newaxis]         
        img_rgb = np.concatenate((img_np, img_np.copy(), img_np.copy()), 2)
        # print("img_np.shape", img_np.shape)

        # load points(15,2)
        pts_csv = self.key_pts_frame.iloc[idx, 0 : -1]
        key_pts = pts_csv.values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        if key_pts.shape[0] != 15:
            print("wrong size:", key_pts.shape)

        key_pts_real = key_pts.copy().astype('float')
        if self.replace_nan == "weighted":
            for i in range(key_pts.shape[0]):
                for j in range(key_pts.shape[1]):
                    if np.isnan(key_pts[i, j]):
                        key_pts[i, j] = 96/2

        # point to rectangular
        rects = []
        labels = []
        for i in range(key_pts.shape[0]):
            x1 = (key_pts[i, 0]-0.5) / 96.0
            y1 = (key_pts[i, 1]-0.5) / 96.0
            x2 = (key_pts[i, 0]+0.5) / 96.0
            y2 = (key_pts[i, 1]+0.5) / 96.0
            rects.append((x1, y1, x2, y2))
        for i in range(key_pts.shape[0]):
            labels.append(0)
        
        # transforms of rect detect
        if self.aug_transform is not None:
            rects = np.array(rects)
            labels = np.array(labels)  
            img_rgb, rects, labels = self.aug_transform(img_rgb, rects, labels)    

        # rect to point
        img_np = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        aug_key_pts = np.zeros(key_pts.shape)
        for i in range(rects.shape[0]):
            aug_key_pts[i, 0] = (rects[i, 0] + rects[i, 2])/2.0 * 96.0
            aug_key_pts[i, 1] = (rects[i, 1] + rects[i, 3])/2.0 * 96.0
      
        height, width = (self.height, self.width)
        
        # ToTensor and normalize
        img_tensor = torch.from_numpy(img_np).permute(0, 1).unsqueeze(0).float()/255.0
        image = self.norm(img_tensor)
        ratio_x = self.width / float(width)
        ratio_y = self.height / float(height)    # ratio = 1

        # generate label map
        lbl = self.genLabelMap(aug_key_pts, label_size=label_size, classes=self.classes, ratio_x=ratio_x, ratio_y=ratio_y)
        label_maps = torch.from_numpy(lbl)

        # weighted mask
        if self.replace_nan == "weighted":
            for i in range(key_pts_real.shape[0]):
                for j in range(key_pts_real.shape[1]):
                    if np.isnan(key_pts_real[i, j]):
                        key_pts_real[i, j] = 0.0
                    else:
                        key_pts_real[i, j] = 1.0
            return image.float(), label_maps.float(), torch.from_numpy(aug_key_pts).float() , torch.from_numpy(key_pts_real).float()
        else:
            return image, label_maps.float(), torch.from_numpy(aug_key_pts) 

    def genCenterMap(self, x, y, sigma, size_w, size_h):
        """
        generate Gaussian heat map
        :param x: center point
        :param y: center point
        :param sigma:
        :param size_w: image width
        :param size_h: image height
        :return:            numpy           w * h
        """
        gridy, gridx = np.mgrid[0:size_h, 0:size_w]
        D2 = (gridx - x) ** 2 + (gridy - y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)  # numpy 2d

    def genLabelMap(self, center_points, label_size, classes, ratio_x, ratio_y):
        """
        generate label heat map
        :param label:               list            15 * 2
        :param label_size:          int             96
        :param classes:              int            15
        :param ratio_x:             float           1
        :param ratio_y:             float           1
        :return:  heatmap           numpy           joints * boxsize/stride * boxsize/stride
        """
        # initialize
        label_maps = np.zeros((classes, label_size, label_size))
        background = np.zeros((label_size, label_size))

        # each joint
        for i in range(len(center_points)):
            lbl = center_points[i]                      # [x, y]
            x = lbl[0] * ratio_x          
            y = lbl[1] * ratio_y
            heatmap = self.genCenterMap(y, x, sigma=self.sigma, size_w=label_size, size_h=label_size)  # numpy
            background += heatmap               # numpy
            label_maps[i, :, :] = np.transpose(heatmap)  

        return label_maps  # numpy           

class FacialKeypointsTestDataset(Dataset):
    """Face Landmarks dataset."""

    # def __init__(self, csv_file, transform=transforms.Compose([Normalize(), ToTensor()])):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file of testset.
        """
        self.key_pts_frame = pd.read_csv(csv_file)   
        self.norm = transforms.Compose([
                transforms.Normalize(mean=[0.485,],std=[0.229, ]),
                ])
        
    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        img_str = self.key_pts_frame.iloc[idx, -1]
        pixels = img_str.split(" ")
        img_list = []
        for i in range(0, 96, 1):
            img_list.append([int(pixel) for pixel in pixels[i*96: (i+1)*96]])
        img_np = np.array(img_list).astype(np.uint8)
        
        # print("img_np.shape", img_np.shape)

        # load points
        pts_csv = self.key_pts_frame.iloc[idx, 0 : -1]
        img_id = pts_csv.values
        img_id = img_id.astype('int32')
        
        img_tensor = torch.from_numpy(img_np).permute(0, 1).unsqueeze(0).float()/255.0
        image = self.norm(img_tensor)

        sample = {'image': image,
                'img_id': torch.from_numpy(img_id)}
        return sample

def write_result(result_list, lookup_path, write_path):    
    # # write test result to file
    # print(len(points_classes))
    points_dict = {}
    for i in range(30):
        points_dict[points_classes[i]] = i
    idmap = pd.read_csv(lookup_path)
    imageid = list(idmap['ImageId'])
    feature = list(idmap['FeatureName'])
    # print(len(imageid))
    # print(len(feature))
    save = []
    l = len(imageid)
    for i in range(l):
        save.append(result_list[(imageid[i] - 1) * 30 + points_dict[feature[i]]])

    f = open(write_path, "w+")
    f.write("RowId,Location\n")
    for i in range(1, l + 1):
        f.write("{},{}".format(i, save[i-1]))
        if i != l:
            f.write("\n")
    f.close()
    return save

# test case
if __name__ == "__main__":
    data_path = "F:/temp/facial-keypoints-detection/training.csv"
    test_dataset = FLandmarksDataset(csv_file=data_path, mode="train", replace_nan = "weighted", aug_transform=FCN_train_aug, classes=15, sigma=3)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,num_workers=0)
    iter_loader = iter(test_dataloader)
    # progress = tqdm(test_dataloader, ascii = True)

    de_norm = transforms.Compose([
                transforms.Normalize(mean=[0,],std=[1.0/0.229, ]),
                transforms.Normalize(mean=[-0.485, ],std=[1.0, ]),
                ])
    img, label, _ , _ = test_dataset[20]
    print('dataset info ...')
    print(img.shape)         # 3D Tensor 1 * 96 * 96
    print(label.shape)       # 3D Tensor 15 * 96 * 96

    # ***************** draw label map *****************
    print('draw label map ...')
    lab = np.asarray(label)
    out_labels = np.zeros((96, 96))
    for i in range(lab.shape[0]):
        out_labels += lab[i, :, :]
        # imageio.imwrite('img/facialLandmark_label_%02d.jpg'%i, lab[i, :, :])
    imageio.imwrite('img/facialLandmark_label.jpg', out_labels)

    # ***************** draw image *****************
    print('draw image ')
    img = transforms.ToPILImage()(de_norm(img))
    img.save('img/facialLandmark_img.jpg')

    heatmap = np.asarray(label[0, :, :])






