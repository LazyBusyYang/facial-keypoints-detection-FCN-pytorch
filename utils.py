import Config
from itertools import product as product
from math import sqrt as sqrt
from torchvision import transforms
import torch
import os
import shutil
import pandas as pd
import numpy as np
import pickle
import cv2
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class myAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
    def average(self):        
        if self.count != 0:
            self.avg = float(self.sum) / self.count
            return self.avg
        else:
            self.avg = 0
            return 0

def save_checkpoint(state, is_best, checkpoint, model_best):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best)

def record_info(info,filename,mode):

    if mode =='train':
        result = (
              'Data Time {data_time} \n'            
              'Batch Time {batch_time} \n'
              'Perf {Perf} \n'
              'Loss {Loss} \n'.format(data_time=info['Data Time'], batch_time=info['Batch Time'],
               Perf=info['Perf'], Loss=info['Loss']))    
        print (result)

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Data Time','Perf','Loss','lr']
        
    if mode =='test':
        result = (
              'Data Time {data_time} \n'            
              'Batch Time {batch_time} \n'
              'Perf {Perf} \n'
              'Loss {Loss} \n'.format(data_time=info['Data Time'], batch_time=info['Batch Time'],
               Perf=info['Perf'], Loss=info['Loss']))    
        print (result)
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Perf','Loss']
    
    if not os.path.isfile(filename):
        df.to_csv(filename,index=False,columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names) 

def getPointByMap(maps_tensor, topk=35):
    """
        input: maps_tensor.size(): (batch_size, channel, h, w)
        function: find the topk pixels in one channel and caculate the average location
        output: locations_tensor.size(): (batch_size, channel, 2)
    """
    np_pts = np.zeros((maps_tensor.size(0), maps_tensor.size(1), 2))
    for b in range(maps_tensor.size(0)):
        for ch in range(maps_tensor.size(1)):
            map_tensor = maps_tensor[b, ch, :, :]
            values, indices = torch.topk(map_tensor.view(-1), topk, dim=-1, largest=True, sorted=True, out=None)
            cur = topk-1
            while(values[cur] <= 0):
                cur -= 1
                if cur == 0:
                    break
            x = 0.0
            y = 0.0
            for i in range(cur+1):
                idx = int(indices[i])
                x += int(idx % maps_tensor.size(2))
                y += int(idx / maps_tensor.size(2))
            np_pts[b, ch, 0] = x / float(cur+1)
            np_pts[b, ch, 1] = y / float(cur+1)
    return torch.from_numpy(np_pts)

def save_predict(tensor_map, tensor_img,tensor_name, len_limit = 20):
    """
        input: tensor.size(): (batch_size, channel, ...)
        function: save the Gaussian heat map with origin image to PNG file
    """
    de_norm = transforms.Compose([
                transforms.Normalize(mean=[0,],std=[1.0/0.229, ]),
                transforms.Normalize(mean=[-0.485, ],std=[1.0, ]),
                ])
    for b in range(0, tensor_map.size(0), 1):
        if b >= len_limit:
            break
        img_gray = de_norm(tensor_img[b, :, :, :])
        img_rgb = torch.cat((img_gray, img_gray.clone(), img_gray.clone()), 0)
        img_rgb = 255 * img_rgb
        img_rgb = tensor_to_np(img_rgb)
        for point_idx in range(0, tensor_map.shape[1], 1):
            heat_map = tensor_to_heatmap(tensor_map[b, point_idx : point_idx+1, :, :], 96)
            output_map = add_weightedmap(img_rgb, heat_map)
            cv2.imwrite(os.path.join("./img", "batch%02d_point%02d_%s.png"%(b, point_idx, tensor_name)), output_map)  
    return

def save_image(tensor_img, tensor_name, len_limit = 20):
    """
        input: tensor.size(): (batch_size, channel, ...)
        function: save the origin image to PNG file
    """
    de_norm = transforms.Compose([
                transforms.Normalize(mean=[0,],std=[1.0/0.229, ]),
                transforms.Normalize(mean=[-0.485, ],std=[1.0, ]),
                ])
    for b in range(0, tensor_img.size(0), 1):
        if b >= len_limit:
            break
        img_gray = de_norm(tensor_img[b, :, :, :])
        img_gray = 255 * img_gray
        img_gray = tensor_to_np(img_gray)
        cv2.imwrite(os.path.join("./img", "batch%02d_%s.png"%(b, tensor_name)), img_gray)  
    return

def save_feature_map(feature_map, feature_name, len_limit = 20):
    """
        input: tensor.size(): (batch_size, channel, h, w)
        function: save the feature heat map with origin image to PNG file
    """
    feature_map = torch.abs(feature_map)
    feature_map = torch.sum(feature_map, dim=1, keepdim=True)
    for b in range(0, feature_map.size(0), 1):
        if b >= len_limit:
            break
        feature_map_onepic = feature_map[b, :, :, :]
        heat_map_onepic = tensor_to_heatmap(feature_map_onepic, size=96)
        cv2.imwrite(os.path.join("./img", "batch%02d_%s.png"%(b, feature_name)), heat_map_onepic)  
    return

    
def tensor_to_heatmap(tensor, size = 384):
    map_tensor = tensor.clone().detach()
    max_tmp = torch.max(map_tensor)
    # print("Max pixel in output:", max_tmp)
    map_np = tensor_to_np(map_tensor.mul(1.0/max_tmp).mul(255.0))
    heatmap_g = map_np.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_g, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (size,size), interpolation = cv2.INTER_AREA)
    return heatmap_color

def tensor_to_np(tensor):
    img = tensor.byte()
    # print(img.size())
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img

def add_weightedmap(origin_frame, mask):
    alpha = 0.3
    output_frame = origin_frame.copy()
    # print("output_frame.shape", output_frame.shape)
    # print("mask.shape", mask.shape)
    cv2.addWeighted(mask, alpha, origin_frame, 1-alpha, 0, output_frame) 
    return output_frame

if __name__ == '__main__':
    center = []
    with open('center.txt','rb') as f:
        center=pickle.load(f)
    print(center)

    times = []
    with open('times.txt','rb') as f:
        times=pickle.load(f)
    print(times)

    print(times * center)
    # times = np.ones((7,7),dtype=float)
    # for i in range(0, times.shape[0], 1):        
    #     for j in range(0, times.shape[1], 1):
    #         times[i, j] = 1.0 + 0.2*(abs(i-3)+abs(j-3))
    # print(times)
    # with open('times.txt','wb') as f_write:        
    #     pickle.dump(times, f_write)