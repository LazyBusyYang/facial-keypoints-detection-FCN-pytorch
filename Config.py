import os.path as osp
batch_size = 96
data_load_number_worker = 8
lr = 3e-3

trainset_path = "/home1/gaoyang/facial-keypoints-detection/training.csv"
testset_path = "/home1/gaoyang/facial-keypoints-detection/test.csv"
idLookupTable = "/home1/gaoyang/facial-keypoints-detection/IdLookupTable.csv"
ResultPath = "/home1/gaoyang/facial-keypoints-detection/result.csv"
checkpoint_path = '/home1/gaoyang/models/Facial_FCN/checkpoint.pth.tar'
modelbest_path = '/home1/gaoyang/models/Facial_FCN/model_best.tar'
use_cuda = True
