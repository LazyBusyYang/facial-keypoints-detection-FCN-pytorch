import os.path as osp
batch_size = 96
data_load_number_worker = 8
lr = 3e-3

trainset_path = "/yourpath/facial-keypoints-detection/training.csv"
testset_path = "/yourpath/facial-keypoints-detection/test.csv"
idLookupTable = "/yourpath/facial-keypoints-detection/IdLookupTable.csv"
ResultPath = "/yourpath/facial-keypoints-detection/result.csv"
checkpoint_path = '/yourpath/models/Facial_FCN/checkpoint.pth.tar'
modelbest_path = '/yourpath/models/Facial_FCN/model_best.tar'
use_cuda = True
