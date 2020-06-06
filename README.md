facial-keypoints-detection-FCN  
==============================
Kaggle "Facial Keypoints Detection" competition, using FCN, pytorch
-------------------------------------------------------------------
# result
----    train   val   private   public<br>
rmse    0.719   1.786   1.462   1.783<br>
@400epoch<br>
# where does it come from
https://fairyonice.github.io/Achieving-top-5-in-Kaggles-facial-keypoints-detection-using-FCN.html<br>
# what's different
## augment
Only random expand and crop(no affine and horizontal flip)<br>
## locate target point from heatmap
Using top 35 points(not 25)<br>
# requirements
python3.7, pytorch1.2.0, numpy, pandas, opencv, etc.<br>
# devices
One GTX 1080ti for training(if you test or visualize on one image, cpu is ok)<br>
# data
download from https://www.kaggle.com/c/facial-keypoints-detection/data<br>
# train
## step 1
run Train_FacialFCN.py with:<br>
python Train_FacialFCN.py<br>
It uses only the 2140 faces without nan point. Should be able to reach rmse 3.1 on kaggle test<br>
Also you have to edit Config.py to locate the right path of dataset, and the path to save checkpoint, etc.<br>
## step 1
run Train_FacialFCN.py with:<br>
python Train_FacialFCN.py<br>
It uses only the 2140 faces without nan point. Should be able to reach rmse 3.1 on kaggle test<br>
## step 2
run Train_WeightedFacialFCN.py with, loading your best model in step1 as arg of --resume as well:<br>
python Train_WeightedFacialFCN.py<br>
It uses all the faces in trainset, but gain no loss in point-missing channels. <br>
# test and validate
run Test_FacialFCN.py with:<br>
python Test_FacialFCN.py<br>
It will load the best checkpoint and fill a csv file with output on testset. The file is available to upload.<br>
Also uncomment the function calls start with "utils.save" if you want to visualize the data and output.<br>
# license
No license. Leave a star if it helps you out with some experiments or assignments.<br>
