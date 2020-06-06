import torch
import os, sys
import time
import torch.backends.cudnn as cudnn
import Config
#if Config.use_cuda:
#    torch.set_default_tensor_type('torch.cuda.FloatTensor')

import argparse
import torch.nn as nn
import cv2
import utils
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

from data_loader import facial_data, CPM_train_aug, CPM_val_aug

from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description='N/A')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=Config.batch_size, type=int, metavar='N', help='mini-batch size (default: %d)'%Config.batch_size)
parser.add_argument('--lr', default=Config.lr, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default=Config.modelbest_path, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
def main():
    global arg
    arg = parser.parse_args()
    print (arg)

    val_dataset  = facial_data.FLandmarksDataset(csv_file=Config.trainset_path, 
                                        mode="val", replace_nan = "drop", 
                                        aug_transform=CPM_val_aug, 
                                        classes=15, sigma=3)
    val_dataloader = data.DataLoader(val_dataset, Config.batch_size,
                                  num_workers=Config.data_load_number_worker,
                                  shuffle=False)
    test_dataset  = facial_data.FacialKeypointsTestDataset(csv_file=Config.testset_path)
    test_dataloader = data.DataLoader(test_dataset, Config.batch_size,
                                  num_workers=Config.data_load_number_worker,
                                  shuffle=False)                              
    model = New_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        val_loader=val_dataloader,
                        test_loader=test_dataloader
    )
    #Training
    model.run()
                         
class New_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, val_loader, test_loader):
        self.start_epoch = 0
        self.epoch = 0
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.val_loader=val_loader
        self.test_loader=test_loader

        self.best_perf = 0.0
        self.perf= 0.0
        self.best_loss = 9999.9
        self.loss_val = 9999.9
        self.step_index = 0

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.vgg_model = VGGNet(pretrained=False, requires_grad=True)
        self.model = FCNs(pretrained_net=self.vgg_model, n_class=15) 
        if Config.use_cuda:
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

         
        self.loss_func = nn.MSELoss(size_average=True)  
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
    
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume, map_location="cpu")
                self.start_epoch = checkpoint['epoch'] + 1
                self.epoch = checkpoint['epoch']
                self.best_perf = checkpoint['best_perf']
                self.perf = checkpoint['perf'] # note!
                self.best_loss = checkpoint['best_loss']
                self.loss_val = checkpoint['loss_val']
                self.model.load_state_dict(checkpoint['state_dict'])
                if self.lr == Config.lr:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    print(self.optimizer)
                print("==> loaded checkpoint '{}' (epoch {}) (best_perf {})"
                  .format(self.resume, checkpoint['epoch'], self.best_perf))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            #self.epoch = 0
            self.perf, self.loss_val = self.validate_1epoch()
            print("Eval result:\n    performance: %f \t\t loss:%f"%(self.perf, self.loss_val))
    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        # test on testset and write
        test_result = self.test_set()
        save_struct = facial_data.write_result(test_result, lookup_path= Config.idLookupTable, write_path=Config.ResultPath)

        # test on valset and visualize
        self.perf, self.loss_val = self.validate_1epoch()

    def test_set(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        
        #switch to train mode
        self.model.eval()      
        
        end = time.time()
        # mini-batch training
        progress = tqdm(self.test_loader, ascii = True)
        with torch.no_grad():
            ret_list = []
            for _, data_dict in enumerate(progress):
                img_tensor = data_dict['image']
                pts_tensor = data_dict['img_id']
                if Config.use_cuda:
                    img_tensor = img_tensor.cuda()
                    pts_tensor = pts_tensor.cuda()

                # measure data loading time
                data_time.update(time.time() - end)            
                
                pred = self.model(img_tensor)

                pred_pts = utils.getPointByMap(pred, topk=35)

                for t in pred_pts.reshape(pred_pts.shape[0], -1):
                    for s in t:
                        ret_list.append(float(s.cpu()))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        return ret_list
    

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.myAverageMeter()
        perf = utils.myAverageMeter()
        
        #switch to train mode
        self.model.eval()      
        
        end = time.time()
        # mini-batch training
        progress = tqdm(self.val_loader, ascii = True)
        rmse_with_npoints = []
        for i in range(50):
            rmse_with_npoints.append([])
        with torch.no_grad():
            for _, (image, label_map, label_pts) in enumerate(progress):
                if Config.use_cuda:
                    image = image.cuda()
                    # 4D Tensor to 5D Tensor
                    label_map = label_map.cuda()

                # measure data loading time
                data_time.update(time.time() - end)            
                
                pred, deconv5_input = self.model(image)

                # # code to visualize feature_map
                # utils.save_feature_map(deconv5_input, "feature_alldata")
                # utils.save_image(image, "srcimg_alldata")
                # break

                loss = self.loss_func(pred, label_map)

                # measure accuracy and record loss
                losses.update(loss.item(), image.size(0))                
                
                # # code to study RMSE with topk (1/2)
                # for i in range(1, 49, 1):
                #     # pred_pts = utils.getPointByMap(label_map, topk=i)
                #     pred_pts = utils.getPointByMap(pred, topk=i)
                #     rmse = 0.0
                #     for b in range(0, label_pts.size(0)):
                #         x_mse = 0.0
                #         y_mse = 0.0
                #         for p in range(0, label_pts.size(1)):
                #             x_mse += (label_pts[b, p, 0] - pred_pts[b, p, 0]).pow(2)                        
                #             y_mse += (label_pts[b, p, 1] - pred_pts[b, p, 1]).pow(2)
                #         b_rmse = torch.sqrt((x_mse+y_mse)/(2*label_pts.size(1)))                    
                #         # print("RMSE of the item:", b_rmse)
                #         # print("pred :", pred_pts[b:, :, :])
                #         # print("label:", label_pts[b:, :, :])
                #         rmse += b_rmse
                #     rmse = rmse / label_pts.size(0)
                #     rmse_with_npoints[i].append(rmse.cpu().numpy())
                #     # print("RMSE @ top %02d:"%i, rmse)
                
                pred_pts = utils.getPointByMap(pred, topk=35)
                rmse = 0.0
                for b in range(0, label_pts.size(0)):
                    x_mse = 0.0
                    y_mse = 0.0
                    for p in range(0, label_pts.size(1)):
                        x_mse += (label_pts[b, p, 0] - pred_pts[b, p, 0]).pow(2)                        
                        y_mse += (label_pts[b, p, 1] - pred_pts[b, p, 1]).pow(2)
                    b_rmse = torch.sqrt((x_mse+y_mse)/(2*label_pts.size(1)))  
                    rmse += b_rmse
                rmse = rmse / label_pts.size(0)
                # print("RMSE of the batch:", rmse)
                
                perf.update(rmse, image.size(0))
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # # code to save predict heatmap and label heatmap
                # utils.save_predict(pred, image, "pred")
                # utils.save_predict(label_map, image, "label")
                # break
                
    
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Perf':[round(perf.average(),5)],
                'Loss':[round(losses.average(),5)]
                }
        print(info)

        # # code to study RMSE with topk (2/2)
        # for i in range(0, len(rmse_with_npoints), 1 ):
        #     src_data = np.array(rmse_with_npoints[i]) 
        #     avg_rmse = np.mean(src_data)
        #     rmse_with_npoints[i] = avg_rmse
        # f = open("./record/npoints_test.csv", "w+")
        # f.write("nPoints,RMSE\n")
        # for i in range(0, len(rmse_with_npoints), 1):
        #     f.write("{},{}".format(i, rmse_with_npoints[i]))
        #     if i < len(rmse_with_npoints)-1:
        #         f.write("\n")
        # f.close()
        # utils.record_info(info, 'record/test.csv','test')

        return round(perf.average(),5), round(losses.average(),5)

if __name__ == '__main__':
    main()



