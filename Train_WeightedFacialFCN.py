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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
parser = argparse.ArgumentParser(description='N/A')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=Config.batch_size, type=int, metavar='N', help='mini-batch size (default: %d)'%Config.batch_size)
parser.add_argument('--lr', default=Config.lr, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default=Config.checkpoint_path, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
def main():
    global arg
    arg = parser.parse_args()
    print (arg)
    train_dataset = facial_data.FLandmarksDataset(csv_file=Config.trainset_path, 
                                        mode="train", replace_nan = "weighted", 
                                        aug_transform=CPM_train_aug, 
                                        classes=15, sigma=3)
    train_dataloader = data.DataLoader(train_dataset, Config.batch_size,
                                  num_workers=Config.data_load_number_worker,
                                  shuffle=True)

    val_dataset  = facial_data.FLandmarksDataset(csv_file=Config.trainset_path, 
                                        mode="val", replace_nan = "weighted", 
                                        aug_transform=CPM_val_aug, 
                                        classes=15, sigma=3)
    val_dataloader = data.DataLoader(val_dataset, Config.batch_size,
                                  num_workers=Config.data_load_number_worker,
                                  shuffle=False)
    model = New_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        train_loader=train_dataloader,
                        test_loader=val_dataloader
    )
    #Training
    model.run()
                         
class New_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader):
        self.start_epoch = 0
        self.epoch = 0
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader

        self.best_perf = 9999.9
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
                # self.best_perf = 999.0
                self.best_perf = checkpoint['best_perf']
                self.perf = checkpoint['perf'] 
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
            self.perf, self.loss_val = self.validate_1epoch()
            print("Eval result:\n    performance: %f \t\t loss:%f"%(self.perf, self.loss_val))
    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            if True:
                self.perf, self.loss_val = self.validate_1epoch() 
                is_best = self.perf <= self.best_perf
                # save model
                if is_best:
                    print("Got a better model! \n New perf: %f \t\t Last best perf:%f"%(self.perf, self.best_perf))
                    print(" New loss: %f \t\t Last best loss:%f"%(self.loss_val, self.best_loss))
                    self.best_perf = self.perf
                    self.best_loss = self.loss_val
                
                utils.save_checkpoint({
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),                    
                    'best_perf': self.best_perf,
                    'perf': self.perf,
                    'best_loss': self.best_loss,
                    'loss_val': self.loss_val,
                    'optimizer' : self.optimizer.state_dict()
                }, is_best, Config.checkpoint_path, Config.modelbest_path)    

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.myAverageMeter()
        perf = utils.myAverageMeter()
        
        #switch to train mode
        self.model.train()       
        
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader, ascii = True)
        for _, (image, label_map, label_pts, loss_weight) in enumerate(progress):
            
            if Config.use_cuda:
                image = image.cuda()
                label_map = label_map.cuda()
                loss_weight = loss_weight.cuda()

            # measure data loading time
            data_time.update(time.time() - end)            
            
            self.optimizer.zero_grad()
            pred, _ = self.model(image)
            loss = self.weighted_loss(pred, label_map, loss_weight)
            # loss = self.loss_func(pred, label_map)
            loss.backward()
            self.optimizer.step()

            # measure accuracy and record loss
            losses.update(loss.item(), image.size(0))
            pred_pts = utils.getPointByMap(pred)
            with torch.no_grad():
                rmse = 0.0
                for b in range(0, label_pts.size(0)):
                    x_mse = 0.0
                    y_mse = 0.0
                    b_count = 0.0
                    for p in range(0, label_pts.size(1)):
                        x_mse += loss_weight[b, p, 0] * ((label_pts[b, p, 0] - pred_pts[b, p, 0]).pow(2))
                        y_mse += loss_weight[b, p, 1] * ((label_pts[b, p, 1] - pred_pts[b, p, 1]).pow(2))
                        b_count += loss_weight[b, p, 0] + loss_weight[b, p, 1]
                    b_rmse = torch.sqrt((x_mse+y_mse)/ b_count)
                    rmse += b_rmse
                rmse = rmse / label_pts.size(0)
            perf.update(rmse, image.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Perf':[round(perf.average(),5)],
                'Loss':[round(losses.average(),5)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        utils.record_info(info, 'record/train.csv','train')
    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.myAverageMeter()
        perf = utils.myAverageMeter()
        
        #switch to eval mode
        self.model.eval()      
        
        end = time.time()
        # mini-batch training
        progress = tqdm(self.test_loader, ascii = True)
        with torch.no_grad():
            for _, (image, label_map, label_pts, loss_weight) in enumerate(progress):
                if Config.use_cuda:
                    image = image.cuda()
                    label_map = label_map.cuda()
                    loss_weight = loss_weight.cuda()

                # measure data loading time
                data_time.update(time.time() - end)            
                
                pred, _ = self.model(image)
                loss = self.weighted_loss(pred, label_map, loss_weight)
                # loss = self.loss_func(pred, label_map)

                # measure accuracy and record loss
                losses.update(loss.item(), image.size(0))
                pred_pts = utils.getPointByMap(pred)

                rmse = 0.0
                for b in range(0, label_pts.size(0)):
                    x_mse = 0.0
                    y_mse = 0.0
                    b_count = 0.0
                    for p in range(0, label_pts.size(1)):
                        x_mse += loss_weight[b, p, 0] * ((label_pts[b, p, 0] - pred_pts[b, p, 0]).pow(2))
                        y_mse += loss_weight[b, p, 1] * ((label_pts[b, p, 1] - pred_pts[b, p, 1]).pow(2))
                        b_count += loss_weight[b, p, 0] + loss_weight[b, p, 1]
                    b_rmse = torch.sqrt((x_mse+y_mse)/ b_count )    
                    rmse += b_rmse
                rmse = rmse / label_pts.size(0)
                
                perf.update(rmse, image.size(0))
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
    
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Perf':[round(perf.average(),5)],
                'Loss':[round(losses.average(),5)]
                }
        utils.record_info(info, 'record/test.csv','test')
        return round(perf.average(),5), round(losses.average(),5)
    def weighted_loss(self, arg1, arg2, weight):
        ret_loss = 0.0
        loss_count = 0.0
        for b in range(weight.size(0)):
            for p in range(weight.size(1)):
                point_weight = weight[b, p, 0] * weight[b, p, 1]
                ret_loss += self.loss_func(arg1[b, p, :, :], arg2[b, p, :, :]) * point_weight
                loss_count += point_weight
        if loss_count > 0:
            ret_loss = ret_loss / loss_count
        return ret_loss

if __name__ == '__main__':
    main()



