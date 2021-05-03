import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import CustomDataset
from model import EAST
from loss import Loss
import os
import time
import logging
import numpy as np


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval, logger, start_from=None, start_from_epoch=0):
    logger.info(f"begin training with: batch_size = {batch_size}, lr = {lr}, start_from_epoch = {start_from_epoch}, start_from = {start_from}")
    
    file_num = len(os.listdir(train_img_path))
    trainset = CustomDataset(train_img_path, train_gt_path)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)
    
    criterion = Loss(logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EAST()
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    if start_from:
        model.load_state_dict(torch.load(start_from))
        logger.info("start from weights {}".format(start_from))
    if start_from_epoch:
        weights_path = "model_epoch_{}.pth".format(start_from_epoch)
        model.load_state_dict(torch.load(os.path.join(pths_path, weights_path)))
        logger.info("start from epoch {}".format(weights_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

    for epoch in range(start_from_epoch, epoch_iter):    
        model.train()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
            pred_score, pred_geo = model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
            
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.info('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
        
        logger.info('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
        logger.info(str(time.asctime(time.localtime(time.time()))))
        logger.info('='*50)
        if (epoch + 1) % interval == 0:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))
        scheduler.step()

if __name__ == '__main__':
    train_img_path = "/home/evjeny/data_dir/perimetry_text_detection_split/train_images"
    train_gt_path  = "/home/evjeny/data_dir/perimetry_text_detection_split/train_gts"
    pths_path      = './pths'
    start_from = None
    start_from_epoch = 0
    batch_size     = 6
    lr             = 1e-3
    num_workers    = 8
    epoch_iter     = 600
    save_interval  = 1
    
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler("log_train.txt")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval, rootLogger, start_from, start_from_epoch)
