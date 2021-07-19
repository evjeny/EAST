import argparse
import os
import time
import logging

import torch
from torch.utils import data
from torch.optim import lr_scheduler

from dataset import CustomDataset
from model import EAST
from loss import Loss


def train(train_img_path, train_gt_path, pths_path, scopes, lr,
          num_workers, epoch_iter, interval, logger, start_from=None,
          start_from_epoch=0, preload_data=False):
    logger.info(f"begin training with: scopes = {scopes}, lr = {lr}, start_from_epoch = {start_from_epoch}, " +
                f"start_from = {start_from}, preload_data = {preload_data}, epochs = {epoch_iter}")
    
    if scopes is None:
        scopes = [{"scope": 256, "batch_size": 8, "min": 200, "max": 1000}]
    
    datasets = []
    train_loaders = []
    for scope_data in scopes:
        datasets.append(CustomDataset(train_img_path, train_gt_path, length=scope_data["scope"],
                                      min_image_size=scope_data["min"], max_image_size=scope_data["max"]))
        if preload_data:
            datasets[-1].init_sa()
            datasets[-1].preload_data()
        train_loaders.append(data.DataLoader(datasets[-1], batch_size=scope_data["batch_size"], shuffle=True,
                                             num_workers=num_workers))

    logger.info("prepared datasets")
    
    criterion = Loss(logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EAST()
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
        
        n_batches = sum(len(loader) for loader in train_loaders)
        i = 0
        iterators = [iter(loader) for loader in train_loaders]
        has_data = [True] * len(iterators)
        while any(has_data):
            for iter_num, iterator in enumerate(iterators):
                if not has_data[iter_num]:
                    continue
                
                try:
                    img, gt_score, gt_geo, ignored_map = next(iterator)
                except StopIteration:
                    has_data[iter_num] = False
                
                start_time = time.time()
                img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
                pred_score, pred_geo = model(img, scope=512)
                loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
                
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logger.info('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(
                    epoch+1, epoch_iter, i+1, n_batches, time.time()-start_time, loss.item()
                ))
                i += 1
        
        logger.info('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss / n_batches, time.time()-epoch_time))
        logger.info(str(time.asctime(time.localtime(time.time()))))
        logger.info('='*50)
        if (epoch + 1) % interval == 0:
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EAST trainer")
    parser.add_argument("--images_path", type=str, help="path to images folder")
    parser.add_argument("--gts_path", type=str, help="path to ground truths folder")
    parser.add_argument("--save_path", type=str, default="./pths", help="path to save weights")
    parser.add_argument("--start_from", type=str, default=None, help="start from weights file")
    parser.add_argument("--start_from_epoch", type=int, default=0, help="start from epoch num")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="num of workers for data loader")
    parser.add_argument("--epochs", type=int, default=600, help="epochs count")
    parser.add_argument("--preload_data", dest='preload_data', action='store_true', help="preload data")
    parser.set_defaults(preload_data=False)
    parser.add_argument("--save_interval", type=int, default=1, help="weights save interval (in epochs)")
    parser.add_argument("--log_file", type=str, default="log_train.txt", help="file to save logs")
    args = parser.parse_args()
    
    scopes = [{"scope": 256, "batch_size": 8, "min": 200, "max": 300}]
    
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(args.log_file)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    
    train(args.images_path, args.gts_path, args.save_path, scopes, args.lr, args.num_workers,
          args.epochs, args.save_interval, rootLogger, args.start_from,
          args.start_from_epoch, args.preload_data)
