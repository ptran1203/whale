import os
import torch
import pandas as pd
import numpy as np
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import shutil
from scipy import spatial
import cv2
import torch.cuda.amp as amp
import logging
from dataloader import val_transform
# from test import build_test_imgs, run_test

def denorm(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    return (img * 255).astype(np.uint8)[:, :, ::-1]

def get_train_logger(log_dir='./logs'):
    logger = logging.getLogger('stamp')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    # logger.addHandler(logging.StreamHandler())
    return logger

scaler = amp.GradScaler()


class Trainer:
    def __init__(self, model, optimizer, criterion=nn.CrossEntropyLoss(), scheduler=None, cfg=None):
        self.model = model
        self.model_name = model.name
        self.optim = optimizer
        self.cfg = cfg
        self.scheduler = scheduler
        self.best_score = -1
        self.criterion = criterion
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))
        self.criterion.to(self.device)

    def init_logger(self, log_dir):
        self.logger = get_train_logger(log_dir)

    def run_epoch(self, loader, is_train=True):
        """
        Train/eval one epoch
        Args:
            loader: data loader
            optim: optimizer
            loss_func: loss function
            device: device
        Returns([dict]): metric score, e.g: {'f1': 0.99}
        """
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        bar = tqdm(loader) if is_train else loader
        scores = defaultdict(list)

        with torch.set_grad_enabled(is_train):
            for images, labels in bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # print(labels)
                
                if self.cfg.amp:
                    with amp.autocast():
                        logit = self.model(images, labels)
                        loss = self.criterion(logit, labels)
                else:
                    logit = self.model(images, labels)
                    loss = self.criterion(logit, labels)
                
                if is_train:
                    self.optim.zero_grad()
                    if self.cfg.amp:
                        scaler.scale(loss).backward() 
                        scaler.step(self.optim)
                        scaler.update()
                    else:
                        loss.backward()
                        self.optim.step()

                # Compute metric score
                pred = torch.softmax(logit.detach(), dim=-1)
                pred_label = torch.argmax(pred, dim=-1).cpu().numpy()
                acc = (pred_label == labels.cpu().numpy()).mean()
                if is_train:
                    msg_loss = f"loss: {loss.item():.4f} - acc: {acc:.4f}"
                    bar.set_description(msg_loss)
                scores["loss"].append(loss.item())
                scores["acc"].append(acc)

        scores = {k: (np.mean(v) if isinstance(v, list) else v) for k, v in scores.items()}
        
        return scores

    def train(self, train_loader, val_loader=None, cfg=None):
        """Train process"""
        output_dir = cfg.outdir
        epochs = cfg.epochs
        weight_dir = os.path.join(output_dir, "weights")
        log_dir = os.path.join(output_dir, 'logs')
        log_example_dir = os.path.join(log_dir, 'train_examples')
        os.makedirs(weight_dir, exist_ok=True)
        # os.path.exists(log_example_dir) and shutil.rmtree(log_example_dir)
        # infer_dir = os.path.join(log_dir, 'infer')
        # os.path.exists(infer_dir) and shutil.rmtree(infer_dir)
        # os.makedirs(infer_dir, exist_ok=True)
        os.makedirs(log_example_dir, exist_ok=True)

        self.init_logger(log_dir)
        early_stop_counter = 0

        for imgs, labels in train_loader:
            for i in range(len(imgs)):
                img, label = imgs[i], labels[i]
                img = img.numpy().transpose(1, 2 ,0)
                img = denorm(img)
                cv2.imwrite(os.path.join(log_example_dir, f"example_{label}_{i}.jpg"),img)
            break

        # history
        train_metrics = defaultdict(list)
        val_metrics = defaultdict(list)

        # load pretraineds
        start_epoch = 0
        last_ckp = os.path.join(weight_dir, f'{self.model_name}_last.pth')
        if cfg.resume:
            if os.path.exists(last_ckp):
                ckp = torch.load(last_ckp)
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in ckp['model'].state_dict().items() if k in model_dict}
                model_dict.update(pretrained_dict) 
                self.model.load_state_dict(pretrained_dict)
                start_epoch = ckp['epoch'] + 1
                self.logger.info(f"Resume training from epoch {start_epoch}")
                print(f"Resume training from epoch {start_epoch}")
            else:
                self.logger.info(f"{last_ckp} not found, train from scratch")
                print(f"{last_ckp} not found, train from scratch")

        # Train
        start = time.time()
        self.model.to(self.device)

        for epoch in range(start_epoch, epochs):
            train_scores = self.run_epoch(train_loader)
            test_scores = self.run_epoch(val_loader, is_train=False)

            lr = self.optim.param_groups[0]["lr"]

            if self.scheduler is not None:
                self.scheduler.step()  # onecyclelr

            msg = [f"Epoch {epoch + 1}/{epochs} (lr={lr:.5f})\nTrain "]
            msg += [f"{k}: {v:.5f}" for k, v in train_scores.items()]
            msg += ["\nVal "] + [f"{k}: {v:.5f}" for k, v in test_scores.items()]
            msg = ", ".join(msg)
            self.logger.info(msg)
            print(msg)

            score = test_scores['acc']
            if epoch > 1:
                if score > self.best_score:
                    m = f"score improved from {self.best_score:.4f} -> {score:.4f}, save model"
                    self.logger.info(m)
                    print(m)
                    self.best_score = score
                    early_stop_counter = 0
                    torch.save({
                        'epoch': epoch,
                        'model': self.model,
                    }, os.path.join(weight_dir, f'{self.model_name}_best.pth'))
                else:
                    early_stop_counter += 1

            # if early_stop_counter >= 3:
            #     print("Model doest not improve anymore, stop")
            #     break

            # Save last epoch
            torch.save({
                'epoch': epoch,
                'model': self.model,
            }, last_ckp)

        self.logger.info(f"Training is completed, elapsed: {(time.time() - start):.3f}s")
        return self.best_score
