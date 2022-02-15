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
from dataloader import val_transform, WhaleDataset
from utils import pickle_save, pickle_load


def denorm(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    return (img * 255).astype(np.uint8)

def get_train_logger(log_dir='./logs'):
    logger = logging.getLogger('stamp')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    # logger.addHandler(logging.StreamHandler())
    return logger

def load_my_state_dict(model, state_dict):
 
        own_state = model.state_dict()
        for name, param in state_dict.items():
            try:
                if name not in own_state:
                    continue
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)
            except Exception as e:
                print(f"Skip {name}: {e}")

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
        if self.cfg.amp:
            scaler = amp.GradScaler()

        with torch.set_grad_enabled(is_train):
            for images, labels, _ in bar:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                # print(labels)

                if is_train:
                    self.optim.zero_grad()
                
                if self.cfg.amp:
                    with amp.autocast():
                        logit = self.model(images, labels)
                        loss = self.criterion(logit, labels)
                        if is_train:
                            scaler.scale(loss).backward() 
                            scaler.step(self.optim)
                            scaler.update()
                else:
                    logit = self.model(images, labels)
                    loss = self.criterion(logit, labels)
                    if is_train:
                        loss.backward()
                        self.optim.step()

                if is_train and self.scheduler is not None:
                    self.scheduler.step()  # onecyclelr

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

    def predict_on_train(self, train_df):
        """
        Run prediction on training dataset
        """
        weight_dir = os.path.join(self.cfg.outdir, "weights")
        last_ckp = os.path.join(weight_dir, f'{self.model_name}_last.pth')
        ckp = torch.load(last_ckp)
        load_my_state_dict(self.model, ckp['model'].state_dict())
        self.model = self.model.to(self.device)
        self.model.eval()
        dataset = WhaleDataset(train_df, self.cfg.img_dir, self.cfg.img_size, transform=val_transform(self.cfg.img_size))
        loader = torch.utils.data.DataLoader(dataset)
        res_dict = {}
        with torch.no_grad():
            for imgs, labels, ids in tqdm(loader):
                imgs = imgs.to(self.device)
                embs = self.model(imgs)
                embs = embs.cpu().numpy()
                for emb, id in zip(embs, ids):
                    res_dict[id] = emb

        pickle_save(res_dict, os.path.join(self.cfg.outdir, "train_embs.pkl"))
        return res_dict

    def train(self, train_loader, val_loader=None):
        """Train process"""
        cfg = self.cfg
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

        for imgs, labels, _ in train_loader:
            for i in range(len(imgs)):
                img, label = imgs[i], labels[i]
                img = img.numpy().transpose(1, 2 ,0)
                img = denorm(img)
                cv2.imwrite(os.path.join(log_example_dir, f"example_{i}.jpg"),img)
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
                load_my_state_dict(self.model, ckp['model'].state_dict())
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
