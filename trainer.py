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
from dataloader import WhaleDataset
from utils import pickle_save, pickle_load
import importlib
from losses import HardTripletLoss

def adjust_hard_ratio(ep):
    if ep < 3:
        hard_ratio = 1 * 1e-2
    elif ep < 10:
        hard_ratio = 7 * 1e-3
    elif ep < 15:
        hard_ratio = 6 * 1e-3
    elif ep < 20:
        hard_ratio = 5 * 1e-3
    else:
        hard_ratio = 4 * 1e-3
    return hard_ratio

def denorm(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    return (img[:,:,::-1] * 255).astype(np.uint8)

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


def get_embs(args, df, save_to=''):
    aug = importlib.import_module(f'augments.{args.aug}')
    val_transform = aug.val_transform

    if args.device == 'tpu':
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.weight, map_location='cpu')['model']

    model = model.to(device)
    model.eval()

    os.makedirs(args.output, exist_ok=True)
    transform = val_transform(args.img_size)
    print(transform)


    dataset = WhaleDataset(df, args.img_dir, args.img_size, transform=transform)

    if args.device == 'tpu':
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
        SERIAL_EXEC = xmp.MpSerialExecutor()
        dataset = SERIAL_EXEC.run(lambda: dataset)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    if args.device == 'tpu':
        loader = pl.MpDeviceLoader(loader, device)

    res_dict = {}
    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader):
            imgs = imgs.to(device)
            embs = model(imgs)
            logit = torch.softmax(embs, dim=-1)
            # top5_conf, top5_pred = torch.topk(logit, 5, dim=1)
            embs = embs.cpu().numpy()
            for emb, path in zip(embs, paths):
                # print(emb)
                # img_id = os.path.basename(path)
                img_id = path
                res_dict[img_id] = emb

    if save_to:
        pickle_save(res_dict, save_to)
    return res_dict


class Trainer:
    def __init__(self, model, optimizer, criterion=None, scheduler=None, cfg=None):
        self.model = model
        self.model_name = model.name
        self.optim = optimizer
        self.cfg = cfg
        self.scheduler = scheduler
        self.best_score = -1
        self.criterion = criterion
        self.tpu = not isinstance(cfg.device, str)
        print(f"Use TPU={self.tpu}")
        self.triplet_w = cfg.triplet_w
        if not isinstance(cfg.device, str):
            self.device = cfg.device
        else:
            self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))

        print("device", self.device, type(self.device))
        self.triplet_loss = HardTripletLoss()

    def init_logger(self, log_dir):
        self.logger = get_train_logger(log_dir)

    def run_epoch(self, loader, is_train=True, epoch=0):
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
            for batch_idx, (images, labels, _) in enumerate(bar):

                # labels = self.onehot()

                hard_ratio = adjust_hard_ratio(epoch) if self.cfg.ohem else 0.0

                if is_train and epoch >= self.cfg.warmup_epochs and np.random.rand() <= self.cfg.mixup:
                    # Do mixup
                    # images = torch.stack(images).cuda()
                    shuffle_indices = torch.randperm(images.size(0))
                    indices = torch.arange(images.size(0))
                    lam = np.clip(np.random.beta(1.0, 1.0), 0.35, 0.65)
                    images = lam * images + (1 - lam) * images[shuffle_indices, :]
                    labels = lam * labels + (1 - lam) * labels[shuffle_indices, :]

                images = images.to(self.device)
                labels = labels.to(self.device).long()

                do_update = is_train and ((batch_idx + 1) % self.cfg.gradient_accum_steps == 0) or (batch_idx + 1 == len(loader))

                # print(do_update)
                
                if self.cfg.amp:
                    with amp.autocast():
                        feat, logit = self.model(images, labels)
                        loss = self.criterion(logit, labels, ohem=hard_ratio)
                        if self.triplet_w > 0.0:
                            loss = loss + self.triplet_w * self.triplet_loss(feat, labels)
                        loss = loss / self.cfg.gradient_accum_steps
                        if is_train:
                            scaler.scale(loss).backward()
                            if do_update:
                                scaler.step(self.optim)
                                scaler.update()
                                # self.optim.zero_grad()
                                for param in self.model.parameters():
                                    param.grad = None
                else:
                    feat, logit = self.model(images, labels)
                    loss = self.criterion(logit, labels, ohem=hard_ratio)
                    if self.triplet_w > 0.0:
                        loss = loss + self.triplet_w * self.triplet_loss(feat, labels)
                    loss = loss / self.cfg.gradient_accum_steps
                    if is_train:
                        loss.backward()
                        if do_update:
                            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
                            if self.tpu:
                                import torch_xla.core.xla_model as xm
                                xm.optimizer_step(self.optim)
                            else:
                                self.optim.step()
                            # self.optim.zero_grad()
                            for param in self.model.parameters():
                                param.grad = None

                # if is_train and self.scheduler is not None:
                #     self.scheduler.step()

                # Compute metric score
                pred = torch.softmax(logit.detach(), dim=-1)
                pred_label = torch.argmax(pred, dim=-1)
                acc = ((pred_label == labels).sum() / len(labels)).item()
                loss_val = loss.item() * self.cfg.gradient_accum_steps
                if is_train:
                    msg_loss = f"loss: {loss_val:.4f} - acc: {acc:.4f}"
                    bar.set_description(msg_loss)
                scores["loss"].append(loss_val)
                scores["acc"].append(acc)

        scores = {k: (np.mean(v) if isinstance(v, list) else v) for k, v in scores.items()}
        
        return scores

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
                img = img.cpu().numpy().transpose(1, 2 ,0)
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
                ckp = torch.load(last_ckp, map_location='cpu')
                # self.optim.load_state_dict(ckp['optim'])
                # self.scheduler.load_state_dict(ckp['scheduler'])
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
            train_scores = self.run_epoch(train_loader, epoch=epoch)

            do_valid = epoch % 2 == 0

            if do_valid:
                test_scores = self.run_epoch(val_loader, is_train=False)

            lr = self.optim.param_groups[0]["lr"]

            msg = [f"Epoch {epoch + 1}/{epochs} (lr={lr:.5f})\nTrain "]
            msg += [f"{k}: {v:.5f}" for k, v in train_scores.items()]
            if do_valid:
                msg += ["\nVal "] + [f"{k}: {v:.5f}" for k, v in test_scores.items()]
            msg = ", ".join(msg)
            self.logger.info(msg)
            print(msg)

            if self.scheduler is not None:
                self.scheduler.step()

            if do_valid:
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
                            'optim': self.optim.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
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
                'optim': self.optim.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, last_ckp)

        self.logger.info(f"Training is completed, elapsed: {(time.time() - start):.3f}s")
        return last_ckp
