from model import Net
from dataloader import WhaleDataset
from trainer import Trainer
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import random
import pandas as pd
import numpy as np
import torch
import importlib
import losses
import math
from sklearn.preprocessing import LabelEncoder
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup



def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="tf_efficientnet_b0_ns")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="runs/exp")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--warmup_epochs", default=1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--min_class_samples", type=int, default=0)
    parser.add_argument("--nrows", default=0, type=int)
    parser.add_argument("--img_dir", type=str, default='/content/jpeg-happywhale-384x384/train_images-384-384')
    parser.add_argument("--loss", type=str, default='ce', help='ce|ce_smooth|focal')
    parser.add_argument("--neck", type=str, default='D', help='D|F|N')
    parser.add_argument("--aug", type=str, default='aug1', help='aug config')


    return parser.parse_args()

def get_loss_fn(loss_type, n_labels):
    if loss_type == 'ce':
        return torch.nn.CrossEntropyLoss()
    elif loss_type == 'ce_smooth':
        return losses.CrossEntropyLossWithLabelSmoothing(n_labels, ls_=0.9)
    elif loss_type == 'focal':
        return losses.FocalLoss()

def main(args):
    df = pd.read_csv('data/train_kfold.csv')

    df = df[df['sample_count'] > args.min_class_samples]
    if args.nrows != 0:
        df = df.sample(args.nrows)

    df['label'] = LabelEncoder().fit_transform(df.individual_id)

    if args.skip_train:
        train_df = df
        val_df = df
    else:
        train_df = df[df.subset == 'train'].reset_index(drop=True)
        val_df = df[df.subset == 'test'].reset_index(drop=True)
        # val_df = val_df[val_df.label.isin(train_df.label.unique())]

    print(f'Train={len(train_df)}, validate={len(val_df)}')

    aug = importlib.import_module(f'augments.{args.aug}')
    train_transform, val_transform = aug.train_transform, aug.val_transform

    dataset = WhaleDataset(train_df, args.img_dir, args.img_size, transform=train_transform(args.img_size))
    val_data = WhaleDataset(val_df, args.img_dir, args.img_size, transform=val_transform(args.img_size))

    print("Train aug", dataset.transform)

    train_loader = DataLoader(dataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=2, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0, shuffle=False)

    print(f'nlabel={df.label.nunique()}, train={train_df.label.nunique()}, test={val_df.label.nunique()}')
    model = Net(args.backbone, df.label.nunique(), args.neck, pretrained=True)

    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, weight_decay=1e-5, momentum=0.9)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    num_train_steps = len(train_loader)
    print('Training steps:', num_train_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_train_steps * args.warmup_epochs, 
                                                        num_training_steps=int(num_train_steps * (args.epochs)))
    criterion = get_loss_fn(args.loss, df.label.nunique())
    trainer = Trainer(model, optimizer, criterion=criterion, scheduler=scheduler, cfg=args)
    if not args.skip_train:
        trainer.train(train_loader, val_loader)
    trainer.predict_on_train(train_df)

if __name__ == '__main__':
    init_seeds()
    main(parseargs())