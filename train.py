from model import Net
from dataloader import WhaleDataset
from trainer import Trainer, get_embs
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
import pandas as pd
import numpy as np
import os
import torch
import importlib
import losses
import math
from losses import ce_loss
from sklearn.preprocessing import LabelEncoder
from utils import get_cosine_schedule_with_warmup, GradualWarmupSchedulerV2

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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--freeze_bn", action="store_true")
    parser.add_argument("--m", type=float, default=0.3)
    parser.add_argument("--cv_aug", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ohem", action="store_true")
    parser.add_argument("--warmup_epochs", default=1, type=int)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--min_class_samples", type=int, default=0)
    parser.add_argument("--nrows", default=0, type=int)
    parser.add_argument("--pool", default='gem', type=str)
    parser.add_argument("--gradient_accum_steps", default=1, type=int, help='Gradient accumulation steps')
    parser.add_argument("--img_dir", type=str, default='/content/jpeg-happywhale-384x384/train_images-384-384')
    parser.add_argument("--loss", type=str, default='ce', help='ce|focal')
    parser.add_argument("--neck", type=str, default='F', help='D|F|N')
    parser.add_argument("--ls_eps", type=float, default=0.0, help='label smoothing eps')
    parser.add_argument("--aug", type=str, default='aug1', help='aug config')
    parser.add_argument("--triplet_w", type=float, default=0.0)
    parser.add_argument("--head", type=str, default='arcface', help='arcface|adacos')
    
    args = parser.parse_args()
    # for arg in vars(args):
    #     print(f'{arg}={getattr(args, arg)}')
    # exit()
    print(args)
    return args

def get_loss_fn(loss_type, n_labels):
    if loss_type == 'ce':
        return ce_loss
    elif loss_type == 'focal':
        return losses.FocalLoss()

def main(args):

    if args.device == "cuda":
        print(f"GPU {torch.cuda.get_device_name(0)}")
    # if "P100" in torch.cuda.get_device_name(0):
    #     args.amp = False
    #     print("Turn off amp when using P100")

    df = pd.read_csv('data/train_kfold.csv')

    df = df[df['sample_count'] >= args.min_class_samples]
    if args.nrows != 0:
        df = df.sample(args.nrows)

    df['label'] = LabelEncoder().fit_transform(df.individual_id)
    n_classes = df.label.nunique()

    
    # train_df = df[df.subset == 'train'].reset_index(drop=True)
    # val_df = df[df.subset == 'test'].reset_index(drop=True)
    train_df = df[df.fold != args.fold].reset_index(drop=True)
    val_df = df[df.fold == args.fold].reset_index(drop=True)

    print(f'Train={len(train_df)}, validate={len(val_df)}')

    aug = importlib.import_module(f'augments.{args.aug}')
    train_transform, val_transform = aug.train_transform, aug.val_transform

    dataset = WhaleDataset(train_df, args.img_dir, args.img_size, transform=train_transform(args.img_size), cv2_aug=args.cv_aug)
    val_data = WhaleDataset(val_df, args.img_dir, args.img_size, transform=val_transform(args.img_size))

    print("Train aug", dataset.transform)

    train_loader = DataLoader(dataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.workers, shuffle=True, drop_last =True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0, shuffle=False)

    print(f'nlabel={n_classes}, train={train_df.label.nunique()}, test={val_df.label.nunique()}')
    model = Net(args.backbone, n_classes, cfg=args, pretrained=True)

    optimizer = optim.SGD(filter(lambda l: l.requires_grad, model.parameters()), lr=args.init_lr, weight_decay=5e-4, momentum=0.9, nesterov=False)
    # optimizer = optim.Adam(filter(lambda l: l.requires_grad, model.parameters()), lr=args.init_lr, weight_decay=5e-4)

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warmup_epochs)
    scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=args.warmup_epochs,
                                        after_scheduler=cosine_scheduler)

    num_train_steps = len(train_loader)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_train_steps * args.warmup_epochs, 
    #                                                     num_training_steps=int(num_train_steps * (args.epochs)))
    criterion = get_loss_fn(args.loss, n_classes)
    trainer = Trainer(model, optimizer, criterion=criterion, scheduler=scheduler, cfg=args)
    weight = trainer.train(train_loader, val_loader)

    from evaluate import evaluate
    from infer import infer

    args.weight = weight
    args.source = args.img_dir.replace("train_images", "test_images")
    args.output = args.outdir

    train_embs = get_embs(args, train_df, save_to=os.path.join(args.outdir, 'train_embs.pkl'))
    val_embs = get_embs(args, val_df, save_to=os.path.join(args.outdir, 'val_embs.pkl'))

    score, sim_df = evaluate(pd.read_csv('data/train_kfold.csv'), train_embs, val_embs)
    print(f"Eval={score:.4f}")
    
    infer(args)

if __name__ == '__main__':
    init_seeds(42)
    main(parseargs())
