from model import Net
from dataloader import StampDataset, train_transform, val_transform
from trainer import Trainer
from torch.utils.data import DataLoader
from madgrad import MADGRAD
import torch.optim as optim
import argparse
import random
import numpy as np
import torch

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
    parser.add_argument("--outdir", type=str, default="runs/exp")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--test_img_dir", type=str, default='stamp_comp/img_test_20210118')

    return parser.parse_args()

def main(args):
    dataset = StampDataset('stamp_comp/Sorted_data1', args.img_size, transform=train_transform)

    train_loader = DataLoader(dataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=2, shuffle=True, drop_last=True)
    model = Net(args.backbone, dataset.n_classes, pretrained=True)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # optimizer = MADGRAD(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    trainer = Trainer(model, optimizer, scheduler=scheduler, cfg=args)
    trainer.train(train_loader, cfg=args)

if __name__ == '__main__':
    init_seeds()
    main(parseargs())