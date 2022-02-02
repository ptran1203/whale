from model import Net
from dataloader import WhaleDataset, train_transform, val_transform
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
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="runs/exp")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--nrows", default=0, type=int)
    parser.add_argument("--img_dir", type=str, default='/content/jpeg-happywhale-384x384/train_images-384-384')

    return parser.parse_args()

def main(args):
    df = pd.read_csv('train_kfold.csv')

    train_df = df[df.fold != args.fold]
    val_df = df[df.fold == args.fold]

    print(f'Train={len(train_df)}, validate={len(val_df)}')

    dataset = WhaleDataset(train_df, args.img_dir, args.img_size, transform=train_transform)
    val_data = WhaleDataset(train_df, args.img_dir, args.img_size, transform=val_transform)

    train_loader = DataLoader(dataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0, shuffle=False)

    model = Net(args.backbone, dataset.n_classes, pretrained=True)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # optimizer = MADGRAD(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    trainer = Trainer(model, optimizer, scheduler=scheduler, cfg=args)
    trainer.train(train_loader, val_loader, cfg=args)

if __name__ == '__main__':
    init_seeds()
    main(parseargs())