import os
import numpy as np
import argparse
import torch
from dataloader import InferDataset
from tqdm.auto import tqdm
from utils import pickle_save, pickle_load
import augments
import importlib

def infer(args):
    if args.device == 'tpu':
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(args.weight, map_location='cpu')['model']
    model = model.to(device)
    model.eval()
    aug = importlib.import_module(f'augments.{args.aug}')
    val_transform = aug.val_transform

    os.makedirs(args.output, exist_ok=True)

    transform = val_transform(args.img_size)

    print(transform)

    dataset = InferDataset(args.source, args.img_size, transform=transform)

    if args.device == 'tpu':
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
        SERIAL_EXEC = xmp.MpSerialExecutor()
        dataset = SERIAL_EXEC.run(lambda: dataset)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    if args.device == 'tpu':
        loader = pl.MpDeviceLoader(loader, device)

    # train_embs = pickle_load(args.train_embs)

    res_dict = {}
    with torch.no_grad():
        for imgs, paths in tqdm(loader):
            imgs = imgs.to(device)
            embs = model(imgs)
            logit = torch.softmax(embs, dim=-1)
            top5_conf, top5_pred = torch.topk(logit, 5, dim=1)
            embs = embs.cpu().numpy()
            for emb, path in zip(embs, paths):
                # print(emb)
                # img_id = os.path.basename(path)
                img_id = path
                res_dict[img_id] = emb

    pickle_save(res_dict, os.path.join(args.output, 'test_embs.pkl'))
    return res_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--source', type=str, default='test_images')
    parser.add_argument('--output', type=str, default='inferences')
    parser.add_argument('--train_embs', default='train_embs.npy')
    parser.add_argument('--aug', default='aug1')

    args = parser.parse_args()
    infer(args)