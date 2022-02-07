import os
import numpy as np
import argparse
import torch
from dataloader import InferDataset, val_transform
from tqdm.auto import tqdm
from utils import pickle_save, pickle_load

def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.weight, map_location='cpu')['model']
    model = model.to(device)
    model.eval()

    os.makedirs(args.output, exist_ok=True)

    dataset = InferDataset(args.source, args.img_size, transform=val_transform)
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset)

    train_embs = pickle_load(args.train_embs)

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
                res_dict[img_id] = [emb, top5_pred, top5_conf]

    pickle_save(res_dict, os.path.join(args.output, 'test_embs.npy'))
    return res_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--source', type=str, default='test_images')
    parser.add_argument('--output', type=str, default='inferences/infer')
    parser.add_argument('--train_embs', default='train_embs.npy')

    args = parser.parse_args()
    infer(args)