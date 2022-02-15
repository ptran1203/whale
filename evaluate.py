import os
import numpy as np
import pandas as pd
import argparse
import torch
from dataloader import InferDataset, WhaleDataset, val_transform
from tqdm.auto import tqdm
from utils import pickle_save, pickle_load
from collections import defaultdict
import augments


def create_val_embs(args, val_df):
    aug = getattr(augments, args.aug)
    val_transform = aug.val_transform

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.weight, map_location='cpu')['model']
    model = model.to(device)
    model.eval()

    os.makedirs(args.output, exist_ok=True)

    dataset = WhaleDataset(val_df, args.img_dir, args.img_size, transform=val_transform(args.img_size))
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset)

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

    pickle_save(res_dict, os.path.join(args.output, 'val_embs.pkl'))
    return res_dict


def dict2list(embs):
    keys = list(embs.keys())
    values = [embs[k] for k in keys]
    return keys, np.stack(values)

def map_per_image(label, predictions):
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0

def map_per_set(labels, predictions):
    """
    Competition metric
    """
    return np.mean([map_per_image(l, p) for l,p in zip(labels, predictions)])

def compute_sim(train_df, train_embs, test_embs):
    # Compute center of each individual id
    label2emb = defaultdict(list)
    for label, d in train_df.groupby('individual_id'):
        for img_id in d.image.values:
            if img_id in train_embs:
                label2emb[label].append(train_embs[img_id])

    for k, v in label2emb.items():
        avg = np.mean(np.stack(v), 0)
        label2emb[k] = avg / np.linalg.norm(avg)

    train_k, train_v = dict2list(label2emb)
    test_k, test_v = dict2list(test_embs)

    cos = np.matmul(test_v, train_v.T)

    records = []

    for i, scores in enumerate(tqdm(cos)):
        sort_idx = np.argsort(scores)[::-1]
        top5 = [train_k[j] for j in sort_idx[:5]]

        for j in range(5):
            sim_score = scores[sort_idx[j]]
            if j == 1:#sim_score < 0.5:
                top5 = top5[:j] + ['new_individual'] + top5[j:4]
                break
            
        records.append([test_k[i], " ".join(top5)])

    sim_df = pd.DataFrame(records, columns=['image', 'predictions'])
    return sim_df

def evaluate(val_df, train_embs, val_embs):
    sim_df = compute_sim(val_df, train_embs, val_embs)

    label_map = dict(zip(val_df.image, val_df.individual_id))
    
    predictions = []
    labels = []
    for i, row in sim_df.iterrows():
        label = label_map[row['image']]
        pred = row['predictions'].split(" ")
        labels.append(label)
        predictions.append(pred)

    score = map_per_set(labels, predictions)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--output', type=str, default='inferences/eval')
    parser.add_argument("--img_dir", type=str, default='/content/whale-512/kaggle/working/data/train_images')
    parser.add_argument('--train_embs', default='train_embs.npy')
    parser.add_argument('--aug', default='aug1')

    args = parser.parse_args()

    df = pd.read_csv('data/train_kfold.csv')
    train_df = df[df.subset == 'train']
    val_df = df[df.subset == 'test']

    val_embs = create_val_embs(args, val_df)
    train_embs = pickle_load(args.train_embs)

    score = evaluate(val_df, train_embs, val_embs)
    print(f"Score={score:.4f}")