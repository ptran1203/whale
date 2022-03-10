import os
import numpy as np
import pandas as pd
import argparse
import torch
from dataloader import InferDataset, WhaleDataset
from tqdm.auto import tqdm
from utils import pickle_save, pickle_load
from collections import defaultdict
import importlib
from trainer import get_embs

def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

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


def compute_sim(train_df, train_embs, test_embs, thr=0.65, norm=False):
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

    if norm:
        # train_v = l2norm_numpy(train_v)
        test_v = l2norm_numpy(test_v)

    cos = np.matmul(test_v, train_v.T)

    records = []

    for i, scores in enumerate(tqdm(cos)):
        sort_idx = np.argsort(scores)[::-1]
        top5 = [train_k[j] for j in sort_idx[:5]]
        # top5 = [train_map[x] for x in top5]

        if scores[sort_idx[0]] < thr:
            top5 = ['new_individual'] + top5[:4]
        else:
            top5 = top5[:1] + ['new_individual'] + top5[1:4]
        
        # print(test_k[i], [f"{train_k[j]}({scores[j]:.3f})" for j in sort_idx[:5]])
        # print(scores[-1])
        records.append([test_k[i], " ".join(top5)])

    sim_df = pd.DataFrame(records, columns=['image', 'predictions'])
    return sim_df


def compute_simv2(train_df, train_embs, test_embs, thr=0.65, norm=False):
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

    if norm:
        train_v = l2norm_numpy(train_v)
        test_v = l2norm_numpy(test_v)
    else:
        print("[WARN] You should use norm to apply threshold correctly")

    cos = np.dot(test_v, train_v.T)

    records = []

    for i, scores in enumerate(tqdm(cos)):
        sort_idx = np.argsort(scores)[::-1]
        top5 = [train_k[j] for j in sort_idx[:5]]
        # top5 = [train_map[x] for x in top5]

        if thr > 0.0:
            for j in range(5):
                if scores[sort_idx[j]] < thr:
                    top5 = top5[:j] + ['new_individual'] + top5[j:4]
                    break
        
        # print(top5, scores[sort_idx][:5])
        # print(test_k[i], [f"{train_k[j]}({scores[j]:.3f})" for j in sort_idx[:5]])
        # print(scores[-1])
        records.append([test_k[i], " ".join(top5)])
    sim_df = pd.DataFrame(records, columns=['image', 'predictions'])
    return sim_df

def evaluate(val_df, train_embs, val_embs, norm=False):
    sim_df = compute_sim(val_df, train_embs, val_embs, thr=1.0, norm=norm)

    label_map = dict(zip(val_df.image, val_df.individual_id))
    
    predictions = []
    labels = []
    for i, row in sim_df.iterrows():
        label = label_map[row['image']]
        pred = row['predictions'].split(" ")
        labels.append(label)
        predictions.append(pred)

    score = map_per_set(labels, predictions)
    return score, sim_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--output', type=str, default='inferences/')
    parser.add_argument("--img_dir", type=str, default='/content/whale-512/kaggle/working/data/train_images')
    parser.add_argument('--train_embs', default='train_embs.npy')
    parser.add_argument('--aug', default='aug1')

    args = parser.parse_args()

    df = pd.read_csv('data/train_kfold.csv')
    train_df = df[df.fold != 0].reset_index(drop=True)
    val_df = df[df.fold == 0].reset_index(drop=True)

    train_embs = get_embs(args, train_df, save_to=os.path.join(args.output, 'train_embs.pkl'))
    val_embs = get_embs(args, val_df, save_to=os.path.join(args.output, 'val_embs.pkl'))
    # train_embs = pickle_load(args.train_embs)

    score, _ = evaluate(df, train_embs, val_embs)

    print(f"Score={score:.4f}") 