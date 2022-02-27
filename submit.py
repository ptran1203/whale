"""
Script to submib to competition
"""

import subprocess
from utils import pickle_load
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
from evaluate import compute_sim, evaluate, map_per_image
import argparse

def l2norm(embs):
    return {k: v/np.linalg.norm(v) for k, v in embs.items()}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='.')
    parser.add_argument('--thr', default=0.5, type=float)
    parser.add_argument('--norm', action='store_true')

    args = parser.parse_args()
    infer_dir = args.src

    train_embs = pickle_load(f"{infer_dir}/train_embs.pkl")
    test_embs = pickle_load(f"{infer_dir}/test_embs.pkl")
    val_embs = pickle_load(f"{infer_dir}/val_embs.pkl")
    train_df = pd.read_csv('data/train_kfold.csv')

    if args.norm:
        train_embs = l2norm(train_embs)
        test_embs = l2norm(test_embs)
        val_embs = l2norm(val_embs)
        
        
    print("Embedding size", len(train_embs) + len(val_embs))

    val_df = train_df[train_df.subset == 'test'].reset_index()
    val_map = dict(zip(val_df.image, val_df.individual_id))
    train_map = dict(zip(train_df.image, train_df.individual_id))

    score, val_sim_df = evaluate(train_df, train_embs, val_embs)
    val_sim_df["gt"] = val_sim_df.image.map(val_map)
    val_sim_df["map"] = val_sim_df.apply(lambda row: map_per_image(row["gt"], row.predictions.split(" ")), axis=1)
    val_sim_df = val_sim_df.sort_values("map")
    
    print(f"Evaluate score={score:.4f}")

    sim_df = compute_sim(train_df, {**train_embs, **val_embs}, test_embs, thr=args.thr)
    sim_df[["image", "predictions"]].to_csv("submission.csv", index=False)
    print(sim_df.head())
    subprocess.check_output('kaggle competitions submit -c happy-whale-and-dolphin -f submission.csv -m "Message"'.split(" "))