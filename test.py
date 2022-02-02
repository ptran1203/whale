
import xml.etree.ElementTree as ET
import glob
import cv2
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataloader import val_transform

def extract_boxes_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    classes = []
    for i, obj in enumerate(root.iter('object')):
        xmlname = obj.find('name')
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymin = int(xmlbox.find('ymin').text)
        ymax = int(xmlbox.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))

        classes.append(xmlname.text)

    return boxes, classes

def build_test_imgs(img_dir):
    test_imgs = []

    for img_path in glob.glob(img_dir + '/**/*.jpg', recursive=True):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        xml_path = img_path.replace(".jpg", ".xml")
        boxes, classes = extract_boxes_from_xml(xml_path)
        for idx, bb in enumerate(boxes):
            
            x_min, y_min, x_max, y_max = boxes[idx]
            crop_img = img[y_min:y_max, x_min:x_max]
            cls = classes[idx]
            test_imgs.append((crop_img, cls))
    return test_imgs

## Test functions

def preprocess_img(img, img_size):
    img = cv2.resize(img, (img_size, img_size))
    img = val_transform(image=img)['image']
    return torch.from_numpy(img.transpose(2, 0 , 1))

def cosine(a, b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))

def run_test(model, device, test_imgs, img_size, infer_dir='', plot=False):
    model.eval()
    feat_mat = []
    with torch.no_grad():
        for img, label in test_imgs:
            feat = model(preprocess_img(img.copy(), img_size).to(device).unsqueeze(0))[0].cpu().numpy()
            feat_mat.append([feat, label])

    tp = 0
    fp = 0
    fn = 0
    c = 0
    same_cos = []
    diff_cos = []
    if plot:
        plt.figure(figsize=(20, 20))
    for i in range(len(feat_mat)):
        for j in range(len(feat_mat)):
            if i != j:
                f1, l1 = feat_mat[i]
                f2, l2 = feat_mat[j]
                cos = cosine(f1, f2)
                if l1 == l2 and cos >= 0.5:
                    tp += 1
                elif l1 == l2 and cos < 0.5:
                    fp += 1
                elif l1 != l2 and cos >= 0.5:
                    fn += 1
                    
                if plot:
                    if l1 == l2:
                        same_cos.append(cos)
                    else:
                        diff_cos.append(cos)
                    
                    if np.random.rand() <= 0.3 and c < 25 and (l1 == l2 if c < 15 else l1 != l2):
                        img1 = cv2.resize(test_imgs[i][0].copy(), (img_size, img_size))
                        img2 = cv2.resize(test_imgs[j][0].copy(), (img_size, img_size))
                        space = np.zeros((img1.shape[0], 10, 3))
                        img = np.concatenate([img1, space, img2], axis=1)
                        code = 'same' if l1 == l2 else 'diff'
                        code += f' - {cos:3f}'
                        plt.subplot(5,5, c + 1)
                        plt.title(code)
                        plt.imshow(img / 255.0)
                        c += 1
    
    if plot:
        plt.savefig(os.path.join(infer_dir, "example.jpg"))
        plt.close()

        plt.hist(same_cos, density=True, bins=20)
        plt.title('Same label')
        plt.ylabel('Density')
        plt.xlabel('Similarity')
        plt.savefig(os.path.join(infer_dir, "same_hist.jpg"))
        plt.close()

        plt.hist(diff_cos, density=True, bins=20)
        plt.title('Different label')
        plt.ylabel('Density')
        plt.xlabel('Similarity')
        plt.savefig(os.path.join(infer_dir, "diff_hist.jpg"))

    # F1 score
    return tp / (tp + 0.5 * (fp + fn)), (tp, fp, fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--outdir", type=str, default='runs/infer')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--img_dir", type=str, default='stamp_comp/img_test_20210118')

    args = parser.parse_args()

    test_imgs = build_test_imgs(args.img_dir)

    print(f"Test on {len(test_imgs)} images")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.weight, map_location='cpu')['model']
    model = model.to(device)

    os.makedirs(args.outdir, exist_ok=True)
    f1 = run_test(model, device, test_imgs, args.img_size, args.outdir, plot=True)
