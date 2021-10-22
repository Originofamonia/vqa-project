"""
https://mipt-oulu.github.io/solt/Medical_Data_Augmentation_CXR14.html
Visualize:
1. original image
2. plot all boxes with central points
3. plot winning box and its neighbors
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import os
import glob
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler


from run_imageclef import input_args
from sparse_graph_model import Model
from torch_dataset import ImageclefDataset, collate_fn
from utils import batch_to_cuda, xyxy2xywh


def save_plot_nodes():
    """
    1. get all boxes
    2. get winner box
    3. winner box neighbors
    """
    image_path = '/home/qiyuan/2021summer/imageclef/images'
    neighbors_list = [32]  # for 51 nodes best
    kernels_list = [32]
    args, parser, unparsed = input_args()
    args.n_kernels = kernels_list[0]
    args.neighbourhood_size = neighbors_list[0]

    model_file = os.path.join(args.save_dir, 'gcn_51_30.000.pt')
    dataset_test = ImageclefDataset(args, train=False)
    test_sampler = SequentialSampler(dataset_test)
    loader_test = DataLoader(dataset_test, batch_size=args.bsize,
                             sampler=test_sampler, shuffle=False,
                             num_workers=4, collate_fn=collate_fn)
    model = Model(vocab_size=dataset_test.q_words,
                  emb_dim=args.emb,
                  feat_dim=dataset_test.feat_dim,
                  hid_dim=args.hid,
                  out_dim=dataset_test.n_answers,
                  dropout=args.dropout,
                  neighbourhood_size=args.neighbourhood_size,
                  n_kernels=args.n_kernels,
                  pretrained_wemb=dataset_test.pretrained_wemb,
                  n_obj=args.n_obj)
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    model.eval()

    for i, test_batch in tqdm(enumerate(loader_test)):
        q_batch, a_batch, vote_batch, i_batch, k_batch, qlen_batch = \
            batch_to_cuda(test_batch)
        image_ids = test_batch[-1]
        logits, _ = model(q_batch, i_batch, k_batch, qlen_batch)
        for j, iid in enumerate(image_ids):
            boxes = i_batch[j][:, -4:]  # between [0, 1]
            boxes = xyxy2xywh(boxes)
            boxes = boxes.detach().cpu().numpy()
            img = cv2.imread(os.path.join(image_path, iid))
            height, width, channels = img.shape
            x1, x2 = boxes[:, 0] * width, boxes[:, 2] * width
            y1, y2 = boxes[:, 1] * height, boxes[:, 3] * height
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(img, cmap=plt.cm.Greys_r)

            for entry in zip(x1, y1, x2, y2):
                ax.add_patch(
                    Rectangle((entry[0], entry[1]), entry[2], entry[3], fill=False,
                              color='r', lw=2))
            box_file = f"{iid.strip('.jpg')}_boxes.jpg"
            plt.savefig(os.path.join(args.plot_dir, box_file))
            plt.close()


if __name__ == '__main__':
    save_plot_nodes()
