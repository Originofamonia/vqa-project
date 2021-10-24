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
import math
import random
import os
import glob
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler


from run_imageclef import input_args
from sparse_graph_model import Model
from torch_dataset import ImageclefDataset, collate_fn
from utils import batch_to_cuda, xyxy2xywh
from yolo_datasets import get_yolo_dataset


def color_list():
    # Return first 10 plt colors as (r,g,b)
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in
            plt.rcParams['axes.prop_cycle'].by_key()['color']]


colors = color_list()  # list of colors


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color if color is not None else [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # thickness must be integer
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    center = (int((x[0] + x[2]) / 2), int((x[1] + x[3]) / 2))
    cv2.circle(img, center, radius=0, color=color, thickness=tl*2)
    # if label:
    #     tf = max(tl - 1, 1)  # font thickness
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #     cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    #     cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],
    #                 thickness=tf, lineType=cv2.LINE_AA)


def plot_connect_lines(img, h_max_boxes, fname, color=None, line_thickness=None):
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    for i, box_i in enumerate(h_max_boxes):
        for j, box_j in enumerate(h_max_boxes[i:]):
            center_i = (
                int((box_i[0] + box_i[2]) / 2), int((box_i[1] + box_i[3]) / 2))
            center_j = (
            int((box_j[0] + box_j[2]) / 2), int((box_j[1] + box_j[3]) / 2))
            cv2.line(img, center_i, center_j, color=color, thickness=tl)

    if fname:
        cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # cv2 save

    return img


def plot_connect_lines2(img, rows, cols, boxes, fname, color=None, line_thickness=None):
    """
    plot by edge weight
    """
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    for i, box_i in enumerate(h_max_boxes):
        for j, box_j in enumerate(h_max_boxes[i:]):
            center_i = (
                int((box_i[0] + box_i[2]) / 2), int((box_i[1] + box_i[3]) / 2))
            center_j = (
            int((box_j[0] + box_j[2]) / 2), int((box_j[1] + box_j[3]) / 2))
            cv2.line(img, center_i, center_j, color=color, thickness=tl)

    if fname:
        cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # cv2 save

    return img


def plot_boxes(image, boxes, findings, paths=None, fname='images.jpg',
               names=None, max_size=1024, max_subplots=16):
    """
    image: [h, w, ch] ndarray
    boxes: [n, 4] ndarray, xyxy
    """
    # if isinstance(image, torch.Tensor):
    #     image = image.cpu().float().numpy()
    # for k, v in targets.items():
    #     if isinstance(v, torch.Tensor):
    #         targets[k] = v.cpu().numpy()

    # un-normalize
    if np.max(image) <= 1:
        image *= 255

    tl = 2  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    h, w, ch = image.shape  # height, width, ch
    bs = 1
    # bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    if scale_factor < 1:
        image = cv2.resize(image, (w, h))

    mosaic = image
    lower_red = np.uint8([0,100,100])  # RGB or HSV
    # hsv_red = cv2.cvtColor(lower_red, cv2.COLOR_BGR2HSV)
    white = np.uint8([0, 0, 100])
    # hsv_white = cv2.cvtColor(white, cv2.COLOR_BGR2HSV)
    color_step = (lower_red - white) / len(boxes)
    for j, box in enumerate(boxes):
        c = lower_red + j * color_step
        plot_one_box(box, mosaic, label=None, color=c,
                     line_thickness=tl)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)),
                            interpolation=cv2.INTER_AREA).astype(np.uint8)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save

    return mosaic


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
    results = []

    for i, test_batch in tqdm(enumerate(loader_test)):
        q_batch, a_batch, vote_batch, i_batch, k_batch, qlen_batch = \
            batch_to_cuda(test_batch)
        image_ids = test_batch[-1]
        logits, adj_mat, h_max_indices = model(q_batch, i_batch, k_batch, qlen_batch)

        qid_batch = test_batch[3]
        _, oix = logits.data.max(1)
        oix = oix.cpu().numpy()
        # record predictions
        for i, qid in enumerate(qid_batch):
            qid = int(qid.cpu().numpy())
            results.append(
                f"{dataset_test.vqa[qid]['image_id']},"
                f"{dataset_test.vqa[qid]['question']},"
                f"{dataset_test.a_itow[oix[i]]},"
                f"{dataset_test.vqa[qid]['answer']}")

        # topn, topn_ind = torch.max(adj_mat, dim=-1)  # select top n node_i
        # topn, topn_ind = torch.topk(topn, k=topn.size(1), dim=-1, sorted=True)
        # topn_ind = topn_ind.detach().cpu().numpy()

        topm, topm_ind = torch.topk(  # select topm neighbors node_j
            adj_mat, k=args.neighbourhood_size, dim=-1, sorted=True)
        topm = torch.stack(  # all edges
            [F.softmax(topm[:, k], dim=-1) for k in range(topm.size(1))]).transpose(0,
                                                                  1)  # (batch_size, K, neighbourhood_size)
        topm_degree = torch.count_nonzero(topm, dim=-1)
        topm_deg_sorted, topm_deg_ind = torch.sort(topm_degree, dim=-1)  # to sort boxes by degree
        topm_deg_ind = topm_deg_ind.detach().cpu().numpy()

        for j, iid in enumerate(image_ids):
            boxes = np.asarray(dataset_test.bbox[str(iid)])
            boxes = boxes[topm_deg_ind[j]]
            img_h, img_w = np.asarray(dataset_test.sizes[str(iid)])
            img = cv2.imread(os.path.join(image_path, iid))
            resized_img = cv2.resize(img, (img_h, img_w))

            f1 = os.path.join(args.plot_dir, f"{iid.strip('.jpg')}_boxes.jpg")
            mosaic = plot_boxes(resized_img, boxes, None, None, f1, None)

            # h_max_idx, count = np.unique(h_max_indices[j].detach().cpu().numpy(), return_counts=True)
            # count_sort_ind = np.argsort(-count)
            # h_max_boxes = boxes[h_max_idx[count_sort_ind][:10]]
            edges = topm[j].flatten()
            edges_sorted, edges_ind = torch.sort(edges, descending=True)
            rows = edges_ind // topm.size(1)
            cols = edges_ind % topm.size(-1)

            f2 = os.path.join(args.plot_dir, f"{iid.strip('.jpg')}_h_max.jpg")
            # plot_connect_lines(mosaic, h_max_boxes, f2, color=None, line_thickness=None)
            plot_connect_lines2(img, rows, cols, boxes, f2, color=None,
                        line_thickness=None)
    with open('infer_imageclef.csv', 'w') as f:
        f.write('image_id,question,prediction,answer\n')
        for line in results:
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    save_plot_nodes()
