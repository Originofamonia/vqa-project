"""
https://mipt-oulu.github.io/solt/Medical_Data_Augmentation_CXR14.html
Visualize:
1. original image
2. plot all boxes with central points
3. plot winning box and its neighbors
"""
from os.path import exists
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
import matplotlib.path as mpath
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

# from run_imageclef import input_args
from run import input_args
from sparse_graph_model import Model
from torch_dataset import ImageclefDataset, collate_fn, VQA_Dataset
from utils import batch_to_cuda, xyxy2xywh
from yolo_datasets import get_yolo_dataset


os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def color_list():
    # Return first 10 plt colors as (r,g,b)
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in
            plt.rcParams['axes.prop_cycle'].by_key()['color']]


colors = color_list()  # list of colors


def plot_one_box(box, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color if color is not None else [random.randint(0, 255) for _ in
                                             range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    # thickness must be integer
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
    cv2.circle(img, center, radius=0, color=color, thickness=tl * 2)
    # if label:
    #     tf = max(tl - 1, 1)  # font thickness
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #     cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    #     cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],
    #                 thickness=tf, lineType=cv2.LINE_AA)


def plot_connect_lines(img, h_max_boxes, fname, color=None,
                       line_thickness=None):
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


def plot_connect_lines2(img, boxes, rows, cols, fname, color=None,
                        line_thickness=None):
    """
    plot by edge weight
    """
    num_lines = 60
    rows = rows[:num_lines]
    cols = cols[:num_lines]
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 0  # line/font thickness
    # color = color or [random.randint(0, 255) for _ in range(3)]
    from_color = np.uint8([255, 0, 0])
    to_color = np.uint8([0, 0, 0])
    color_step = (to_color - from_color) / num_lines
    for i, (r, c) in enumerate(zip(rows, cols)):
        box_i = boxes[r]
        box_j = boxes[c]
        center_i = (
            int((box_i[0] + box_i[2]) / 2), int((box_i[1] + box_i[3]) / 2))
        center_j = (
            int((box_j[0] + box_j[2]) / 2), int((box_j[1] + box_j[3]) / 2))
        c = from_color + i * color_step
        cv2.line(img, center_i, center_j, color=c, thickness=tl)

    if fname:
        cv2.imwrite(fname, img)  # cv2 save
        # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

    tl = 1  # line thickness
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
    from_color = np.uint8([0, 0, 255])  # RGB or HSV
    # hsv_red = cv2.cvtColor(from_color, cv2.COLOR_BGR2HSV)
    to_color = np.uint8([0, 0, 0])
    # hsv_white = cv2.cvtColor(to_color, cv2.COLOR_BGR2HSV)
    color_step = (to_color - from_color) / len(boxes)
    for j, box in enumerate(boxes):
        c = from_color + j * color_step
        plot_one_box(box, mosaic, label=None, color=c,
                     line_thickness=tl)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)),
                            interpolation=cv2.INTER_AREA).astype(np.uint8)
        cv2.imwrite(fname, mosaic)  # cv2 save
        # cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
    return mosaic


def save_plot_nodes():
    """
    1. get all boxes
    2. get winner box
    3. winner box neighbors
    """
    # image_path = '/home/qiyuan/2021summer/imageclef/images'
    image_path = '/home/qiyuan/2021summer/vqa-project/data/coco/train2014'
    # coco_imgs = os.listdir(image_path)
    args, parser, unparsed = input_args()
    # args.n_kernels = kernels_list[0]
    # args.neighbourhood_size = neighbors_list[0]

    model_file = os.path.join(args.save_dir, 'vqa_36_8_16_54.17.pt')
    # dataset_test = ImageclefDataset(args, train=False)
    dataset_test = VQA_Dataset(args.data_dir, args.emb, train=True)
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
        idxs = test_batch[-1]  # vqa2.0 is idx, imageclef is iid
        if i == 100:
            break
        logits, adj_mat, h_max_indices = model(q_batch, i_batch, k_batch,
                                               qlen_batch)

        qid_batch = test_batch[3]
        _, oix = logits.data.max(1)
        oix = oix.cpu().numpy()

        # topn, topn_ind = torch.max(adj_mat, dim=-1)  # select top n node_i
        # topn, topn_ind = torch.topk(topn, k=topn.size(1), dim=-1, sorted=True)
        # topn_ind = topn_ind.detach().cpu().numpy()

        topm, topm_ind = torch.topk(  # select topm neighbors node_j
            adj_mat, k=args.neighbourhood_size, dim=-1, sorted=True)
        # topm = F.normalize(topm, p=2.0, dim=-1)
        # topm = torch.stack(  # all edges
        #     [F.softmax(topm[:, k], dim=-1) for k in
        #      range(topm.size(1))]).transpose(0, 1)  # (batch_size, K, neighbourhood_size)
        # print(topm[0], topm[1], topm[2], topm[3], topm[4])
        # topm_degree = torch.count_nonzero(topm, dim=-1)
        # print(topm_degree)
        # topm_deg_sorted, topm_deg_ind = torch.sort(topm_degree, dim=-1)  # to sort boxes by degree
        # topm_deg_ind = topm_deg_ind.detach().cpu().numpy()

        for j, idx in enumerate(idxs):
            idx = int(idx.cpu().numpy())
            iid = dataset_test.vqa[idx]['image_id']
            img_path = os.path.join(image_path,
                                    'COCO_train2014_000000' + str(iid) + '.jpg')
            img = cv2.imread(img_path)
            if img is None:
                continue
            results.append(
                f"{iid},"
                f"{dataset_test.vqa[idx]['question']},"
                f"{dataset_test.a_itow[oix[j]]},"
                f"{dataset_test.vqa[idx]['answer']}"
            )
            boxes = np.asarray(dataset_test.bbox[str(iid)])
            # sort boxes by sum of neighbors
            _, box_ind = torch.sort(torch.sum(topm[j], dim=1), dim=0)
            box_ind = np.asarray(box_ind.cpu())
            boxes = boxes[box_ind]
            img_h, img_w = np.asarray(dataset_test.sizes[str(iid)])

            resized_img = cv2.resize(img, (img_h, img_w))

            f1 = os.path.join(args.plot_dir,
                              f"{iid.strip('.jpg')}_{dataset_test.vqa[idx]['question'].strip('?')}_boxes.jpg")
            mosaic = plot_boxes(resized_img, boxes, None, None, f1, None)

            # h_max_idx, count = np.unique(h_max_indices[j].detach().cpu().numpy(), return_counts=True)
            # count_sort_ind = np.argsort(-count)
            # h_max_boxes = boxes[h_max_idx[count_sort_ind][:10]]
            edges = topm[j].flatten()
            edges_sorted, edges_ind = torch.sort(edges, descending=True)
            rows = torch.div(edges_ind, topm.size(1), rounding_mode='trunc')
            cols = edges_ind % topm.size(-1)
            # rows = rows.detach().cpu().numpy()
            # cols = cols.detach().cpu().numpy()
            real_ind = topm_ind[j][rows, cols]  # fetch real indices
            real_rows = torch.div(real_ind, adj_mat.size(1),
                                  rounding_mode='trunc')
            real_cols = real_ind % adj_mat.size(-1)
            f2 = os.path.join(args.plot_dir,
                              f"{iid.strip('.jpg')}_{dataset_test.vqa[idx]['question'].strip('?')}_lines.jpg")
            # plot_connect_lines(mosaic, h_max_boxes, f2, color=None, line_thickness=None)
            plot_connect_lines2(mosaic, boxes, real_rows, real_cols, f2,
                                color=None, line_thickness=None)

    with open(f'{args.plot_dir}/infer_vqa20.csv', 'w') as f:
        f.write('image_id,question,prediction,answer\n')
        for line in results:
            f.write(line)
            f.write('\n')


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'),
              norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_by_mpl():
    """
    plot with matplotlib
    https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
    https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
    """
    image_path = '/home/qiyuan/2021summer/vqa-project/data/coco/train2014'
    # coco_imgs = os.listdir(image_path)
    args, parser, unparsed = input_args()
    # args.n_kernels = kernels_list[0]
    # args.neighbourhood_size = neighbors_list[0]

    model_file = os.path.join(args.save_dir, 'vqa_36_8_16_54.17.pt')
    dataset = VQA_Dataset(args.data_dir, args.emb, train=True)
    test_sampler = SequentialSampler(dataset)
    loader_test = DataLoader(dataset, batch_size=args.bsize,
                             sampler=test_sampler, shuffle=False,
                             num_workers=4, collate_fn=collate_fn)

    model = Model(vocab_size=dataset.q_words,
                  emb_dim=args.emb,
                  feat_dim=dataset.feat_dim,
                  hid_dim=args.hid,
                  out_dim=dataset.n_answers,
                  dropout=args.dropout,
                  neighbourhood_size=args.neighbourhood_size,
                  n_kernels=args.n_kernels,
                  pretrained_wemb=dataset.pretrained_wemb,
                  n_obj=args.n_obj)
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    model.eval()
    results = []

    for i, test_batch in tqdm(enumerate(loader_test)):
        q_batch, a_batch, vote_batch, i_batch, k_batch, qlen_batch = \
            batch_to_cuda(test_batch)
        idxs = test_batch[-1]  # vqa2.0 is idx, imageclef is iid
        if i == 100:
            break
        logits, adj_mat, h_max_indices = model(q_batch, i_batch, k_batch,
                                               qlen_batch)

        qid_batch = test_batch[3]
        _, oix = logits.data.max(1)
        oix = oix.cpu().numpy()

        for j, idx in enumerate(idxs):
            idx = int(idx.cpu().numpy())
            iid = dataset.vqa[idx]['image_id']
            img_path = os.path.join(image_path,
                                    'COCO_train2014_000000' + str(iid) + '.jpg')
            if exists(img_path):
                im = plt.imread(img_path)
                boxes = np.asarray(dataset.bbox[str(iid)])  # xyxy
                boxes = sort_boxes(boxes, adj_mat[j])
                plot_box_edge_adj(args, boxes, dataset, idx, iid, im, adj_mat[j])
                # plot_connection_mpl(args, boxes, dataset, adj_mat, idx, iid, im)


def plot_given_fig():
    """
    reuse this one, change this
    plot fig4 in paper
    """
    task = 'val2014'
    image_path = f'/home/qiyuan/2021summer/vqa-project/data/coco/{task}'
    # coco_imgs = os.listdir(image_path)
    args, parser, unparsed = input_args()

    model_file = os.path.join(args.save_dir, 'vqa_36_8_16_83.44.pt')
    # model_file = os.path.join(args.save_dir, 'model_38.pth.tar')  # xinyue's model
    dataset = VQA_Dataset(args.data_dir, args.emb, train=False)
    question = 'What is the room?'
    iid = '325114'
    test_batch = get_iid_from_question(dataset, question, iid)
    # test_sampler = SequentialSampler(dataset)
    # loader_test = DataLoader(dataset, batch_size=args.bsize,
    #                          sampler=test_sampler, shuffle=False,
    #                          num_workers=4, collate_fn=collate_fn)

    model = Model(vocab_size=dataset.q_words,
                  emb_dim=args.emb,
                  feat_dim=dataset.feat_dim,
                  hid_dim=args.hid,
                  out_dim=dataset.n_answers,
                  dropout=args.dropout,
                  neighbourhood_size=args.neighbourhood_size,
                  n_kernels=args.n_kernels,
                  pretrained_wemb=dataset.pretrained_wemb,
                  n_obj=args.n_obj)
    model.load_state_dict(torch.load(model_file))  # mine
    # model.load_state_dict(torch.load(model_file)['state_dict'])  # xinyue's
    model = model.cuda()
    model.eval()

    q_batch, a_batch, vote_batch, i_batch, k_batch, qlen_batch = \
        batch_to_cuda(test_batch)
    idx = int(test_batch[-1])  # vqa2.0 is idx, imageclef is iid
    logits, adj_mat, h_max_indices = model(q_batch, i_batch, k_batch,
                                           qlen_batch)

    qid_batch = test_batch[3]
    _, oix = logits.data.max(1)
    oix = oix.cpu().numpy()
    iid = dataset.vqa[idx]['image_id']
    if len(iid) < 6:
        l_iid = len(iid)
        iid = '0' * (6 - l_iid) + iid
    caption = f"img id: {iid}, q: {dataset.vqa[idx]['question']}, \npred: " \
              f"{dataset.a_itow[oix[0]]}, ans: {dataset.vqa[idx]['answer']}"


    img_path = os.path.join(image_path,
                            f'COCO_{task}_000000' + str(iid) + '.jpg')
    if exists(img_path):
        im = plt.imread(img_path)
        boxes = np.asarray(dataset.bbox[str(dataset.vqa[idx]['image_id'])])  # xyxy [36, 4]
        plot_box_edge_adj(args, boxes, dataset, idx, iid, im, adj_mat[0], caption, edge_th=0.5)
        # plot_box_edge_pool(args, boxes, dataset, idx, iid, im, adj_mat[0],
        #                    h_max_indices, edge_th=0.1)
    print(caption)


def get_iid_from_question(dataset, question, iid):
    """
    return input batch in dataset given question and iid
    """
    questions = [q['question'] for q in dataset.vqa]
    images = [q['image_id'] for q in dataset.vqa]
    qs_arr = np.array(questions)
    im_arr = np.array(images)
    a = np.where(qs_arr == question)[0]
    b = np.where(im_arr == iid)[0]
    idx = a[np.in1d(a, b)][0]  # idx is idx in self.vqa[idx]

    # add return all inputs here, follow getitem
    qlen = len(dataset.vqa[idx]['question_toked'])
    q = [0] * 100
    for i, w in enumerate(dataset.vqa[idx]['question_toked']):
        try:
            q[i] = dataset.q_wtoi[w]
        except:
            q[i] = 0

    a = np.zeros(dataset.n_answers, dtype=np.float32)
    for w, c in dataset.vqa[idx]['answers_w_scores']:
        try:
            a[dataset.a_wtoi[w]] = c
        except:
            continue

    # number of votes for each answer
    n_votes = np.zeros(dataset.n_answers, dtype=np.float32)
    for w, c in dataset.vqa[idx]['answers']:
        try:
            n_votes[dataset.a_wtoi[w]] = c
        except:
            continue

    qid = dataset.vqa[idx]['question_id']
    iid = dataset.vqa[idx]['image_id']
    img = dataset.i_feat[str(iid)]
    bboxes = np.asarray(dataset.bbox[str(iid)])
    imsize = dataset.sizes[str(iid)]

    k = 36

    if np.logical_not(np.isfinite(img)).sum() > 0:
        raise ValueError

    for i in range(k):
        bbox = bboxes[i]
        bbox[0] /= imsize[0]
        bbox[1] /= imsize[1]
        bbox[2] /= imsize[0]
        bbox[3] /= imsize[1]
        bboxes[i] = bbox

    q = torch.from_numpy(np.asarray(q)).unsqueeze(0)
    a = torch.from_numpy(np.asarray(a).reshape(-1)).unsqueeze(0)
    n_votes = torch.from_numpy(np.asarray(n_votes).reshape(-1)).unsqueeze(0)
    qid = torch.from_numpy(np.asarray(qid).reshape(-1)).unsqueeze(0)
    i = torch.from_numpy(np.concatenate([img, bboxes], axis=1)).unsqueeze(0)
    k = torch.from_numpy(np.asarray(k).reshape(1)).unsqueeze(0)
    return q, a, n_votes, qid, i, k, torch.tensor([qlen]), idx


def sort_boxes(boxes, adj_mat):
    """
    sort boxes by sum of edges
    """
    adj_mat = adj_mat.detach().cpu().numpy()
    node_weights = adj_mat.sum(-1)
    node_indices = np.argsort(node_weights, axis=0)
    boxes = boxes[node_indices]
    return boxes


def plot_box_edge_adj(args, boxes, dataset, idx, iid, im, adj_mat, caption, edge_th):
    """
    reuse this function, change this
    From paper: box and edge denotes the node degree and edge weight
    plot boxes and edges by adj mat, box sort by sum rows, plot edge by adj
    https://www.tutorialspoint.com/how-do-you-create-line-segments-between-two-points-in-matplotlib
    """
    fig, ax = plt.subplots()
    ax.imshow(im)

    # sum adj by rows
    roi_weights = torch.sum(adj_mat, dim=-1)
    max_box = torch.max(roi_weights).item()
    roi_ws, roi_indices = torch.topk(roi_weights, k=3)
    roi_ws = roi_ws.detach().cpu().numpy()
    roi_indices = roi_indices.detach().cpu().numpy()
    selected_boxes = boxes[roi_indices]
    # n_boxes = len(selected_boxes)

    # plot boxes
    for i, box in enumerate(selected_boxes):
        w = box[2] - box[0]
        h = box[3] - box[1]
        c0 = (box[0] + box[2]) / 2
        c1 = (box[1] + box[3]) / 2
        # Create a Rectangle patch, xywh (xy is top left)
        box_weight = roi_ws[i] / max_box
        rect = Rectangle((box[0], box[1]), w, h, linewidth=(2 * box_weight), edgecolor='m',
                         facecolor='none', alpha=box_weight)
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.plot(c0, c1, 'm.', linewidth=(2 * box_weight), alpha=box_weight)  # can only use plt.plot, not ax, fig
    fig.text(0.01, 0.91, f'{caption}')
    f1 = os.path.join(args.plot_dir,
                      f"{iid.strip('.jpg')}_{dataset.vqa[idx]['question'].strip('?')}_boxes.jpg")
    plt.savefig(f1)

    # plot edges
    adj_mat = adj_mat.detach().cpu().numpy()
    max_edge = adj_mat.max()

    for i in range(len(roi_indices)):
        for j in range(i + 1, len(roi_indices)):
            box_idx_i = roi_indices[i]
            box_idx_j = roi_indices[j]
            edge_weight = adj_mat[box_idx_i][box_idx_j] / max_edge

            box_i = boxes[box_idx_i]
            box_j = boxes[box_idx_j]
            cix = (box_i[0] + box_i[2]) / 2
            ciy = (box_i[1] + box_i[3]) / 2
            cjx = (box_j[0] + box_j[2]) / 2
            cjy = (box_j[1] + box_j[3]) / 2
            x_values, y_values = [cix, cjx], [ciy, cjy]
            plt.plot(x_values, y_values, c='c', linewidth=2 * edge_weight,
                     alpha=edge_weight)

    f2 = os.path.join(args.plot_dir,
                      f"{iid.strip('.jpg')}_{dataset.vqa[idx]['question'].strip('?')}_lines.jpg")
    plt.savefig(f2)
    plt.close()


def plot_box(ax, box, ci0, ci1, h, w, i, n_boxes, edge_weight):
    rect = Rectangle((box[0], box[1]), w, h,
                     linewidth=(2 * edge_weight), edgecolor=np.random.rand(3,),
                     facecolor='none', alpha=2 * edge_weight)
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.plot(ci0, ci1, 'm.')


def plot_box_edge_pool(args, boxes, dataset, idx, iid, im, adj_mat, h_max_indices, edge_th):
    """
    My idea
    plot boxes and edges from maxpooling, box sort by graph after pool,
    edge by adj
    """
    obj_indices, obj_counts = torch.unique(h_max_indices, return_counts=True)
    sorted_counts, sort_indices = torch.sort(obj_counts, descending=True)
    obj_idx_order = obj_indices[sort_indices].detach().cpu().numpy()
    obj_idx_order = obj_idx_order[:10]
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(im)
    n_boxes = len(obj_idx_order)  # n_boxes to plot
    # plot boxes
    for i, box_idx in enumerate(obj_idx_order):
        box = boxes[box_idx]
        w = box[2] - box[0]
        h = box[3] - box[1]
        c0 = (box[0] + box[2]) / 2
        c1 = (box[1] + box[3]) / 2
        # Create a Rectangle patch, xywh (xy is top left)
        rect = Rectangle((box[0], box[1]), w, h, linewidth=(2 - i / n_boxes), edgecolor='m',
                         facecolor='none', alpha=(1 - i / n_boxes))
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.plot(c0, c1, 'm.')
    f1 = os.path.join(args.plot_dir,
                      f"{iid.strip('.jpg')}_{dataset.vqa[idx]['question'].strip('?')}_boxes_pool.jpg")
    plt.savefig(f1)
    # plot edges
    norm = plt.Normalize(0.0, 1.0)
    cmap = plt.get_cmap('jet')
    adj_mat = adj_mat.detach().cpu().numpy()
    z = np.linspace(0, 1, len(adj_mat))
    max_edge = adj_mat.max()
    for i in obj_idx_order:
        for j in obj_idx_order:
            edge_weight = adj_mat[i][j] / max_edge
            if edge_weight > edge_th:
                b_i = boxes[i]
                b_j = boxes[j]
                ci0 = (b_i[0] + b_i[2]) / 2
                ci1 = (b_i[1] + b_i[3]) / 2
                cj0 = (b_j[0] + b_j[2]) / 2
                cj1 = (b_j[1] + b_j[3]) / 2
                seg = np.array([[ci0, ci1], [cj0, cj1]])
                seg = np.expand_dims(seg, axis=0)
                lc = mcoll.LineCollection(seg, array=z, cmap=cmap, norm=norm,
                                          linewidth=2 * edge_weight, alpha=1 * edge_weight)
                ax.add_collection(lc)

    f2 = os.path.join(args.plot_dir,
                      f"{iid.strip('.jpg')}_{dataset.vqa[idx]['question'].strip('?')}_lines_pool.jpg")
    plt.savefig(f2)
    plt.close()


if __name__ == '__main__':
    # save_plot_nodes()
    # plot_by_mpl()
    # main()
    plot_given_fig()
