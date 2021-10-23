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
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler


from run_imageclef import input_args
from sparse_graph_model import Model
from torch_dataset import ImageclefDataset, collate_fn
from utils import batch_to_cuda, xyxy2xywh


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
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],
                    thickness=tf, lineType=cv2.LINE_AA)


def plot_image(image, boxes, findings, paths=None, fname='images.jpg',
               names=None, max_size=1024, max_subplots=16):
    """
    image: [h, w, ch] ndarray
    boxes: [n, 4] ndarray, xyxy
    """
    # Plot image grid with labels

    # if isinstance(image, torch.Tensor):
    #     image = image.cpu().float().numpy()
    # for k, v in targets.items():
    #     if isinstance(v, torch.Tensor):
    #         targets[k] = v.cpu().numpy()

    # un-normalise
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

    # mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    # for i, img in enumerate(image):
    #     if i == max_subplots:  # if last batch has fewer images than we expect
    #         break

    # block_x = int(w * (i // ns))
    # block_y = int(h * (i % ns))

    if scale_factor < 1:
        image = cv2.resize(image, (w, h))

    mosaic = image
    # if len(targets) > 0:
    # image_targets = targets[targets[:, 0] == i]
    # boxes = targets['boxes']
    # classes = targets['labels']  # - 1
    # labels = [names[class_id] for class_id in
    #           classes]  # labels if no conf column
    conf = None  # check for confidence presence (label vs pred)

    if boxes.shape[0]:
        # absolute coords need scale if image scales
        boxes *= scale_factor
    # boxes[[0, 2]] += block_x
    # boxes[[1, 3]] += block_y
    for j, box in enumerate(boxes):
        # cls = names.index(findings[j])
        # color = colors[cls % len(colors)]
        # cls = names[cls] if names else cls
        # if len(findings) > 0:
            # label = '%s' % cls
        plot_one_box(box, mosaic, label=None, color=None,
                     line_thickness=tl)

    # Draw image filename labels
    # if paths:
    #     label = Path(paths[i]).name[:40]  # trim to 40 char
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
    #                 lineType=cv2.LINE_AA)

    # Image border
    # cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)),
                            interpolation=cv2.INTER_AREA).astype(np.uint8)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        # Image.fromarray(mosaic).save(fname)  # PIL save
    return mosaic


def plot_bbox(img_folder_path, bbox_filename):
    """
    https://github.com/thtang/CheXNet-with-localization/issues/9
    """
    actual_bbox = open(bbox_filename)
    img_folder_path = os.path.split(img_folder_path)[-1]
    # print(img_folder_path)
    count = 0
    temp_count = 0
    final_bbox_list = []
    for img in actual_bbox:
        if img.find(img_folder_path) != -1:
            print('file exist:', count)
            print('given image', img)
            temp_count = count
            print("this is temp count", temp_count)
        if count > temp_count:

            if img.find('/') == -1:
                final_bbox_list.append(img)
            else:
                break
        count += 1

    i = final_bbox_list[1]
    temp_i = list(i.split(" "))
    temp_i.pop(0)

    p = np.array(temp_i)
    k = p.astype(float)

    x1 = int(k[0])
    y1 = int(k[1])
    x2 = int(k[2])
    y2 = int(k[3])
    return x1, y1, x2, y2


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
    img_size = 640
    stride = 32
    shapes = [[1, 1]]
    pad = 0.0

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
            boxes = np.asarray(dataset_test.bbox[str(iid)])
            # boxes = i_batch[j][:, -4:]  # between [0, 1]
            # boxes = xyxy2xywh(boxes)
            # boxes = boxes.detach().cpu().numpy()
            img = load_image(iid, image_path)
            batch_shapes = np.ceil(
                np.array(shapes) * img_size / stride + pad).astype(
                np.int) * stride
            shape = batch_shapes[0]
            img, ratio, pad = letterbox(img, shape, auto=False,
                                        scaleup=False)

            f = os.path.join(args.plot_dir, f"{iid.strip('.jpg')}_boxes.jpg")
            plot_image(img, boxes, None, None, f, None)


def load_image(iid, image_path):
    img = cv2.imread(os.path.join(image_path, iid))
    assert img is not None, 'Image Not Found ' + os.path.join(image_path, iid)
    h0, w0 = img.shape[:2]  # orig hw
    r = 640 / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                         interpolation=interp)
    return img


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True,
              scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[
            0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=color)  # add border
    return img, ratio, (dw, dh)


if __name__ == '__main__':
    save_plot_nodes()
