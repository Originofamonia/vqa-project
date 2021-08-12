"""
read saved box and feature pt file, parse to VQA dataset format
"""
import pandas as pd
import zarr
from tqdm import tqdm
import torch


def load_box_feat():
    fieldnames = ['image_id', 'image_w', 'image_h',
                  'num_boxes', 'boxes', 'features']

    filename = 'box_feat.pt'
    tensors = torch.load(filename)
    # print(tensors['boxes'], tensors['feats'])
    boxes = zarr.open_group('imageclef_boxes.zarr', mode='w')
    features = zarr.open_group('imageclef_features.zarr', mode='w')
    image_size = {}

    for i, (box, feat, image_id) in enumerate(zip(tensors['box'],
                                                  tensors['feat'],
                                                  tensors['image_id'])):
        bbox = box[0][:, :4].cpu().detach().numpy()
        feat = feat.cpu().detach().numpy()


if __name__ == '__main__':
    load_box_feat()
