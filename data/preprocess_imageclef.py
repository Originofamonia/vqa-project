"""
read saved box and feature pt file, parse to VQA dataset format
"""
import os
import csv
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
import torch
from PIL import Image


def parse_box_feat():
    # fieldnames = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes',
    #               'features']

    filename = 'box_feat.pt'
    imgpath = '/home/qiyuan/2021summer/imageclef/images/'
    tensors = torch.load(filename)

    boxes = zarr.open_group('imageclef_boxes.zarr', mode='w')
    features = zarr.open_group('imageclef_features.zarr', mode='w')
    image_size = {}

    for i, (box, feat, image_id) in enumerate(zip(tensors['box'],
                                                  tensors['feat'],
                                                  tensors['image_id'])):
        item = {}
        img = Image.open(os.path.join(imgpath, image_id))
        item['image_id'] = image_id
        item['boxes'] = box[0][:, :4].cpu().detach().numpy()
        item['features'] = feat.cpu().detach().numpy()
        item['num_boxes'] = box[0].size()[0]
        item['image_w'], item['image_h'] = img.width, img.height
        # append to zarr files
        boxes.create_dataset(item['image_id'], data=item['boxes'])
        features.create_dataset(item['image_id'], data=item['features'])
        # image_size dict
        image_size[item['image_id']] = {
            'image_h': item['image_h'],
            'image_w': item['image_w'],
        }

    # convert dict to pandas dataframe
    # create image sizes csv
    print('Writing image sizes csv...')
    df = pd.DataFrame.from_dict(image_size)
    df = df.transpose()
    d = df.to_dict()
    dw = d['image_w']
    dh = d['image_h']
    d = [dw, dh]
    dwh = {}
    for k in dw.keys():
        dwh[k] = np.array([d0[k] for d0 in d])
    image_sizes = pd.DataFrame(dwh)
    image_sizes.to_csv('imageclef_image_size.csv')


def get_qa_pairs():
    """
    Get filtered QA pairs and save to a new txt file.
    """
    filename = 'box_feat.pt'
    tensors = torch.load(filename)
    image_ids = tensors['image_id']
    dataset_path = '/home/qiyuan/2021summer/imageclef'
    text0 = 'VQAnswering_2020_Train_QA_pairs.txt'
    text1 = 'VQAnswering_2020_Val_QA_Pairs.txt'
    text2 = 'VQA-Med-2021-VQAnswering-Task1-New-ValidationSet.txt'
    valid_qa_pairs = []
    with open(os.path.join(dataset_path, text0), 'r') as f:
        for row in csv.reader(f, delimiter='|'):
            print(row)


def process_text():
    filename = 'box_feat.pt'
    dataset_path = '/home/qiyuan/2021summer/imageclef'
    text0 = 'VQAnswering_2020_Train_QA_pairs.txt'
    text1 = 'VQAnswering_2020_Val_QA_Pairs.txt'
    text2 = 'VQA-Med-2021-VQAnswering-Task1-New-ValidationSet.txt'
    tensors = torch.load(filename)
    image_ids = tensors['image_id']


if __name__ == '__main__':
    # parse_box_feat()
    get_qa_pairs()
