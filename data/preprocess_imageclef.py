"""
read saved box and feature pt file, parse to VQA dataset format
"""
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
import torch
from PIL import Image


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
        item = {}
        img = Image.open(filename)
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


if __name__ == '__main__':
    load_box_feat()
