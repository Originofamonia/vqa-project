"""
read saved box and feature pt file, parse to VQA dataset format
"""

import torch


def load_box_feat():
    filename = 'box_feat.pt'
    tensors = torch.load(filename)
    print(tensors['boxes'], tensors['feats'])


if __name__ == '__main__':
    load_box_feat()
