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
    Get filtered QA pairs and save to a new txt file. Only need once.
    """
    filename = 'box_feat.pt'
    tensors = torch.load(filename)
    image_ids = tensors['image_id']
    dataset_path = '/home/qiyuan/2021summer/imageclef'
    text0 = 'VQAnswering_2020_Train_QA_pairs.txt'
    text1 = 'VQAnswering_2020_Val_QA_Pairs.txt'
    text2 = 'VQA-Med-2021-VQAnswering-Task1-New-ValidationSet.txt'

    valid_qa_pairs0 = append_valid_qa_pairs(dataset_path, image_ids, text0)
    valid_qa_pairs1 = append_valid_qa_pairs(dataset_path, image_ids, text1)
    valid_qa_pairs2 = append_valid_qa_pairs(dataset_path, image_ids, text2)

    valid_qa_pairs0.extend(valid_qa_pairs1)
    valid_qa_pairs0.extend(valid_qa_pairs2)
    with open('valid_qa_pairs.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(valid_qa_pairs0)


def append_valid_qa_pairs(dataset_path, image_ids, text):
    valid_qa_pairs = []
    with open(os.path.join(dataset_path, text), 'r') as f:
        for row in csv.reader(f, delimiter='|'):
            if row[0] + '.jpg' in image_ids:
                valid_qa_pairs.append(row)
    return valid_qa_pairs


def process_text():
    filename = 'valid_qa_pairs.csv'
    # Combine questions and answers in the same json file
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(', '.join(row))
    for i, q in enumerate(tqdm(questions['questions'])):
        row = {}
        # load questions info
        row['question'] = q['question']
        row['question_id'] = q['question_id']
        row['image_id'] = str(q['image_id'])

        # load answers
        assert q['question_id'] == annotations[i]['question_id']
        row['answer'] = annotations[i]['multiple_choice_answer']

        answers = []
        for ans in annotations[i]['answers']:
            answers.append(ans['answer'])
        row['answers'] = collections.Counter(answers).most_common()

        data.append(row)

    json.dump(data, open('vqa_' + phase + '_combined.json', 'w'))


if __name__ == '__main__':
    # parse_box_feat()
    # get_qa_pairs()  # run once
    process_text()
