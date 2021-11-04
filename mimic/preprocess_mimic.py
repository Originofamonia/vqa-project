"""
read saved box and feature pt file, parse to VQA dataset format
"""
import os
import csv
import json
import numpy as np
import pandas as pd
import zarr
import collections
from tqdm import tqdm
import torch
from PIL import Image
from spacy.tokenizer import Tokenizer
import spacy
import string
import pickle

nlp = spacy.load('en_core_web_sm')
tokenizer = Tokenizer(nlp.vocab)
exclude = set(string.punctuation)


def parse_box_feat(task):
    # fieldnames = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes',
    #               'features']
    n_obj = 17  # n_obj per image
    detect_file = f'mimic_detect_{task}.pt'
    gaze_file = f'mimic_gaze_{task}.pt'
    gaze_on_detect_file = f'mimic_gaze_on_detect_{task}.pt'
    # imgpath = '/home/qiyuan/2021summer/imageclef/images/'

    detect_tensors = torch.load(detect_file)
    gaze_tensors = torch.load(gaze_file)
    gaze_on_detect_tensors = torch.load(gaze_on_detect_file)
    # {'feat': selected_feats, 'image_id': filepaths}

    boxes = zarr.open_group(f'mimic_{task}_boxes.zarr', mode='w')
    features = zarr.open_group(f'mimic_{task}_features.zarr', mode='w')
    image_size = {}
    num_det_boxes = []
    num_gaze_boxes = []
    num_gaze_det_boxes = []
    # image_ids = []
    for i, (det_feat, image_id, img_sizes) in enumerate(zip(detect_tensors['feat'],
                                             detect_tensors['image_id'], detect_tensors['img_sizes'])):
        if det_feat.size(0) >= n_obj and image_id in gaze_tensors['image_id'] \
                and image_id in gaze_on_detect_tensors['image_id']:

            gaze_idx = gaze_tensors['image_id'].index(image_id)
            gaze_feat = gaze_tensors['feat'][gaze_idx]
            # gaze_img_size = gaze_tensors['img_sizes'][gaze_idx]

            gaze_det_idx = gaze_on_detect_tensors['image_id'].index(image_id)
            gaze_det_feat = gaze_on_detect_tensors['feat'][gaze_det_idx]
            # gaze_det_img_size = gaze_on_detect_tensors['img_sizes'][gaze_idx]

            if gaze_feat.size(0) < n_obj:
                continue
            if gaze_det_feat.size(0) < n_obj:
                continue
            num_det_boxes.append(det_feat.size(0))
            num_gaze_boxes.append(gaze_feat.size(0))
            num_gaze_det_boxes.append(gaze_det_feat.size(0))
            det_feat = det_feat[:n_obj]
            gaze_feat = gaze_feat[:n_obj]
            gaze_det_feat = gaze_det_feat[:n_obj]

            item = {}
            # sorted_feat, indices = torch.sort(det_feat, -2) # no need sort, done by NMS
            merged_feat = torch.cat((det_feat[:, :-6], gaze_feat[:, :-6], gaze_det_feat[:, :-4]), dim=0)
            merged_box = torch.cat((det_feat[:, -6:-2], gaze_feat[:, -6:-2], gaze_det_feat[:, -4:]), dim=0)
            # sorted_feat = sorted_feat[:n_obj]  # select top 10 conf feat
            item['num_boxes'] = len(merged_box)
            img = Image.open(image_id)
            item['image_id'] = image_id.split('/')[-1].strip('.jpg')
            # image_ids.append(image_id)
            item['boxes'] = merged_box.cpu().numpy()
            item['feat'] = merged_feat.cpu().numpy()
            item['image_w'], item['image_h'] = img.width, img.height

            # append to zarr files
            boxes.create_dataset(item['image_id'], data=item['boxes'])
            features.create_dataset(item['image_id'], data=item['feat'])
            # image_size dict
            image_size[item['image_id']] = {
                # 'image_h': item['image_h'],  # was
                # 'image_w': item['image_w'],
                'image_h': img_sizes[0],
                'image_w': img_sizes[1],
            }
    print(f'len(num_det_boxes): {len(num_det_boxes)}')
    print(f'len(num_gaze_boxes): {len(num_gaze_boxes)}')
    print(f'len(num_gaze_det_boxes): {len(num_gaze_det_boxes)}')
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
    image_sizes.to_csv(f'mimic_{task}_image_size.csv')


def all_qa_pairs(dataset_path, text):
    valid_qa_pairs = []
    with open(os.path.join(dataset_path, text), 'r') as f:
        for row in csv.reader(f, delimiter='|'):
            valid_qa_pairs.append(row)
    return valid_qa_pairs


def append_valid_qa_pairs(dataset_path, image_ids, text):
    valid_qa_pairs = []
    with open(os.path.join(dataset_path, text), 'r') as f:
        for row in csv.reader(f, delimiter='|'):
            if row[0] + '.jpg' in image_ids:
                valid_qa_pairs.append(row)
    return valid_qa_pairs


def combine_qa(task):
    filename = 'mimic_all_qa_pairs.csv'
    df = pd.read_csv(filename)
    train_df = df.iloc[:10000]
    test_df = df.iloc[10000:13000]
    # Combine questions and answers in the same json file
    data = []
    if task == 'train':
        data = combine_qa_dict(data, train_df)
    else:
        data = combine_qa_dict(data, test_df)

    json.dump(data, open(f'vqa_mimic_{task}_combined.json', 'w'))


def combine_qa_dict(data, df):
    for i, row in df.iterrows():
        # load questions info
        row_dict = {'question': row['question'], 'question_id': i,
                    'image_id': row['dicom_id']}

        # load answers
        answers = row['answer'].split(';')
        row_dict['answers'] = collections.Counter(answers).most_common()
        # row_dict['answers'] is list of dict(ans: count)
        data.append(row_dict)
    return data


def tokenize_questions(task):
    qa = json.load(open(f'vqa_mimic_{task}_combined.json'))
    qas = len(qa)
    for i, row in enumerate(tqdm(qa)):
        row['question_toked'] = [t.text if '?' not in t.text else t.text[:-1]
                                 for t in tokenizer(row['question'].lower())]
        # get spacey tokens and remove question marks
        if i == qas - 1:
            json.dump(qa, open(f'vqa_mimic_{task}_toked.json', 'w'))


def process_questions(q, task):
    # build question dictionary
    def build_vocab(questions):
        count_thr = 0
        # count up the number of times a word is used
        counts = {}
        for row in questions:
            for word in row['question_toked']:
                counts[word] = counts.get(word, 0) + 1
        cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
        print('top words and their counts:')
        print('\n'.join(map(str, cw[:10])))

        # print some stats
        total_words = sum(counts.values())
        print('total words:', total_words)
        bad_words = [w for w, n in counts.items() if n <= count_thr]
        vocab = [w for w, n in counts.items() if n > count_thr]
        bad_count = sum(counts[w] for w in bad_words)
        print('number of bad words: %d/%d = %.2f%%' %
              (len(bad_words), len(counts),
               len(bad_words) * 100.0 / len(counts)))
        print('number of words in vocab would be %d' % (len(vocab),))
        print('number of UNKs: %d/%d = %.2f%%' %
              (bad_count, total_words, bad_count * 100.0 / total_words))

        return vocab

    vocab = build_vocab(q)
    # a 1-indexed vocab translation table
    itow = {i + 1: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table
    pickle.dump({'itow': itow, 'wtoi': wtoi}, open(f'mimic_q_{task}_dict.p', 'wb'))


def process_answers(q, task):
    counts = {}
    for row in q:
        for ans, c in row['answers']:
            counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)

    vocab = [w for c, w in cw]

    # a 0-indexed vocabulary translation table
    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}  # inverse table
    pickle.dump({'itow': itow, 'wtoi': wtoi}, open(f'mimic_a_{task}_dict.p', 'wb'))

    for row in q:
        accepted_answers = 0
        answers_scores = []
        for w, c in row['answers']:
            if w in vocab:
                accepted_answers += c
                answers_scores.append((w, c / accepted_answers))

        row['answers_w_scores'] = answers_scores

    json.dump(q, open(f'vqa_mimic_{task}_final.json', 'w'))


def count_labels():
    df = pd.read_csv('mimic_qa_pairs.csv')
    values, counts = np.unique(df['yes'].values, return_counts=True)
    reverse_indices = counts.argsort()[::-1]
    sorted_values = values[reverse_indices]
    sorted_counts = counts[reverse_indices]
    new_qa_pairs = np.vstack((sorted_values, sorted_counts)).T
    print(new_qa_pairs)


def select_mimic_qa_pairs():
    """
    select qa pairs from visual feature image_ids， no need
    """
    detect_file = 'mimic_detect_feat_path.pt'
    detect_tensors = torch.load(detect_file)
    df = pd.read_csv('mimic_qa_pairs.csv')
    image_ids = [path.split('/')[-1].strip('.jpg') for path in detect_tensors['image_id']]
    for i, row in tqdm(df.iterrows()):
        if row['dicom_id'] not in image_ids:
            df = df.drop(i)

    df.to_csv('selected_mimic_qa_pairs.csv')


def main():
    task = 'train'
    # parse_box_feat(task)

    combine_qa(task)
    tokenize_questions(task)
    t = json.load(open(f'vqa_mimic_{task}_toked.json'))
    process_questions(t, task)
    process_answers(t, task)

    # count_labels()


if __name__ == '__main__':
    main()
