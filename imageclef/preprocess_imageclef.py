"""
read saved box and feature pt file, parse to VQA dataset format
"""
import os
import csv
import json
import numpy as np
import pandas as pd
import zarr
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


def parse_box_feat():
    # fieldnames = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes',
    #               'features']
    n_obj = 17  # n_obj per image
    detect_file = 'detect_feat_path.pt'
    gaze_file = 'gaze_feat_path.pt'
    gaze_on_detect_file = 'gaze_on_detect_feat_path.pt'
    imgpath = '/home/qiyuan/2021summer/imageclef/images/'
    detect_tensors = torch.load(detect_file)
    gaze_tensors = torch.load(gaze_file)
    gaze_on_detect_tensors = torch.load(gaze_on_detect_file)
    # {'feat': selected_feats, 'image_id': filepaths}

    boxes = zarr.open_group('imageclef_boxes.zarr', mode='w')
    features = zarr.open_group('imageclef_features.zarr', mode='w')
    image_size = {}
    num_det_boxes = []
    num_gaze_boxes = []
    num_gaze_det_boxes = []
    image_ids = []
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
            image_id = image_id.split('/')[-1]
            img = Image.open(os.path.join(imgpath, image_id))
            item['image_id'] = image_id
            image_ids.append(image_id)
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
    image_sizes.to_csv('imageclef_image_size.csv')

    dataset_path = '/home/qiyuan/2021summer/imageclef'
    text0 = 'VQAnswering_2020_Train_QA_pairs.txt'
    text1 = 'VQAnswering_2020_Val_QA_Pairs.txt'
    text2 = 'VQA-Med-2021-VQAnswering-Task1-New-ValidationSet.txt'

    # qa_pairs0 = all_qa_pairs(dataset_path, text0)
    # qa_pairs1 = all_qa_pairs(dataset_path, text1)
    # qa_pairs2 = all_qa_pairs(dataset_path, text2)
    # qa_pairs0.extend(qa_pairs1)
    # qa_pairs0.extend(qa_pairs2)

    valid_qa_pairs0 = append_valid_qa_pairs(dataset_path, image_ids, text0)
    valid_qa_pairs1 = append_valid_qa_pairs(dataset_path, image_ids, text1)
    valid_qa_pairs2 = append_valid_qa_pairs(dataset_path, image_ids, text2)

    valid_qa_pairs0.extend(valid_qa_pairs1)
    valid_qa_pairs0.extend(valid_qa_pairs2)
    with open('../data/imageclef_qa_pairs.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(valid_qa_pairs0)


def get_qa_pairs():
    """
    merged to parse_box_feat func, obsolete.
    Get filtered QA pairs and save to a new txt file. Only need once.
    """
    filename = 'feat_path_yolo.pt'
    tensors = torch.load(filename)
    image_ids = tensors['image_id']


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


def process_text():
    filename = '../data/imageclef_qa_pairs.csv'
    # Combine questions and answers in the same json file
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, q in enumerate(tqdm(reader)):
            row = {}
            # load questions info
            row['question'] = q[1]
            row['question_id'] = i
            row['image_id'] = q[0]

            # load answers
            row['answer'] = q[2]
            row['answers'] = {q[2]: 10}

            data.append(row)

    json.dump(data, open('vqa_imageclef_combined.json', 'w'))


def tokenize_questions():
    qa = json.load(open('vqa_imageclef_combined.json'))
    qas = len(qa)
    for i, row in enumerate(tqdm(qa)):
        row['question_toked'] = [t.text if '?' not in t.text else t.text[:-1]
                                 for t in tokenizer(row['question'].lower())]
        # get spacey tokens and remove question marks
        if i == qas - 1:
            json.dump(qa, open('vqa_imageclef_toked.json', 'w'))


def process_questions(q):
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
    pickle.dump({'itow': itow, 'wtoi': wtoi}, open('imageclef_q_dict.p', 'wb'))


def process_answers(q):
    counts = {}
    for row in q:
        counts[row['answer']] = counts.get(row['answer'], 0) + 1

    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)

    vocab = [w for c, w in cw]

    # a 0-indexed vocabulary translation table
    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}  # inverse table
    pickle.dump({'itow': itow, 'wtoi': wtoi}, open('imageclef_a_dict.p', 'wb'))

    for row in q:
        accepted_answers = 0
        answers_scores = []
        for w, c in row['answers'].items():
            if w in vocab:
                accepted_answers += c
                answers_scores.append((w, c / accepted_answers))
        # for w, c in row['answers'].items():
        #     if w in vocab:
        #         answers_scores.append((w, c / accepted_answers))

        row['answers_w_scores'] = answers_scores

    json.dump(q, open('vqa_imageclef_final.json', 'w'))


def count_labels():
    # dataset_path = '/home/qiyuan/2021summer/imageclef'
    # text0 = 'VQAnswering_2020_Train_QA_pairs.txt'
    # text1 = 'VQAnswering_2020_Val_QA_Pairs.txt'
    # text2 = 'VQA-Med-2021-VQAnswering-Task1-New-ValidationSet.txt'
    #
    # qa_pairs0 = all_qa_pairs(dataset_path, text0)
    # qa_pairs1 = all_qa_pairs(dataset_path, text1)
    # qa_pairs2 = all_qa_pairs(dataset_path, text2)
    # qa_pairs0.extend(qa_pairs1)
    # qa_pairs0.extend(qa_pairs2)
    #
    # with open('imageclef_all_qa_pairs.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['image_id', 'question', 'answer'])
    #     writer.writerows(qa_pairs0)

    df = pd.read_csv('../data/imageclef_qa_pairs.csv')
    values, counts = np.unique(df['yes'].values, return_counts=True)
    reverse_indices = counts.argsort()[::-1]
    sorted_values = values[reverse_indices]
    sorted_counts = counts[reverse_indices]
    new_qa_pairs = np.vstack((sorted_values, sorted_counts)).T
    print(new_qa_pairs)


if __name__ == '__main__':
    parse_box_feat()  # run once
    process_text()
    tokenize_questions()
    t = json.load(open('vqa_imageclef_toked.json'))
    process_questions(t)
    process_answers(t)
    # count_labels()
