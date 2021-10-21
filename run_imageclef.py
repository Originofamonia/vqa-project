#    Copyright 2018 AimBrain Ltd.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import json
import csv
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import MultiStepLR

from sparse_graph_model import Model
from torch_dataset import *
from utils import *


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train(args, f):
    """
        Train a VQA model using the training set
    """

    # set random seed
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, script requires cuda')

    # Load the VQA training set
    print('Loading data')
    dataset = ImageclefDataset(args)
    train_sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.bsize, sampler=train_sampler,
                        num_workers=4, collate_fn=collate_fn)

    # Load the VQA validation set
    dataset_test = ImageclefDataset(args, train=False)
    test_sampler = SequentialSampler(dataset_test)
    loader_test = DataLoader(dataset_test, batch_size=args.bsize,
                             sampler=test_sampler, shuffle=False,
                             num_workers=4, collate_fn=collate_fn)

    n_batches = len(dataset) // args.bsize

    # Print data and model parameters
    print('Parameters:\n\t'
          'vocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d'
          '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words, args.emb,
                                                    dataset.feat_dim,
                                                    args.hid,
                                                    dataset.n_answers))
    print('Initializing model')

    model = Model(vocab_size=dataset.q_words,
                  emb_dim=args.emb,
                  feat_dim=dataset.feat_dim,
                  hid_dim=args.hid,
                  out_dim=dataset.n_answers,
                  dropout=args.dropout,
                  neighbourhood_size=args.neighbourhood_size,
                  n_kernels=args.n_kernels,
                  pretrained_wemb=dataset.pretrained_wemb,
                  k=args.k)

    criterion = nn.MultiLabelSoftMarginLoss()

    # Move it to GPU
    model = model.cuda()
    criterion = criterion.cuda()
    # parallel models
    # if torch.cuda.device_count() > 1:
    #     print('Let\'s use {} GPUs!'.format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model)

    # Define the optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Continue training from saved model
    start_ep = 0
    if args.model_path and os.path.isfile(args.model_path):
        print('Resuming from checkpoint %s' % args.model_path)
        ckpt = torch.load(args.model_path)
        start_ep = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # Update the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

    # Learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=[30], gamma=0.5)
    scheduler.last_epoch = start_ep - 1
    # epoch_loss = 0
    # epoch_acc = 0

    # Train iterations
    print('Start training.')
    for ep in range(start_ep, start_ep + args.ep):

        scheduler.step()
        ep_loss = 0.0
        ep_correct = 0.0
        ave_loss = 0.0
        ave_correct = 0.0
        losses = []
        pbar = tqdm(enumerate(loader))
        for step, batch in pbar:
            model.train()
            # Move batch to cuda
            q_batch, a_batch, vote_batch, i_batch, k_batch, qlen_batch = \
                batch_to_cuda(batch)

            # forward pass
            output, adjacency_matrix = model(
                q_batch, i_batch, k_batch, qlen_batch)

            loss = criterion(output, a_batch)

            # Compute batch accuracy based on vqa evaluation
            correct = total_vqa_score(output, vote_batch)
            ep_correct += correct
            ep_loss += loss.item()
            ave_correct += correct
            ave_loss += loss.item()
            losses.append(loss.item())

            # This is a 40 step average
            if step % 40 == 0 and step != 0:
                print(
                    'Epoch %02d(%03d/%03d), ave loss: %.7f, ave accuracy: '
                    '%.2f%%' % (ep + 1, step, n_batches, ave_loss / 40,
                                ave_correct * 100 / (args.bsize * 40)))

                ave_correct = 0
                ave_loss = 0

            # Compute gradient and do optimisation step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save model and compute validation accuracy every 400 steps
            if step % 400 == 0:
                epoch_loss = ep_loss / n_batches
                epoch_acc = ep_correct * 100 / (n_batches * args.bsize)
                desc = f'epoch_loss: {epoch_loss:.3f}; epoch_acc: {epoch_acc:.3f}'
                pbar.set_description(desc)
                # save(model, optimizer, ep, epoch_loss, epoch_acc,
                #      dir=args.save_dir, name=args.name + '_' + str(ep + 1))

                # compute validation accuracy over a small subset of the
                # validation set

    print('Infer')
    test_correct = 0
    model.eval()

    results = []
    for i, test_batch in tqdm(enumerate(loader_test)):
        q_batch, a_batch, vote_batch, i_batch, k_batch, qlen_batch = \
            batch_to_cuda(test_batch)
        output, _ = model(q_batch, i_batch, k_batch, qlen_batch)
        # print(output.size())
        test_correct += total_vqa_score(output, vote_batch)
        qid_batch = test_batch[3]
        _, oix = output.data.max(1)
        oix = oix.cpu().numpy()
        # print(f'oix: {oix}')
        # record predictions
        for i, qid in enumerate(qid_batch):
            qid = int(qid.cpu().numpy())
            results.append(
                f"{dataset_test.vqa[qid]['image_id']},"
                f"{dataset_test.vqa[qid]['question']},"
                f"{dataset_test.a_itow[oix[i]]},"
                f"{dataset_test.vqa[qid]['answer']}")

    # with open('infer_imageclef.csv', 'w') as f:
    #     f.write('image_id,question,prediction,answer\n')
    #     for line in results:
    #         f.write(line)
    #         f.write('\n')

    acc = test_correct / (10 * args.bsize) * 100
    print(f"neighbors: {args.neighbourhood_size}, kernels: {args.n_kernels}, Validation acc: {acc:.2f} %\n")
    f.write(f"neighbors: {args.neighbourhood_size}, kernels: {args.n_kernels}, Validation acc: {acc:.2f} %\n")

    # save model and compute accuracy for epoch
    epoch_loss = ep_loss / n_batches
    epoch_acc = ep_correct * 100 / (n_batches * args.bsize)

    save(model, optimizer, args.ep, epoch_loss, epoch_acc,
         dir=args.save_dir, name=args.name + '_' + str(args.ep + 1))

    print(
        'Epoch %02d done, average loss: %.3f, average accuracy: %.2f%%' % (
            args.ep + 1, epoch_loss, epoch_acc))


def save_plot_nodes(args, f):


def main():
    args, parser, unparsed = input_args()
    neighbors_list = [12, 16, 20, 24, 28, 32, 36]
    kernels_list = [2, 4, 8, 16, 32]  # can't be larger than n_obj
    with open(f'grid_search_nodes_{args.k}.txt', 'w') as f:
        for neighbors in neighbors_list:
            for kernels in kernels_list:
                args.n_kernels = kernels
                args.neighbourhood_size = neighbors
                print(args)
                if len(unparsed) != 0:
                    raise SystemExit('Unknown argument: {}'.format(unparsed))
                if args.train:
                    train(args, f)

                if not args.train and not args.eval and not args.trainval and not args.test:
                    parser.print_help()


def input_args():
    parser = argparse.ArgumentParser(
        description='Conditional Graph Convolutions for VQA')
    parser.add_argument('--train', default=True, type=bool,
                        help='set this to training mode.')
    # parser.add_argument('--trainval', action='store_true',
    #                     help='set this to train+val mode.')
    # parser.add_argument('--eval', action='store_true',
    #                     help='set this to evaluation mode.')
    # parser.add_argument('--test', action='store_true',
    #                     help='set this to test mode.')
    # args.n_kernels
    parser.add_argument('--n_kernels', type=int, default=8,
                        help='number of epochs.')
    parser.add_argument('--lr', metavar='', type=float,
                        default=1e-3, help='initial learning rate')  # was 1e-4
    parser.add_argument('--ep', metavar='', type=int,
                        default=40, help='number of epochs.')
    parser.add_argument('--bsize', type=int, default=8, help='batch size.')
    parser.add_argument('--hid', metavar='', type=int,
                        default=1024, help='hidden dimension')
    parser.add_argument('--emb', metavar='', type=int, default=300,
                        help='question embedding dimension')
    parser.add_argument('--neighbourhood_size', type=int, default=19,
                        help='topm number of graph neighbours to consider')
    parser.add_argument('--k', type=int, default=48,
                        help='number of boxes per image')
    parser.add_argument('--data_dir', metavar='', type=str, default='./data',
                        help='path to data directory')
    parser.add_argument('--save_dir', metavar='', type=str, default='./save')
    parser.add_argument('--name', metavar='', type=str,
                        default='model', help='model name')
    parser.add_argument('--dropout', metavar='', type=float, default=0.4,
                        help='probability of dropping out FC nodes during '
                             'training')
    parser.add_argument('--model_path', metavar='', type=str,
                        help='trained model path.')
    args, unparsed = parser.parse_known_args()
    return args, parser, unparsed


if __name__ == '__main__':
    main()
