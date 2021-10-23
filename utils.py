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

import re
import os
import torch
import numpy as np
from torch.autograd import Variable


def batch_to_cuda(batch):
    # moves dataset batch on GPU

    q = Variable(batch[0], requires_grad=False).cuda()
    a = Variable(batch[1], requires_grad=False).cuda()
    n_votes = Variable(batch[2], requires_grad=False).cuda()
    i = Variable(batch[4], requires_grad=False).cuda()
    k = Variable(batch[5], requires_grad=False).cuda()
    qlen = list(batch[6])
    return q, a, n_votes, i, k, qlen


def save(args, model, path, name):
    # saves model and optimizer state

    # tbs = {
    #     'epoch': ep + 1,
    #     'loss': epoch_loss,
    #     'accuracy': epoch_acc,
    #     'state_dict': model.state_dict(),
    #     'optimizer': optimizer.state_dict()
    # }
    torch.save(model.state_dict(), os.path.join(path, name))


def total_vqa_score(output_batch, n_votes_batch):
    # computes the total vqa score as assessed by the challenge

    vqa_score = 0
    _, oix = output_batch.data.max(1)
    for i, pred in enumerate(oix):
        count = n_votes_batch[i, pred]
        vqa_score += min(count.item() / 3, 1)
    return vqa_score


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)

