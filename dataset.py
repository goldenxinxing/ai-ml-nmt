from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string

import random

import torch
import torch.nn as nn
from torch import optim

from helper import MAX_LENGTH, filterComment, normalizeString


def prepareData(path):
    # Read the file and split into lines
    lines = open(path, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t') if not filterComment(s)] for l in lines]

    return pairs


if __name__ == "__main__":
    path = 'data/%s-%s.txt' % ('eng', 'fra')
    pairs = prepareData(path)
    print(random.choice(pairs))
