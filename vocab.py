from __future__ import print_function
import argparse
from collections import Counter
from itertools import chain

import torch

from dataset import filterComment, normalizeString

class Vocab:
    def __init__(self, input_lang, output_lang):
        self.input_lang = input_lang
        self.output_lang = output_lang

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.data = []

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.data.append(word)
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t') if not filterComment(s)] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def getData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    #pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_vocab_size', default=500000, type=int, help='source vocabulary size')
    parser.add_argument('--tgt_vocab_size', default=500000, type=int, help='target vocabulary size')
    parser.add_argument('--include_singleton', action='store_true', default=False, help='whether to include singleton'
                                                                                        'in the vocabulary (default=False)')

    parser.add_argument('--src_lang', type=str, default='eng', help='file of source sentences')
    parser.add_argument('--tgt_lang', type=str, default='fra', help='file of target sentences')

    parser.add_argument('--train_src', type=str, default='data/train.de-en.en', help='file of source sentences')
    parser.add_argument('--train_tgt', type=str, default='data/train.de-en.de', help='file of target sentences')

    parser.add_argument('--output', default='data/vocab_%s-%s.bin', type=str, help='output vocabulary file')

    args = parser.parse_args()
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang

    input_lang, output_lang, pairs = getData(src_lang, tgt_lang, False)

    vocab = Vocab(input_lang, output_lang)
    print('generated vocabulary, source %d words, target %d words' % (vocab.input_lang.n_words, vocab.output_lang.n_words))

    # save
    torch.save(vocab, args.output % (src_lang, tgt_lang) )
    print('vocabulary saved to %s' % (args.output % (src_lang, tgt_lang)))
