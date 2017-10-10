#! /usr/bin/env python
import os
import argparse
import torch
import torchtext.data as data
from loaddata import mydatasets_self_two
from loaddata import loadingdata_Twitter
from loaddata import loadingdata_CV
from loaddata.handle_wordEmbedding2File import WordEmbedding2File
import random
import hyperparams
# solve encoding
from imp import reload
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)

parser = argparse.ArgumentParser(description="convert word2vec based on the vocab that created ")

# data that create vocab
parser.add_argument('-datafile_path', type=str, default=hyperparams.datafile_path, help='dataset file path')
# word embedding path
parser.add_argument('-need_convert_dim', type=int, default=hyperparams.need_convert_dim, help='need_convert_dim')
parser.add_argument('-need_convert_path', type=str, default=hyperparams.need_convert_path, help='need_convert_path')
parser.add_argument('-save_converted_word_Embedding_Path', type=str,
                    default=hyperparams.save_converted_word_Embedding_Path, help='save_converted_word_Embedding_Path')
# nums of threads
parser.add_argument('-num_threads', type=int, default=hyperparams.num_threads, help='the num of threads')
# option
args = parser.parse_args()


def convert(text_field, label_field, path):
    # train_data, dev_data, test_data = loadingdata_Twitter.Twitter.splits(path, text_field, label_field)
    # print("len(train_data) {} ".format(len(train_data)))
    # print("create the vocab......")
    # text_field.build_vocab(train_data, dev_data, test_data)
    # label_field.build_vocab(train_data, dev_data, test_data)

    train_data = loadingdata_CV.CV.splits(path, text_field, label_field)
    print("len(train_data) {} ".format(len(train_data)))
    print("create the vocab......")
    text_field.build_vocab(train_data)
    label_field.build_vocab(train_data)


# load data
text_field = data.Field(lower=True)
# text_field = data.Field(lower=False)
label_field = data.Field(sequential=False)
print("\nLoading data...")
convert(text_field, label_field, path=args.datafile_path)

# handle external word embedding to file for convenience
wordembedding = WordEmbedding2File(wordEmbedding_path=args.need_convert_path,
                                   save_wordEmbedding_path=args.save_converted_word_Embedding_Path,
                                   vocab=text_field.vocab.itos,
                                   k_dim=args.need_convert_dim)
wordembedding.handle()
