import torch
import random
torch.manual_seed(121)
random.seed(121)

# datafile_path = "./data/Twitter"
datafile_path = "./data/MR"

need_convert_path = "./word2vec/glove.840B.300d.txt"
need_convert_dim = 300
# word_Embedding_Path = "./word2vec/glove.840B.300d.handledword2vec.txt"
save_converted_word_Embedding_Path = "./converted_word.txt"

num_threads = 1

seed_num = 233


