# coding=utf-8

"""
    handle external word embedding to file
"""

import os


class WordEmbedding2File:
    def __init__(self, wordEmbedding_path=None, save_wordEmbedding_path=None, vocab=None, k_dim=100):
        print("handling external word embedding to file based on the vocab that created by data")
        self.wordEmbedding_path = wordEmbedding_path
        self.save_wordEmbedding_path = save_wordEmbedding_path
        self.vocab = vocab
        self.k_dim = k_dim

    def handle(self):
        if os.path.exists(self.save_wordEmbedding_path):
            os.remove(self.save_wordEmbedding_path)
        file = open(self.save_wordEmbedding_path, "w")
        with open(self.wordEmbedding_path, encoding="utf-8") as f:
            count = 0
            lines = f.readlines()
            file.write(str(self.k_dim))
            file.write("\n")
            for line in lines:
                values = line.split(" ")
                word = values[0]
                if word in self.vocab:
                    count += 1
                    print(line)
                    file.writelines(line)
            print("The number of handle is {} ".format(count))
        file.close()
        print("Handle external word embedding has Finished.")



