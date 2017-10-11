## Introduction

**- Convert big wordEmbedding file based on the vocab that create by dataset** 

## Requirement
* python 3
* pytorch > 0.1
* torchtext > 0.1
* numpy

## How to use the folder or file

- the file of **hyperparams.py** contains all hyperparams that need to modify, based on yours nedds, select neural networks what you want and config the hyperparams.

- the file of **main-hyperparams.py** is the main function,run the command ("python main_hyperparams.py") to execute the demo.

- the folder of **loaddata** contains some file of load dataset

- the folder of **word2vec** is the file of word embedding that you want to use

- the folder of **data** contains the dataset file,contains train data,dev data,test data.

- the file of **handle_wordEmbedding2File.py** to hadle the big word2vec

## How to use the Word Embedding in demo? 

- the word embedding file saved in the folder of **word2vec**, but now is empty, because of it is to big,so if you want to use word embedding,you can to download word2vec or glove file, then saved in the folder of word2vec,and make the option of word_Embedding to True and modifiy the value of word_Embedding_Path in the **hyperparams.py** file.


## How to config hyperparams in the file of hyperparams.py

- **datafile_path**: the path of dataset

- **need_convert_path**: the so big wordEmbedding that want to change based one the dateset 

- **save_converted_word_Embedding_Path**:after convert where to save

- **need_convert_dim**：the dim of big wordEmbedding

- **num_threads**:set the value of threads when run the demo

- **seed_num**:set the num of random seed

## Reference 

- Only for myself to load wordEmbedding quickly compare to the big file like Glogle News.

