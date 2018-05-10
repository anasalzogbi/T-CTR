"""
Author: Anas Alzogbi
Description: 
This module provides the functionality of:
 - Reading and writing ratings file 
 - Reading and writing docs-topics distribution file
 - 
Date: October 27th, 2017
alzoghba@informatik.uni-freiburg.de
"""

import csv
import os
import numpy as np
from enum import Enum


class FileFormat(Enum):
    """
    This Enum encodes the two csv files format.
    """
    # No items ids are stored in the file, the ids match line numbers:
    line_num_based = 1

    # First value of each line is the item id, ids don't match line numbers:
    id_based = 2

def read_mappings(mappings_file, delimiter = ",", header= True):
    """
    Reads mappings between two valus from file that has two columns.
    Returns dictionary, the key is the first column, the value is the second column
    """
    dic = {}
    with open(mappings_file) as f:
        if header:
            # Skip the first line, the header:
            f.readline()
        for line in f:
            id1, id2 = line.split(delimiter)
            dic[id1.strip()] = int(id2)
    return dic


def read_ratings(ratings_file, users_num, items_num):
    """
    Reads the ratings file into a numpy 2d array
    :param ratings_file: the path of the ratings file, the file is expected to be formated as following: 
    The first entry in line i is the ratings count (n) of user (i), the rest n-length space separated list contains the
    item ids rated by user i. 
    :param users_num: number of users
    :param items_num: number of items
    :return: numpy 2d array, the ith row contains a list of relevant items ids to user i
    """
    ratings_mat = np.zeros((users_num, items_num))
    with open(ratings_file) as f:
        user_id = 0
        for line in f:
            ratings = [int(x) for x in line.replace("\n", "").split(" ")[1:] if x != ""]
            for r in ratings:
                ratings_mat[user_id, r] = 1
            user_id += 1
    return ratings_mat


def convert(value, type_):
    """
    Casts a value into type given in a string (type_)
    """
    import importlib
    try:
        # Check if it's a builtin type
        module = importlib.import_module('__builtin__')
        cls = getattr(module, type_)
    except AttributeError:
        # if not, separate module and class
        module, type_ = type_.rsplit(".", 1)
        module = importlib.import_module(module)
        cls = getattr(module, type_)
    return cls(value)

def read_ratings_as_list(ratings_file, type_=int):
    """
    Reads the ratings file into a list of lists
    :param ratings_file: the path of the ratings file, the file is expected to be formated as following: 
    The first entry in line i is the ratings count (n) of user (i), the rest n-length space separated list contains the
    item ids rated by user i. 
    :param users_num: number of users
    :param items_num: number of items
    :return: 2d list, the ith row contains a list of relevant items ids to user i
    """
    ratings_list = []
    with open(ratings_file) as f:
        for line in f:
            ratings = [type_(x) for x in line.replace("\n", "").split(" ")[1:] if x != ""]
            ratings_list.append(ratings)
    return ratings_list


def write_ratings(ratings_list, filename, delimiter=" ", print_line_length=True):
    """
    writes user matrix to a file, the file will be formated as following: line i has delimiter-separated list of item ids rated by user i
    :param ratings_list: users 2d list, row num = num_users
    :param filename: the path of the users file
    :param delimiter: default: space
    :param print_line_length: if True: the first column of each line will record the line's length
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        for ratings in ratings_list:
            if print_line_length:
                writer.writerow([len(ratings)] + ratings)
            else:
                writer.writerow(ratings)


def read_docs_topics_file(papers_file, delimiter=" "):
    """
    Reads documents latent topics from the file
    :param papers_file: file contains papers topics
    :param delimiter: The separator used in the file
    :return: ndarray (2d), one row for each paper, one column for each topic
    """
    papers_list = []
    with open(papers_file) as f:
        for line in f:
            paper_dis = [float(x) for x in line.replace("\n", "").split(delimiter) if x != ""]
            papers_list.append(paper_dis)
    # Check if all lines have the same length
    if len(set([len(x) for x in papers_list])) > 1:
        print("Error in papers file, papers have different number of features: {}".format(papers_file))
        raise ValueError("Error in papers file, papers have different number of features.")
    return np.array(papers_list)


def print_list(lst):
    print('[%s]' % ', '.join(map(str, lst)))

def read_docs_vocabs_file_as_array(doc_vocab_file, with_header=False, lda_c_format=True, delimiter=' '):
    start = 0
    # lda-c format: first value of each row is the #vocab
    if lda_c_format:
        start = 1
    doc_vocab = []
    vocabs = set()
    with open(doc_vocab_file, 'r') as f:
        line_num = 0
        if with_header:
            next(f)
        for line in f:
            doc_vector = []
            line_split = line.strip().split(delimiter)
            for e in line_split[start:]:
                vocab, freq = e.split(":")
                try:
                    vocab = int(vocab)
                    doc_vector.append((vocab, int(freq)))
                    vocabs.add(vocab)
                except ValueError:
                    print("Error in doc_vocab file {} : line {}, value {} is not int.".format(doc_vocab_file, line_num,
                                                                                              e))
                    raise
            doc_vocab.append(doc_vector)
            line_num += 1
    if (len(vocabs) != max(vocabs) + 1):
        print("Error in doc_vocab file {}: # Vocabs = {}, max vocab = {}".format(doc_vocab_file, len(vocabs),
                                                                                 max(vocabs)))
        raise ValueError
    
    mat = np.zeros( (len(doc_vocab),  len(vocabs)) )
    doc_idx = 0
    for d in doc_vocab:
        for e in d:
            mat[doc_idx, e[0]] = e[1]
            if e[1] <= 0:
                print(e[1])
        doc_idx += 1
    return mat

def read_docs_vocabs_file(doc_vocab_file, with_header=False, lda_c_format=True, delimiter=' '):
    start = 0
    # lda-c format: first value of each row is the #vocab
    if lda_c_format:
        start = 1
    doc_vocab = []
    vocabs = set()
    with open(doc_vocab_file, 'r') as f:
        line_num = 0
        if with_header:
            next(f)
        for line in f:
            doc_vector = []
            line_split = line.strip().split(delimiter)
            for e in line_split[start:]:
                vocab, freq = e.split(":")
                try:
                    vocab = int(vocab)
                    doc_vector.append((vocab, int(freq)))
                    vocabs.add(vocab)
                except ValueError:
                    print("Error in doc_vocab file {} : line {}, value {} is not int.".format(doc_vocab_file, line_num,e))
                    raise
            doc_vocab.append(doc_vector)
            line_num += 1
    if (len(vocabs) != max(vocabs) + 1):
        print("Error in doc_vocab file {}: # Vocabs = {}, max vocab = {}".format(doc_vocab_file, len(vocabs),
                                                                                 max(vocabs)))
        raise ValueError
    return doc_vocab



# Writing distributions to the file
def write_distribution(distribution, file_out, delimiter=" "):
    with open(file_out, 'w') as f:
        for row in distribution:
            f.write(delimiter.join([str(j) for j in row]) + "\n")
