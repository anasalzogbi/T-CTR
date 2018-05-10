"""
Author: Anas Alzogbi
Description: 
This module provides the functionality of:
 - Generating LDA topics for a set of documents using gensim library
 - Saving the documents-topic matrix (theta) and the topics-vocab matrix (beta) to files.
 - Exploring the generated topics
Date: November 06th, 2017
alzoghba@informatik.uni-freiburg.de
"""
import sys
import os
import argparse
import time
import multiprocessing
import numpy as np
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.files_utils import read_docs_vocabs_file, write_distribution



def readVocabs(vocab_file):
    """
    # Reads the vocabulary file into dictionary
    :param vocab_file: file path
    :return: dictionary(vocab_id, vocabulary)
    """
    vocabs = {}
    idx = 0
    with open(vocab_file, 'r') as f:
        for line in f:
            vocabs[idx] = line.strip()
            idx += 1
    return vocabs


# print_lda_topics(lda.components_, vocab_dict)
def print_lda_topics(topics, vocabs_dict, n_top_words=10):
    
    """
    This method prints the top n words of each document for a list of latent topics.
    # source: https://stackoverflow.com/questions/44208501/getting-topic-word-distribution-from-lda-in-scikit-learn
    :param topics: 2d array: topics_count x vocabs_count
    :param vocabs_dict: vocabulary dictionary( vocab_id, vocabulary)
    :param n_top_words: The desired number of the topic' most important word. 
    """
    topic_words = {}    
    for topic, comp in enumerate(topics):
        # for the n-dimensional array "arr":
        # argsort() returns a ranked n-dimensional array of arr, call it "ranked_array"
        # which contains the indices that would sort arr in a descending fashion
        # for the ith element in ranked_array, ranked_array[i] represents the index of the
        # element in arr that should be at the ith index in ranked_array
        # ex. arr = [3,7,1,0,3,6]
        # np.argsort(arr) -> [3, 2, 0, 4, 5, 1]
        # word_idx contains the indices in "topic" of the top num_top_words most relevant
        # to a given topic ... it is sorted ascending to begin with and then reversed (desc. now)    
        word_idx = np.argsort(comp)[::-1][:n_top_words]

        # store the words most relevant to the topic
        topic_words[topic] = [vocabs_dict[i] for i in word_idx]
    for topic, words in topic_words.items():
        print('Topic: %d' % topic)
        print('  %s' % ', '.join(words))


# main code;
if __name__ == '__main__':
    docs_file = '../../citeulike-crawled/converted_data/2k_1_P3/terms_keywords_based/2k_100_P3_reduced/mult.dat'
    output_directory = '../../citeulike-crawled/converted_data/2k_1_P3/terms_keywords_based/2k_100_P3_reduced/lda_topics'
    vocab_file = '../../citeulike-crawled/converted_data/2k_1_P3/terms_keywords_based/2k_100_P3_reduced/terms.dat'
    k_list = [50, 100, 150, 200, 250]
    cores = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", help="The output directory")
    parser.add_argument("--docs_file", "-f", help="The documents bag of words file (mult.dat)")
    parser.add_argument("--vocabs_file", "-v", help="The vocabulary file")
    parser.add_argument("--k_list", "-k", help="The list of number of topics desired.", type=int, nargs='+')
    parser.add_argument("--cores", "-c", type=int, help="The number of cores to be used, default: single core")

    args = parser.parse_args()
    multicore = False

    # Checking and setting the arguments:
    if args.output_dir:
        output_directory = args.output_dir
    if args.docs_file:
        docs_file = args.docs_file

    if not os.path.exists(docs_file):
        print("Error, document file not exists: {}".format(docs_file))

    if args.vocabs_file:
        vocab_file = args.vocabs_file

    if not os.path.exists(vocab_file):
        print("Error, vocabulary file not exists: {}".format(vocab_file))

    if args.k_list:
        k_list = args.k_list
    if args.cores:
        cores = args.cores

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Reading the documents bag of words file:
    documents = read_docs_vocabs_file(docs_file)

    print("Reading vocabularies...")
    vocabs_dict = readVocabs(vocab_file)

    # Computing LDA
    for k in k_list:
        t1 = time.time()
        if cores > 1:
            print("Building Lda model k = {}, multicores...".format(k))
            lda = LdaMulticore(corpus=documents, num_topics=k, id2word=vocabs_dict, workers=min(cores,multiprocessing.cpu_count()), alpha=1e-5, eta=5e-1, minimum_probability=0.0)
        else:
            print("Building Lda model k = {} ...".format(k))
            lda = LdaModel(corpus=documents, num_topics=k, minimum_probability=0.0)

        # theta:
        print("Extracting theta...")
        document_distribution = []
        for doc in documents:
            document_distribution.append(lda[doc])
            #l = sorted(lda[doc], key=lambda x: x[0])
            #document_distribution.append([i for _, i in l])

        print("Extracting beta...")
        # Beta: # topics x # terms np array
        beta = lda.get_topics()

        t2 = time.time()
        print("Done k = {} in {} mins, saving theta and beta to files ...".format(k, (t2 - t1) / 60))

        # Writing distributions to the file
        write_distribution(document_distribution, os.path.join(output_directory, "theta_{}.dat".format(k)))
        write_distribution(beta, os.path.join(output_directory, "beta_{}.dat".format(k)))