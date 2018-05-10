"""
Author: Anas Alzogbi
Description: 
This module provides the functionality of:
 - Generate papers recommendations following the PubRec method, the approach is explained in the paper:
 PubRec: Recommending Publications Based On Publicly Available Meta-Data
Date: November 15th, 2017
alzoghba@informatik.uni-freiburg.de
"""

import argparse
import os
import numpy as np
from util.files_utils import read_docs_topics_file, read_ratings_as_list
from util.utils import convert_to_probability_dist
import scipy 

def interest_extent(ratings_age_array, forgetting_factor):
    """
    computes interest extent of a paper for a user based on the age of the rating
    ratings_age_array - a list of ages of the ratings in years or months
    returns 1d array of float representing the interest of the user to the papers
    """
    return np.exp(-np.power(ratings_age_array, 2) / forgetting_factor)


def cost_function( x):
    """
    function to be minimized by L-BFGS algorithm
    x - matrix of shape 1*k, where k is the # of features
    returns a float, representing deviation the interest extent model
    """
    return 1 / (2 * m) * (np.power(np.dot(x, self.V) - self.IE, 2).sum())


def calculatePubRecPredictions(ratings_list, papers_topics, ratings_timestamps, forgetting_factor):
    for u_ratings in ratings_list:
        u_papers = papers_topics[u_ratings]
        y = interest_extent(ratings_timestamps[u_ratings], forgetting_factor)
        x = np.array(RandomVector(len(self.docTermDict[library[0][0]])))
        res = scipy.optimize.fmin_l_bfgs_b(cost_function, x, approx_grad=1)
        return (res[0], res[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", "-u",
                        help="The base directory that contains the folds, expecting to have sub folders: fold-[1..5], "
                             "each folder has the training users ratings file")
    parser.add_argument("--papers_file", "-p", help="The papers-topics file")
    parser.add_argument("--experiment", "-e", help="The experiment name, used to name the output folder, where the "
                                                   "resulting prediction files (score.npy) will be saved, the output directory "
                                                   "will appear under the base directory")
    parser.add_argument("--users_num", "-n", type=int, help="The number of users in the dataset")
    parser.add_argument("--items_num", "-m", type=int, help="The number of items in the dataset")
    parser.add_argument("--step_lag", "-l", type=int, help="The step lag for reporting progress")
    #parser.add_argument("--is_prob_dist", "-d", action="store_true", default=False, help="A flag idicates if the provided papers vectors form probability distribution or not, default: False")
    args = parser.parse_args()

    # Checking and setting the arguments:
    if args.base_dir:
        base_dir = args.base_dir
        if not os.path.exists(base_dir):
            print("Input directory isnot found: {}".format(base_dir))
            raise NameError("Input directory is not found")

    if args.papers_file:
        papers_file = args.papers_file
        if not os.path.exists(papers_file):
            print("papers file not found: {}".format(papers_file))
            raise NameError("Papers file not found")
        # Reading papers file
        print("Reading papers-topics file: {}".format(papers_file))
        papers_topics = read_docs_topics_file(papers_file)
        print("papers matrix dimensions: {}".format(papers_topics.shape))

    if args.experiment:
        experiment_name = args.experiment

    if args.users_num:
        users_num = args.users_num

    if args.items_num:
        items_num = args.items_num

    if args.step_lag:
        lag = args.step_lag

    for fold in range(1, 6):
        users_rtings_file = os.path.join(base_dir, "fold-" + str(fold), "train-fold_" + str(fold) + "-users.dat")
        print("Reading users ratings file: {}".format(users_rtings_file))
        output_dir = os.path.join(base_dir, experiment_name, "fold-" + str(fold))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Read users ratings
        ratings_list = read_ratings_as_list(users_rtings_file)
        print("Rating matrix dimensions: {}".format(ratings_list.shape))

        # Calculate predictions for all users:
        predictions = calculatePubRecPredictions(ratings_list, papers_topics, ratings_timestamps)

        # Saving the predictions:
        print("Storing predictions to: [{}]".format(os.path.join(output_dir, "score")))
        np.save(os.path.join(output_dir, "score"), predictions)
