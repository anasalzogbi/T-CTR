"""
Author: Anas Alzogbi
Description: 
This module provides the functionality of:
 - Computing ratings weights based on the ratings age
Date: December 12th, 2017
alzoghba@informatik.uni-freiburg.de
"""

import argparse
import os
import numpy as np
from util.files_utils import read_ratings_as_list, write_ratings

import pandas as pd

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end
    
    
def interest_extent(ratings_age_array, forgetting_factor):
    """
    computes interest extent of a paper for a user based on the age of the rating
    ratings_age_array - a list of ages of the ratings in years or months
    returns 1d array of float representing the interest of the user to the papers
    """
    if forgetting_factor == 0:
        forgetting_factor = 0.1
    return [i if i > 0.01 else 0.01 for i in np.exp(-np.power(ratings_age_array, 2) / forgetting_factor)]

# Retuns values between 0 and 100, proportional to the similarity. This is to be used to generate factors for interst_extent function
def calculate_users_lambda(ratings_list, papers_topics):
    us_sims = []
    i = 0
    for u in ratings_list:
        sims = []
        for i, _ in enumerate(u):
            if i < len(u) - 1:
                sims.append(jackard_sim(papers_topics[u[i]], papers_topics[u[i + 1]]))
        if len(sims) == 0:
            i += 1
            us_sims.append(1)
        else:
            us_sims.append(np.mean(sims)*100)
    return us_sims

def interest_extent_2(ages_list, factor):
    return (2/(1+ np.exp(np.array(ages_list) * factor))).tolist()


def jackard_sim(l1, l2):
    if len(l1) == 0 or len(l2) == 0:
        return 0
    s1 = set(l1)
    s2 = set(l2)
    c = len(s1.intersection(s2))
    return float(c) / (len(s1) + len(s2) - c)


def calculate_users_lambda2(ratings_list, papers_topics):
    us_sims = []
    i = 0
    for u in ratings_list:
        sims = []
        for i, _ in enumerate(u):
            if i < len(u) - 1:
                sims.append(1 - jackard_sim(papers_topics[u[i]], papers_topics[u[i + 1]]))
        if len(sims) == 0:
            i += 1
            us_sims.append(1)
        else:
            us_sims.append(np.mean(sims))
    return us_sims

if __name__ == '__main__':
    adaptive = False
    """
    ratings_file = '../../../datasets/citeulike/citeulike_2004_2007/time-based_split_in-matrix/' + fold + '/train-items.dat'
    ratings_ages_file = '../../../datasets/citeulike/citeulike_2004_2007/time-based_split_in-matrix/' + fold + '/train-items-ages.dat'
    theta_file = "../../../datasets/citeulike/citeulike_2004_2007/time-based_split_in-matrix/" + fold + "/lda_sklearn/theta_150.dat"
    # fold = 'fold-6'
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings_ages_file", "-r", help="The ratings ages file (-ages.dat)")
    parser.add_argument("--factor", "-f", type=float, choices=[Range(0.1, 1)], help="The anti-aging factor")
    parser.add_argument("--adaptive", "-a", action="store_true", default=False,
                        help="Set factor to be fit for adaptively, the factor value if given will be discarded.")
    args = parser.parse_args()
    factor = 1
    # Checking and setting the arguments:
    if args.ratings_ages_file:
        ratings_file = args.ratings_ages_file
    if args.factor:
        factor = args.factor
    """


    for bla in ['items', 'users']:
        for fold in [0, 1, 2, 3, 5]:
            ratings_file = '../../../datasets/citeulike/citeulike_2004_2007/time-based_split_out-of-matrix/fold-' + str(fold + 1) + '/train-' + bla + '.dat'
            ratings_ages_file = '../../../datasets/citeulike/citeulike_2004_2007/time-based_split_out-of-matrix/fold-' + str(fold + 1) + '/train-' + bla + '-ages.dat'
            theta_file = "../../../datasets/citeulike/citeulike_2004_2007/time-based_split_out-of-matrix/fold-" + str(fold + 1) + "/lda_sklearn/theta_150.dat"
            ratings_file_name = os.path.splitext(os.path.basename(ratings_file))[0]
            if '-ages' in ratings_file_name:
                output_file = os.path.join(os.path.dirname(ratings_file), ratings_file_name.replace('-ages', '-weights'))
            else:
                output_file = os.path.join(os.path.dirname(ratings_file), ratings_file_name + "-weights")

            if not os.path.exists(ratings_file):
                print("Input file is not found: {}".format(ratings_file))
                raise NameError("Input file is not found")
            # Read ratings
            print("Reading ratings file: {}".format(ratings_file))
            ratings_list = read_ratings_as_list(ratings_file)
            ratings_ages_list = read_ratings_as_list(ratings_ages_file, type_=float)

            if adaptive:
                print("Factor: {}, Fold: {}, bla: {}".format('Adaptive', fold + 1, bla))
                output_file += '_Adaptive'

                # Load papers topics
                l = []
                with open(theta_file, 'r') as f:
                    for line in f:
                        l.append([float(i) for i in line.split(" ")])
                papers_topics = []
                for i in l:
                    papers_topics.append([k for k, j in enumerate(i) if j > 0.01])

                # Sort ratings per ages:
                sorted_ratings_list = []
                for i in range(len(ratings_list)):
                    r = ratings_list[i]
                    ages = ratings_ages_list[i]
                    r_ages = list(zip(r, ages))
                    sorted_ratings_list.append([i for i, _ in sorted(r_ages, key=lambda x: x[1], reverse=True)])
                lambdas = calculate_users_lambda2(sorted_ratings_list, papers_topics)
                # lambdas = calculate_users_lambda(sorted_ratings_list, papers_topics)

                # the weights:
                weights = []
                for u, user_ratings in enumerate(ratings_ages_list):
                    # Convert to years:
                    l = [i / 12 for i in user_ratings]
                    weights.append(["{:5.3f}".format(i) for i in interest_extent_2(l, lambdas[u])])
                    # weights.append(["{:5.3f}".format(i) for i in interest_extent(l, lambdas[u])])

                output_file += '_IE2.dat'
                # Saving the predictions:
                write_ratings(weights, output_file)

            else:
                for factor in [0.1, 0.5, 1]:
                    output_f = output_file + '_Factor_{}'.format(factor)
                    print("Factor: {}, Fold: {}, bla: {}".format(factor, fold+1, bla))
                    # the weights:
                    weights = []
                    for u,user_ratings in enumerate(ratings_ages_list):
                        # Convert to years:
                        l = [i/12 for i in user_ratings]
                        weights.append(["{:5.3f}".format(i) for i in interest_extent_2(l, factor)])

                    output_f += '_IE2.dat'
                    # Saving the predictions:
                    print("Saving to file: [{}]".format(output_f))
                    write_ratings(weights, output_f)