import sys
import os
import argparse
import time
import numpy as np
from gensim.matutils import kullback_leibler, jaccard, hellinger
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.files_utils import read_ratings, read_docs_topics_file
from util.similarity_metrics import jensen_shannon_div
from util.utils import convert_to_probability_dist


def calculate_hellinger_predictions(users_profiles, papers_topics, fold, splits, lag=500):
    print("Calculating predictions based on hellinger distance...")
    s_time = time.time()
    predictions = np.zeros((users_profiles.shape[0], papers_topics.shape[0]))
    step = 0
    if not (splits is None):
        print("Calculating for test items only...")
        for (i, u) in enumerate(users_profiles):
            # Get the test items, calculate the predictions for the test items only
            test_items = np.array(splits[i, fold])
            for j in test_items:
                predictions[i,j] = 1 - hellinger(u, papers_topics[j])
            step += 1
            if step % lag == 0:
                print("{} users done, time since prediction calculation: {:5.2f} minutes".format(step, (
                    time.time() - s_time) / 60))
    else:
        print("Calculating for all items ...")
        for (i, u) in enumerate(users_profiles):
            for (j, p) in enumerate(papers_topics):
                    predictions[i, j] = 1 - hellinger(u, p)
            step += 1
            if step % lag == 0:
                print("{} users done, time since prediction calculation: {:5.2f} minutes".format(step, (
                    time.time() - s_time) / 60))

    return predictions


def calculate_jensen_shannon_predictions(users_profiles, papers_topics, fold, splits, lag=100):
    print("Calculating predictions based on jensen shannon divergence...")
    s_time = time.time()
    predictions = np.zeros((users_profiles.shape[0], papers_topics.shape[0]))
    step = 0
    if not (splits is None):
        print("Calculating for test items only...")
        for (i, u) in enumerate(users_profiles):
            # Get the test items
            test_items = np.array(splits[i, fold])
            for j in test_items:
                predictions[i, j] = 1 - jensen_shannon_div(u, papers_topics[j])
            step += 1
            if step % lag == 0:
                print("{} users done, time since prediction calculation: {:5.2f} minutes".format(step, (
                    time.time() - s_time) / 60))
    else:
        print("Calculating for all items...")
        for (i, u) in enumerate(users_profiles):
            for (j, p) in enumerate(papers_topics):
                    predictions[i, j] = 1 - jensen_shannon_div(u, p)
            step += 1
            if step % lag == 0:
                print("{} users done, time since prediction calculation: {:5.2f} minutes".format(step, (
                    time.time() - s_time) / 60))

    return predictions


def calculate_cosine_sim_predictions(users_profiles, papers_topics):
    print("Calculating predictions based on cosine similarity...")
    s_time = time.time()
    predictions = cosine_similarity(users_profiles, papers_topics)
    print("Elapsed time since prediction calculation: {:5.2f} minutes".format((time.time() - s_time) / 60))
    return predictions


if __name__ == '__main__':
    base_dir = "../../data"
    papers_file = '../data/lda_topics/theta_200.dat'
    output_directory = '../../data/rocchio_based'
    users_num = 5551
    papers_num = 16980
    lag = 1000
    papers_file_list = []
    papers_to_prob_dist = False
    normalize_user_profile = False
    lda_user_profile = False
    similarity_measure = "hellinger"

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", "-u",
                        help="The base directory that contains the folds, expecting to have sub folders: fold-[1..5], "
                             "each folder has the training users ratings file")
    parser.add_argument("--papers_file", "-p",
                        help="The papers-topics file, specify this if you have a single paper file for all folds, otherwise use --papers_file_list to give a list of papers files, one for each fold")
    parser.add_argument("--papers_file_list", "-ps",
                        help="The list of papers-topics files, make sure the list is orderd (fold1, fold2,...)",
                        nargs='+')
    parser.add_argument("--similarity_measure", "-s", choices=['jenson_shannon', 'hellinger', 'cosine'],
                        help="The similarity measure used to compare users profiles with  papers' vectors.")
    parser.add_argument("--experiment", "-e", help="The experiment name, used to name the output folder, where the "
                                                   "resulting prediction files (score.npy)  will be saved, the output directory "
                                                   "will appear under the base directory")
    parser.add_argument("--split_directory", "-splt",
                        help="The directory that contains the folds. The folds folders are named as 'fold[1-5]', each one should contain the test files, the test files are needed here, so that we predict only for the tests")
    parser.add_argument("--users_num", "-n", type=int, help="The number of users in the dataset")
    parser.add_argument("--items_num", "-m", type=int, help="The number of items in the dataset")
    parser.add_argument("--step_lag", "-l", type=int, help="The step lag for reporting progress")
    parser.add_argument("--papers_to_prob_dist", "-ptdst", action="store_true", default=False,
                        help="A flag whither to convert papers vectors to a probability distribution or not, Not needed for cosine similarity. Needed for probability distribution similarity metrics (jenson shannon, and hellinger). default: False")
    parser.add_argument("--normalize_user_profile", "-no", action="store_true", default=False,
                        help="A flag idicates whither to normalize the user profile or not")
    parser.add_argument("--lda_user_profile", "-ldau", action="store_true", default=False,
                        help="A flag idicates whether to use LDA topics for building user profile")
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

    if args.papers_file_list:
        papers_file_list = args.papers_file_list

    split_directory = None
    if args.split_directory:
        split_directory = args.split_directory
        if not os.path.exists(split_directory):
            print("Split directory not found: {}".format(split_directory))
            raise NameError("Split directory not found")

    if args.experiment:
        experiment_name = args.experiment

    if args.users_num:
        users_num = args.users_num

    if args.items_num:
        items_num = args.items_num

    if args.step_lag:
        lag = args.step_lag

    if args.similarity_measure:
        similarity_measure = args.similarity_measure
    if args.papers_to_prob_dist:
        papers_to_prob_dist = True

    if args.normalize_user_profile:
        normalize_user_profile = True

    if args.lda_user_profile:
        lda_user_profile = True

    splits = None
    if not (split_directory is None):
        # reading the splits file
        splits_file = os.path.join(split_directory, "splits.npy")

        if not os.path.exists(splits_file):
            print("Splits file not found: {} ".format(splits_file))
            raise NameError("Splits file not found")
        print("loading {} ...\n".format(splits_file))
        splits = np.load(splits_file)


    if lda_user_profile:
        # Reading papers file
        print("Reading papers LDA topics file: {}".format(papers_file))
        papers_lda_topics = read_docs_topics_file(papers_file)
        print("papers LDA matrix dimensions: {}".format(papers_lda_topics.shape))

        for fold in range(1, 6):
            users_ratings_file = os.path.join(base_dir, "fold-" + str(fold), "train-fold_" + str(fold) + "-users.dat")
            print("Reading users ratings file: {}".format(users_ratings_file))
            output_dir = os.path.join(base_dir, experiment_name, "fold-" + str(fold))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Read users ratings
            ratings_mat = read_ratings(users_ratings_file, users_num, items_num)
            print("Rating matrix dimensions: {}".format(ratings_mat.shape))

            # Load the papers CTR topics:
            if len(papers_file_list) != 0:
                print("Reading papers CTR topics file: {}".format(os.path.join(base_dir, papers_file_list[fold - 1])))
                papers_ctr_topics = read_docs_topics_file(os.path.join(base_dir, papers_file_list[fold - 1]))
                print("papers CTR matrix dimensions: {}".format(papers_ctr_topics.shape))
                # Convert CTR papers_topics to probability distribution
                print("Converting papers vectors into probability distributions...")
                papers_ctr_topics = convert_to_probability_dist(papers_ctr_topics)

            # Build user profiles: Each user profile is the summation of the relevant papers' LDA topics.
            print("Building users profiles...")
            M = ratings_mat.dot(papers_lda_topics)
            # Normalize user profiles:
            print("Normalizing users profiles...")
            ratings_count = np.sum(ratings_mat, axis=1)
            users_profiles = M / ratings_count[:, None]
            """
            # Check if the normalization works, users vectors should sum up to 1
            np.sum(M, axis=1)[1:5]
            sum(np.sum(users_profiles, axis=1))
            # M[0,1:10], normalized_M[0,1:10]
            """

            if similarity_measure == "hellinger":
                predictions = calculate_hellinger_predictions(users_profiles, papers_ctr_topics, fold - 1, splits, lag)
            elif similarity_measure == "jenson_shannon":
                predictions = calculate_jensen_shannon_predictions(users_profiles, papers_ctr_topics, fold - 1, splits, lag)
            elif similarity_measure == "cosine":
                predictions = calculate_cosine_sim_predictions(users_profiles, papers_ctr_topics)

            # Saving the predictions:
            print("Storing predictions to: [{}]".format(os.path.join(output_dir, "score")))
            np.save(os.path.join(output_dir, "score"), predictions)

    else:
        # Reading papers file
        papers_topics = None
        if len(papers_file_list) == 0:
            print("Reading papers-topics file: {}".format(papers_file))
            papers_topics = read_docs_topics_file(papers_file)
            print("papers matrix dimensions: {}".format(papers_topics.shape))
            # Convert papers_topics to probability distribution if needed
            if papers_to_prob_dist:
                print("Converting papers vectors into probability distributions...")
                papers_topics = convert_to_probability_dist(papers_topics)

        splits = None
        if not (split_directory is None):
            # reading the splits file
            splits_file = os.path.join(split_directory, "splits.npy")

            if not os.path.exists(splits_file):
                print("Splits file not found: {} ".format(splits_file))
                raise NameError("Splits file not found")
            print("loading {} ...\n".format(splits_file))
            splits = np.load(splits_file)

        for fold in range(1, 6):
            users_rtings_file = os.path.join(base_dir, "fold-" + str(fold), "train-fold_" + str(fold) + "-users.dat")
            print("Reading users ratings file: {}".format(users_rtings_file))
            output_dir = os.path.join(base_dir, experiment_name, "fold-" + str(fold))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Read users ratings
            ratings_mat = read_ratings(users_rtings_file, users_num, items_num)
            print("Rating matrix dimensions: {}".format(ratings_mat.shape))

            # if the there is a different file for each fold, the files will be in papers_file_list, load the corresponding one now:
            if len(papers_file_list) != 0:
                print("Reading papers-topics file: {}".format(os.path.join(base_dir, papers_file_list[fold - 1])))
                papers_topics = read_docs_topics_file(os.path.join(base_dir, papers_file_list[fold - 1]))
                print("papers matrix dimensions: {}".format(papers_topics.shape))
                # Convert papers_topics to probability distribution if needed
                if papers_to_prob_dist:
                    print("Converting papers vectors into probability distributions...")
                    papers_topics = convert_to_probability_dist(papers_topics)

            # Build user profiles: Each user profile is the summation of the relevant papers' LDA topics.
            print("Building users profiles...")
            M = ratings_mat.dot(papers_topics)

            if normalize_user_profile:
                # Normalize user profiles:
                print("Normalizing users profiles...")
                ratings_count = np.sum(ratings_mat, axis=1)
                users_profiles = M / ratings_count[:, None]
                """
                # Check if the normalization works, users vectors should sum up to 1
                np.sum(M, axis=1)[1:5]
                sum(np.sum(users_profiles, axis=1))
                # M[0,1:10], normalized_M[0,1:10]
                """
            else:
                users_profiles = M

            if similarity_measure == "hellinger":
                predictions = calculate_hellinger_predictions(users_profiles, papers_topics, fold-1, splits, lag)
            elif similarity_measure == "jenson_shannon":
                predictions = calculate_jensen_shannon_predictions(users_profiles, papers_topics, fold - 1, splits, lag)
            elif similarity_measure == "cosine":
                predictions = calculate_cosine_sim_predictions(users_profiles, papers_topics)

            # Saving the predictions:
            print("Storing predictions to: [{}]".format(os.path.join(output_dir, "score")))
            np.save(os.path.join(output_dir, "score"), predictions)
