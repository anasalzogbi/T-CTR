import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from util.files_utils import read_mappings
from util.files_utils import read_ratings_as_list
from lib.Probabilistic_Matrix_Factorization import PMF
from util.utils import kron_A_N, calculate_metrics_user
import time
class CE():
    def __init__(self, num_users, num_items, intervals_count, num_factors=50, epochs=30, phi=10, reg_param=0.5, tolerance = 1e-5, momentum = 0.5):
        self.factors = num_factors
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.max_epochs = epochs
        self.reg_param = reg_param
        self.phi = phi
        self.intervals_count = intervals_count
        self.tol = tolerance
        self.momentum = momentum

    def bootstrap(self, ratings_df, T0):
        # Decide
        pmf = PMF(self.num_users, self.num_items, num_feat=self.factors, epsilon=0.1, reg_param=self.reg_param, momentum=0.8, maxepoch=self.max_epochs, num_batches=5, batch_size=1000)
        return pmf.fit(ratings_df, T0=T0)


    def calculate_a_user(self, U, t, verbos = False):
        # U: K X |T|
        U = U[:, 0:t]
        if verbos:
            print("U:")
            print("-----------")
            print(U)
        Z = np.zeros((U.shape[1] - self.phi, self.phi * self.num_factors))
        for j in range(self.phi - 1, U.shape[1] - 1):
            end = j - self.phi if j >= self.phi else -U.shape[1] - 1
            a = U[:, j:end:-1].T.reshape((U.shape[0] * (end - j)))
            Z[j - self.phi + 1, :] = a
        if verbos:
            print("Z:")
            print("-----------")
            print(Z)
            print("Y: ")
            print("-----------")
            print(U.flatten('F')[self.phi * self.num_factors:, np.newaxis])
        eye = 0.01 *np.eye(Z.shape[1])
        # zzt=Z.dot(Z.T)
        zzt = Z.T.dot(Z)+eye
        if verbos:
            print("zzt:")
            print("-----------")
            print(zzt.shape)
        eye = np.eye(zzt.shape[0])
        # zztinv = np.linalg.inv(zzt)
        zztinv = np.linalg.solve(zzt, eye)
        if verbos:
            if np.allclose(zzt.dot(zztinv), eye, rtol=1e-02, atol=1e-02):
                print("inverted correctly")
            else:
                print("was not inverted correctly")
                print(zzt.dot(zztinv))
        zztz = zztinv.dot(Z.T)

        if verbos:
            print(zztz.shape)
            # print(U.flatten('F')[self.phi*k:, np.newaxis].shape)
        cron = kron_A_N(zztz, self.num_factors)
        if verbos:
            print('cron shape', cron.shape)
        B = cron.dot(U.flatten('F')[self.phi * self.num_factors:, np.newaxis])
        if verbos:
            print("B:")
            print("-----------")
            print(B.shape)
        A = []
        for i in B.reshape(self.phi, self.num_factors * self.num_factors):
            A.append(i.reshape((self.num_factors, self.num_factors), order='F'))
        return A

    def calculate_a(self, UT, t):
        #UT: self.num_users, self.intervals_count, self.num_factors
        A = []
        times = []
        for u in range(self.num_users):
            U = UT[u]
            # U dimensions: intervals_count X num_factors
            if u % 50 ==0:
                print('Calculating A for user {}/ {}'.format(u,self.num_users))
            t0 = time.time()
            A.append(self.calculate_a_user(U.T, t))
            times.append(time.time() - t0)
        print("Calculating A for {} users in {:5.4f} seconds in average".format(self.num_users, sum(times)/len(times)))
        return A

    def calculate_W(self, A, UT,t):
        W = np.zeros(self.num_factors)
        for i in range(self.phi):
            W += A[i].dot(UT[t - i-1])
        return W



    def fit(self, ratings_df):
        # Define T0:
        T0 = self.num_factors * self.phi+self.phi

        # Bootstrap
        UT0, V0 = self.bootstrap(ratings_df, T0)

        # initialize A's for all users:
        A = self.calculate_a(UT0, T0)
        UT = 0.1 * np.random.randn(self.num_users, self.intervals_count, self.num_factors)
        VT = 0.1 * np.random.randn(self.num_items, self.intervals_count, self.num_factors)
        UT[:,0:T0,:] = UT0
        converge_num = 0
        for t in range(T0, self.intervals_count):
            print("Time point: {}".format(t))
            # Get the ratings of time (time_slot)
            train_vec = ratings_df[ratings_df.idx == t][['user', 'paper']].values
            if len(train_vec) > 0:
                W = []
                for u in range(self.num_users):
                    W.append(self.calculate_W(A[u], UT[u], t))
                W = np.array(W)
                converged = False
                epoch = 1
                last_rmse = None
                while not converged and epoch <= self.max_epochs:
                    #print("Epoch: {}".format(epoch))
                    user_IDs = np.array(train_vec[:,0], dtype='int32')
                    item_IDs = np.array(train_vec[:,1], dtype='int32')

                    # Compute Objective Function
                    pred_out = np.sum(np.multiply(UT[user_IDs, t, :], VT[item_IDs,t, :]), axis=1)

                    rawErr = pred_out - len(pred_out) * [1]

                    # Compute gradients
                    #print("Updating U, V")
                    U_W = UT[user_IDs, t, :] - W[user_IDs]
                    Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], VT[item_IDs, t, :]) + 2 * self.reg_param * U_W
                    V_V0 = VT[item_IDs, t, :] - V0[item_IDs]
                    Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], UT[user_IDs, t, :]) + 2 * self.reg_param * V_V0

                    dw_Item = np.zeros((self.num_items, self.num_factors))
                    dw_User = np.zeros((self.num_users, self.num_factors))

                    # loop to aggreate the gradients of the same element
                    for i in range(len(train_vec)):
                        dw_Item[item_IDs[i], :] += Ix_Item[i, :]
                        dw_User[user_IDs[i], :] += Ix_User[i, :]

                    # Update:
                    VT[:, t, :] = VT[:, t, :] - (self.momentum * dw_Item / len(train_vec))
                    UT[:, t, :] = UT[:, t, :] - (self.momentum * dw_User / len(train_vec))

                    # stop when converge
                    pred_out = np.sum(np.multiply(UT[user_IDs, t, :], VT[item_IDs, t, :]), axis=1)
                    rawErr = pred_out - len(pred_out) * [1]
                    obj = np.linalg.norm(rawErr) ** 2 +self.reg_param * (np.linalg.norm(U_W) ** 2 + np.linalg.norm(V_V0) ** 2)

                    train_rmse = np.sqrt(obj / len(train_vec))
                    #print("RMSE: {:5.3f}".format(train_rmse))
                    #if last_rmse:
                    #    print("rmse diff: {:5.3f}".format(train_rmse - last_rmse))
                    if last_rmse and abs(train_rmse - last_rmse) <= self.tol:
                        print('converges at iteration {:d}, RMSE: {:5.4f}/{:5.4f} #ratings: {:d}'.format(epoch,train_rmse,last_rmse,  len(train_vec)))
                        converged = True
                        converge_num +=1
                    else:
                        last_rmse = train_rmse
                    epoch += 1
                #if not converged:
                #    print("Epochs count: {}, not converged".format(epoch-1))
                #print("Updating A...")
                A = self.calculate_a(UT, t)
        print("Number of convergence: {}".format(converge_num))
        self.A = A
        self.UT = UT
        self.V0 = V0
        self.VT = VT

    def evaluate(self, test_ratings_df, candidate_papers, recall_breaks=[5, 10] + list(range(20, 201, 20)), mrr_breaks=[10], ndcg_breaks=[5, 10], folds_num=5, top=200):
        print("Evaluating...")
        metrics_count = len(recall_breaks) + len(mrr_breaks) + len(ndcg_breaks)
        max_t = test_ratings_df.idx.max()
        predictions = np.zeros((self.num_users, self.num_items))
        U = 0.1 * np.random.randn(self.num_users, max_t, self.num_factors)
        U[:, 0:self.intervals_count, :] = self.UT
        test_intervals= test_ratings_df.idx.drop_duplicates()
        users_with_test = set()
        results = np.zeros(shape=(len(test_intervals),num_users, metrics_count))
        t_idx = 0
        results.fill(-1)
        for t in range(self.intervals_count, test_ratings_df.idx.max()):
            print("Interval: {}".format(t))
            users_with_zero_test = 0
            # Calculate W:
            W = []
            for u in range(self.num_users):
                W.append(self.calculate_W(self.A[u], U[u], t))
            test_vec = test_ratings_df[test_ratings_df.idx == t][['user', 'paper']].values

            # If the interval t has tes ratings
            if len(test_vec) >0:
                print("Evaluating interval: {}".format(t))
                # There are some test ratins at this time interval, run evaluation:
                W = np.array(W)
                R = W.dot(self.V0.T)
                for user in range(num_users):
                    # Get the test positive items for the user
                    user_test_positive = list(set(test_vec[test_vec[:,0]==user][:,1]))
                    if len(user_test_positive) == 0:
                        # This user has no test positives, add Not available as result values
                        users_with_zero_test += 1
                        # Write results to the file
                        l = map(str, ['{:7d}'.format(user), '{:7d}'.format(fold + 1)] + ["{:>7}".format('NA')] * metrics_count)
                        continue
                    users_with_test.add(user)
                    # Get the prediction scores for the candidate items
                    scores_u = R[user, candidate_papers[user]]

                    # Identify the top recommendations
                    recommended_items_idx = np.argsort(scores_u)[::-1][0:top]
                    recommended_items_ids = np.array(candidate_papers[user])[recommended_items_idx]

                    # Identify the hits:
                    hits = [1 if i in user_test_positive else 0 for i in recommended_items_ids]

                    # Calculate the metrics:
                    metrics_values = calculate_metrics_user(hits, len(user_test_positive), recall_breaks, mrr_breaks, ndcg_breaks)
                    try:
                        results[t_idx,user,:] = metrics_values
                    except:
                        print(t_idx,user, results.shape)
                # Increment for the intervals only if t has ratings
                t_idx +=1
        # Keep the maximum metric value across all test intervals:
        res = np.max(results, axis=(0))

        # The evaluation result, average over users with test:
        res =  np.average(res[list(users_with_test)], axis=0)
        print( ["{:7.3f}".format(i) for i in res])
        return res
def check_ex(x, ratings):
    if ratings[x.user, x.paper] == 1:
        return True
    return False

def load_data(timestamped_ratings_file, training_ratings_file, test_ratings_file, users_mapping_file, items_mapping_file, interval):

    # 1- Load training ratings into a dataframe and add timestamps to them: (u_id, p_id, timestamp)
    training_list = read_ratings_as_list(training_ratings_file)
    test_list = read_ratings_as_list(test_ratings_file)

    # Read userid- citeulikeid mapping
    users_dict = read_mappings(users_mapping_file)

    # Read citeulike_paperid mapping
    items_dict = read_mappings(items_mapping_file)

    num_items = len(items_dict)
    num_users = len(users_dict)

    training_ratings_mat = np.zeros((len(training_list), num_items))
    for u_id, u_list in enumerate(training_list):
        for i in u_list:
            training_ratings_mat[u_id, i] = 1

    test_ratings_mat = np.zeros((len(training_list), num_items))
    for u_id, u_list in enumerate(test_list):
        for i in u_list:
            test_ratings_mat[u_id, i] = 1

    print("#Users: {}, #Items: {}, #Ratings: {}".format(len(training_list), num_items, int(training_ratings_mat.sum())))

    # Load timestamped ratings:
    ratings_df = pd.read_csv(timestamped_ratings_file, sep=',', header=0, names=['user', 'paper', 'date', 'tag'], parse_dates=['date'])[
        ['user', 'paper', 'date']].drop_duplicates()

    ratings_df_f5 = ratings_df[(ratings_df.user.isin(users_dict.keys()) & (ratings_df.paper.isin(items_dict.keys())))].drop_duplicates()
    ratings_df_f5['user'] = ratings_df_f5.user.apply(lambda x: users_dict[x])
    ratings_df_f5['paper'] = ratings_df_f5.paper.apply(lambda x: items_dict[str(x)])

    ratings_df_f5.sort_values(by=['date'], inplace=True)
    per = ratings_df_f5.date.dt.to_period(interval)
    g = ratings_df_f5.groupby(per)
    ratings_df_f5['idx'] = g.grouper.group_info[0]

    ratings_train_df = ratings_df_f5[ratings_df_f5.apply(check_ex, axis=1, args=(training_ratings_mat,))]
    ratings_test_df = ratings_df_f5[ratings_df_f5.apply(check_ex, axis=1, args=(test_ratings_mat,))]

    return ratings_train_df, ratings_test_df, num_users, num_items

local = False
if local:
    # Local configs:
    fold = 1
    num_factors, phi, epochs = (5, 20, 30)
    # Load data:
    base_dir = '../../../datasets/citeulike/citeulike_2004_2007'
    exp_dir = '../../../datasets/citeulike/citeulike_2004_2007/time-based_split_out-of-matrix/CE/fold-'+str(fold)
    interval = 'W'


else:
    # Server configs:
    fold = 5
    num_factors, epochs, phi = (5, 10, 4)
    base_dir = "/vol1/data_anas/citeulike_2004_2007"
    exp_dir = "/vol1/data_anas/citeulike_2004_2007/time-based_split_out-of-matrix/CE/fold-"+str(fold)
    interval = 'M'

parser = argparse.ArgumentParser()
parser.add_argument("--fold", "-f", type=int, help="The fold number.")
parser.add_argument("--phi", "-phi", type=int)
parser.add_argument("--num_factors", "-k", type=int, help="The number of latent factors.")
parser.add_argument("--epochs", "-e", type=int, help="The number of epochs.")
parser.add_argument("--interval", "-i", choices=['D', 'W', 'M'], help="The interval. Day (D), Week (W) or Month (M).")

args = parser.parse_args()

# Checking and setting the arguments:
if args.fold:
    fold = args.fold
if args.phi:
    phi = args.phi
if args.num_factors:
    num_factors = args.num_factors
if args.epochs:
    epochs = args.epochs
if args.interval:
    interval = args.interval


timestamp_ratings_file = os.path.join(base_dir, "ratings.csv")
training_ratings_file = os.path.join(base_dir,'time-based_split_out-of-matrix', 'fold-'+str(fold), 'train-users.dat')
test_ratings_file = os.path.join(base_dir,'time-based_split_out-of-matrix', 'fold-'+str(fold), 'test-users.dat')
users_mapping_file = os.path.join(base_dir,'time-based_split_out-of-matrix', 'fold-'+str(fold), 'userhash_user_id_map.csv')
items_mapping_file = os.path.join(base_dir,'time-based_split_out-of-matrix', 'fold-'+str(fold), 'citeulike_id_doc_id_map.csv')

if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
print("Resulting Matrices will be stored in: [{}]".format(exp_dir))

training_ratings_df, test_ratings_df, num_users, num_items = load_data(timestamp_ratings_file, training_ratings_file, test_ratings_file, users_mapping_file, items_mapping_file, interval)

intervals_training = training_ratings_df.idx.max()


if intervals_training < phi * num_factors +phi:
    print("phi {} * k {} + phi {} cann't be larger than #intervals in training {}. Stop!".format(phi, num_factors, phi, intervals_training))
    sys.exit()

# Load Candidate papers:
candidates_file = os.path.join(base_dir,'time-based_split_out-of-matrix', 'fold-'+str(fold), 'candidate-items.dat')
if os.path.exists(candidates_file):
    print("Loading candidates from [{}]".format(candidates_file))
    candidate_papers = read_ratings_as_list(candidates_file)

    # ce = CE(num_users, num_items,intervals_count=training_ratings_df.idx.max()+1, num_factors=5, epochs=3, phi = 3, reg_param=0.5, tolerance = 1e-5, momentum = 0.5)
    ce = CE(num_users, num_items, intervals_count=training_ratings_df.idx.max() + 1, phi = phi, num_factors = num_factors, epochs=epochs)
    t0 = time.time()
    print("Start...")
    ce.fit(training_ratings_df)
    ce.evaluate(test_ratings_df, candidate_papers)

    np.save(os.path.join(exp_dir, "A"), ce.A)
    np.save(os.path.join(exp_dir, "UT"), ce.UT)
    np.save(os.path.join(exp_dir, "V0"), ce.V0)
    np.save(os.path.join(exp_dir, "VT"), ce.VT)
    print("factors: {}, phi: {}, epochs: {}, Interval: {}".format(num_factors, phi, epochs, interval))
    print("Finished in {:6.4f} hours ".format((time.time() - t0) / 3600))

else:
    print("Candidate file not found: [{}]".format(candidates_file))
