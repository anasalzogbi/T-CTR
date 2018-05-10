
import numpy as np


def calculate_mrr(ratings_mat, predictions_mat, top_k, mrr_breaks):
    u = 0
    mrr_list = []
    for predictions in predictions_mat:
        positive_idx = set(ratings_mat[u].nonzero()[0])
        rec_idx = np.argsort(predictions)[::-1][0:top_k]
        mrrs = [1/(i+1) if j in positive_idx else 0 for (i,j) in enumerate(rec_idx)]
        for i in range(1,len(mrrs)):
            mrrs[i] = max(mrrs[i], mrrs[i-1])
        mrr_list.append(np.array(mrrs)[mrr_breaks].tolist())
        u +=1
    return np.array(mrr_list)
