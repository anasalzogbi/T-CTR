import numpy as np
def convert_to_probability_dist(v):
    v_min = v.min(axis = 1)[:, None]
    v = (v - v_min) / (v.max(axis=1)[:, None] - v_min)
    v = v / v.sum(axis=1)[:, None]
    return v

def kron_A_N(A, N):  # Simulates np.kron(A, np.eye(N))
    """
    Source: https://stackoverflow.com/questions/44461658/efficient-kronecker-product-with-identity-matrix-and-regular-matrix-numpy-pyt
    One approach would be to initialize an output array of 4D and then assign values into it from A. Such an assignment would broadcast values and this is where we would get efficiency in NumPy.
    """
    # Get shape of A
    m,n = A.shape

    # Initialize output array as 4D
    out = np.zeros((m,N,n,N))

    # Get range array for indexing into the second and fourth axes 
    r = np.arange(N)

    # Index into the second and fourth axes and selecting all elements along
    # the rest to assign values from A. The values are broadcasted.
    out[:,r,:,r] = A

    # Finally reshape back to 2D
    out.shape = (m*N,n*N)
    return out


def calculate_metrics_user(hits, num_user_test_positives, recall_breaks, mrr_breaks, ndcg_breaks):
    # Adjust the breaks lists to be 0-based:
    recall_breaks = [i - 1 for i in recall_breaks]
    mrr_breaks = [i - 1 for i in mrr_breaks]
    ndcg_breaks = [i - 1 for i in ndcg_breaks]
    iDCGs = np.cumsum(np.array([1 / np.log2(i + 2) for i in range(len(hits))]))

    # Calculate recall:
    recall = np.cumsum(hits)
    recall_at_breaks = (np.array(recall)[recall_breaks] / float(num_user_test_positives)).tolist()

    # Calculate MRR
    mrrs = [hits[i] / float(i + 1) for i in range(len(hits))]
    for i in range(1, len(mrrs)):
        mrrs[i] = max(mrrs[i], mrrs[i - 1])
    mrrs_at_breaks = np.array(mrrs)[mrr_breaks].tolist()

    # Calculate nDCG
    dcgs = [hits[i] / np.log2(i + 2) for i in range(len(hits))]
    dcgs = np.array(dcgs)
    dcgs = np.cumsum(dcgs) / iDCGs
    ndcgs_at_breaks = dcgs[ndcg_breaks].tolist()
    return recall_at_breaks + mrrs_at_breaks + ndcgs_at_breaks