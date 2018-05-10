import numpy as np

class PMF(object):
    def __init__(self, num_user, num_item, num_feat=150, epsilon=1, reg_param=0.1, momentum=0.8, maxepoch=20, num_batches=2, batch_size=100):
        self.num_user = num_user
        self.num_item = num_item
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self.reg_param = reg_param  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)

        self.w_item = None  # Item feature vectors
        self.w_user = None  # User feature vectors

        self.rmse_train = []
        self.rmse_test = []

    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
    # ***************** train_vec=TrainData, test_vec=TestData*************#
    # train_vec: list of training data arrays for timestamps
    def fit(self, train_vec_time_series, T0):


        # initialize
        self.epoch = 0
        # The items' latent factors are shared among all time-slots
        self.w_item = 0.1 * np.random.randn(self.num_item, self.num_feat)

        # Initialize the usres' latent factors for each time slot.
        #self.w_user = [0.1 * np.random.randn(self.num_user, self.num_feat) for _ in range(T0)]
        self.w_user = 0.1 * np.random.randn(self.num_user, T0, self.num_feat)


        self.w_Item_inc = np.zeros((self.num_item, self.num_feat))
        self.w_User_inc = np.zeros((self.num_user, T0, self.num_feat))

        while self.epoch < self.maxepoch:
            self.epoch += 1
            # Iterate over all time slots
            for time_slot in range(T0):

                # Get the ratings of time (time_slot)
                train_vec = train_vec_time_series[train_vec_time_series.idx==time_slot][['user','paper']].values

                if len(train_vec) > 0:
                    # Count the number of ratings
                    pairs_train = train_vec.shape[0]

                    # Shuffle training truples
                    shuffled_order = np.arange(train_vec.shape[0])
                    np.random.shuffle(shuffled_order)
                    if pairs_train < self.batch_size:
                        batch_s = pairs_train
                    else:
                        batch_s = self.batch_size
                    if pairs_train//batch_s < self.num_batches:
                        batch_num = pairs_train//batch_s
                    else:
                        batch_num = self.num_batches
                    print("time slot: {:d}, #ratings: {}, batch_size: {}, batch_num: {}".format(time_slot, train_vec.shape[0], batch_s, batch_num))
                    # Batch update
                    for batch in range(batch_num):
                        print("epoch: {:d}, batch: {:d} ".format(self.epoch, batch+1))
    
                        test = np.arange(batch_s * batch, batch_s * (batch + 1))
                        batch_idx = np.mod(test, shuffled_order.shape[0]) 
    
                        batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                        batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')
    
                        # Compute Objective Function
                        try:
                            pred_out = np.sum(np.multiply(self.w_user[batch_UserID,time_slot, :], self.w_item[batch_ItemID, :]), axis=1)
                        except Warning:
                            print('RuntimeWarning was rased in multiply, batch is skipped')
                            continue
                        #rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2]
                        rawErr = pred_out - len(pred_out)*[1]
    
                        # Compute gradients
                        Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], self.w_item[batch_ItemID, :]) + self.reg_param * self.w_user[batch_UserID, time_slot, :]
                        Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], self.w_user[batch_UserID,time_slot, :]) + self.reg_param * (self.w_item[batch_ItemID, :])
    
                        dw_Item = np.zeros((self.num_item, self.num_feat))
                        dw_User = np.zeros((self.num_user, self.num_feat))
    
                        # loop to aggreate the gradients of the same element
                        for i in range(batch_s):
                            dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
                            dw_User[batch_UserID[i], :] += Ix_User[i, :]
    
                        # Update with momentum
                        self.w_Item_inc = self.momentum * self.w_Item_inc + self.epsilon * dw_Item / batch_s
                        self.w_User_inc[:,time_slot,:] = self.momentum * self.w_User_inc[:,time_slot,:] + self.epsilon * dw_User / batch_s
    
                        self.w_item = self.w_item - self.w_Item_inc
                        self.w_user[:,time_slot, :] = self.w_user[:,time_slot, :] - self.w_User_inc[:,time_slot,:]
    
                        # Compute Objective Function after
                        if batch == batch_num - 1:
                            pred_out = np.sum(np.multiply(self.w_user[train_vec[:, 0],time_slot, :], self.w_item[train_vec[:, 1], :]), axis=1)
                            #rawErr = pred_out - train_vec[:, 2]
                            rawErr = pred_out - len(pred_out) * [1]
    
                            obj = np.linalg.norm(rawErr) ** 2 + 0.5 * self.reg_param * (np.linalg.norm(self.w_user[:, time_slot]) ** 2 + np.linalg.norm(self.w_item) ** 2)
    
                            self.rmse_train.append(np.sqrt(obj / pairs_train))
                            print('Training RMSE: {:4.3f}'.format(self.rmse_train[-1]))
                        """
                        # Compute validation error
                        if batch == batch_num - 1:
                            pred_out = np.sum(np.multiply(self.w_user[np.array(train_vec[:, 0], dtype='int32'),time_slot, :],
                                                          self.w_item[np.array(test_vec[:, 1], dtype='int32'), :]),
                                              axis=1) 
                            #rawErr = pred_out - test_vec[:, 2] 
                            rawErr = pred_out - len(pred_out) * [1]
    
                            self.rmse_test.append(np.linalg.norm(rawErr) / np.sqrt(pairs_test))
    
                            # Print info
                            if batch == batch_num- 1:
                                print('Training RMSE: {:4.3f}, Test RMSE {:4.3f}'.format(self.rmse_train[-1], self.rmse_test[-1]))
                        """
        return self.w_user,self.w_item  