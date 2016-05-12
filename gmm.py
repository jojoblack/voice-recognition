import sys,os
import numpy as np
import scipy.cluster.vq as vq
import random
import struct

flag_gmm = 'GMM'
split_params = 0.01  # split parameter for a kmeans cluster
n_min_frame = 10     # minimun frame num each kmeans cluster should have
var_floor = 1e-5
var_ceiling = 1e5
n_gmm_iteration = 8 
n_kmeans_iteration = 8

min_valid_prob = 1e-6
relevent_factor = 16


class GMM:
    """
    Brief:
    Gaussian mixture model (GMM).
    """
    def __init__(self, n_mix=None, n_dim=None, means=None, icovs=None, log_weights=None, epsilon=1e-5):
        """
        Brief:
        initialising parameters if needed
        
        Parameters:
        n_mix: number of mixtures (components) to fit
        n_dim: feature dimension
        means: gaussian mixture means
        icovs: gaussian mixture inverse covariances, using diagnal covariances
        weights: mixture weights
        epsilon: convergence criterion

        GMM parameters:
        n_mix: mixture of GMM
        n_dim: dimension of features
        means: mu
        covs: Sigma (diagnose)
        icovs: Sigma^(-1)
        log_const: log( |Sigma|^(1/2) )
        weights: w
        log_weight: log( w )
        """
        # assert n_mix and n_dim, "Need to define dim and ncomps."
        self.n_mix = n_mix
        self.n_dim = n_dim
        self.means = means

        self.icovs = icovs
        self.covs = None
        self.log_const = None
        if self.icovs is not None:
            self.covs = 1.0 / self.icovs
            self.log_const = 0.5 * np.log((np.prod(self.covs, 1)))

        self.log_weights = log_weights
        self.weights = None
        if self.log_weights is not None:
            self.weights = np.exp(log_weights)
        
        self.epsilon = epsilon

    def log_single_gaussian_pdf(self, fea, i):
        """
        Brief:
        Compute multivariate Gaussian log-density of one mixture
        """
        lpr = self.log_weights[i] - self.log_const[i] + np.sum(-0.5 * (fea - self.means[i]) ** 2 * self.icovs[i], 1)
        return lpr

    def log_multiple_gaussian_pdf(self, fea):
        """
        Brief:
        Compute multivariate Gaussian probability
        """
        lpr = np.zeros((fea.shape[0], self.n_mix))
        for i in np.arange(self.n_mix):
            lpr[:, i] = self.log_single_gaussian_pdf(fea, i)
        return lpr

    def multiple_gaussian_pdf(self, fea):
        """
        Brief:
        Compute multivariate Gaussian posterior probability
        """
        pr = np.exp(self.log_multiple_gaussian_pdf(fea))
        return pr
        
    def eu_distance(self, fea, mean):
        """
        Brief:
        Compute euclidean distance
        """
        if fea.ndim == 1:       
            dis = np.linalg.norm(fea - mean)
        else:
            # dis = np.linalg.norm(fea - mean, axis = 1)
            dis = np.sum((fea - mean)**2, 1)
        return dis

    def statistics(self, fea):
        """
        Brief:
        extract acc_zero acc_first statistics for svf

        Note:
        fea is a matrix that contains features in its rows
        return:
            zero_order: a vector (zero-order stat)
                Sum( Pr(i|x_t) )
            firs_order: a matrix contains modified first-order stat in rows
                Sum( Pr(i|x_t) (x_t - mu_i) )
        """
        
        n = fea.shape[0]
        
        # probabilities of each mixtures of features
        pdfs = np.zeros((n, self.n_mix), np.float32)
        
        # acc_zero: accumulate posteriors
        acc_zero = np.zeros(self.n_mix, np.float32)
        
        # acc_first: accumulate posteriors * features
        acc_first = np.zeros((self.n_mix, self.n_dim), np.float32)
        
        for i in np.arange(self.n_mix):
            pdfs[:, i] = self.log_single_gaussian_pdf(fea, i)    # TODO: fea is a matrix, does this function work well?
        pdfs = np.exp(pdfs)                   # TODO: exp lead to small value?
        sum_pdfs = np.sum(pdfs, 1)
        idx_pdfs_nonzero = (sum_pdfs != 0)    # TODO: delete the zero-sum row. No need to care about them
        
        pdfs[idx_pdfs_nonzero, :] = pdfs[idx_pdfs_nonzero, :] / sum_pdfs[idx_pdfs_nonzero][:, np.newaxis]
        var_zero = np.sum(pdfs, 0)
        acc_zero = var_zero
        for i in np.arange(self.n_mix):
            idx_fea_nonzero = (pdfs[:, i] != 0)    # TODO: does zero matters?
            var_first = pdfs[idx_fea_nonzero, i][:, np.newaxis] * (fea[idx_fea_nonzero] - self.means[i])
            acc_first[i, :] = np.sum(var_first, 0)
            
        return acc_zero, acc_first

    def adapt(self, features):
        if features.shape[0] < n_min_frame:
            print('Not enough features to adapt GMM')
            return
        zero_order, first_order = self.statistics(features)
        alpha = zero_order / (zero_order + relevent_factor)
        for i in np.arange(self.n_mix):            # TODO: change to matrix operation
            if zero_order[i] < min_valid_prob:
                continue
            self.means[i] += alpha[i] * (first_order[i, :] / zero_order[i])
        return

    def loglikelihood(self, features):
        n_frame = features.shape[0]
        log_prob = self.log_multiple_gaussian_pdf(features)

        log_p_x = log_prob[:, 0]
        for i in np.arange(1, self.n_mix):
            log_p_x = np.logaddexp(log_p_x, log_prob[:, i])
        log_p = np.sum(log_p_x) / n_frame
        #prob = np.exp(log_prob)
        #log_p = np.mean(np.log(np.sum(prob, 1)))
        return log_p

    def frame_loglikelihood(self, features):
        n_frame = features.shape[0]
        log_prob = self.log_multiple_gaussian_pdf(features)

        log_p_x = log_prob[:, 0]
        for i in np.arange(1, self.n_mix):
            log_p_x = np.logaddexp(log_p_x, log_prob[:, i])
        return log_p_x


    def train(self, features):
        """
        Brief:
        Fits a GMM into a set of feature data.
        """
        # Initialise
        n_fea = features.shape[0]

        old_log_likelihood = 0
        self._initialise_parameters(features)
        iter = 8
        # print features
        while iter > 0:

            print 'iter = ', iter

            pdfs = np.zeros((n_fea, self.n_mix), np.float32)
            #pdfs = np.zeros(n, np.float)
            
            # acc_zero: accumulate posteriors
            acc_zero = np.zeros(self.n_mix, np.float32)
            
            # acc_first: accumulate posteriors * features
            acc_first = np.zeros((self.n_mix, self.n_dim), np.float32)
            
            # acc_second: accumulate posteriors * features * features
            acc_second = np.zeros((self.n_mix, self.n_dim), np.float32)
            
            # E-step too slow
            # for i in np.arange(n_fea):
            #     #print i
            #     var_zero = self.multivariate_gaussian_pdf(features[i])
            #     #print var_zero.shape
            #     pdfs[i] = np.sum(var_zero)
            #     var_zero = var_zero / np.sum(var_zero)
            #     acc_zero += var_zero
            #     var_tmp = var_zero.reshape(self.n_mix, 1) * np.tile(features[i], (self.n_mix, 1))
            #     acc_first += var_tmp
            #     acc_second += var_tmp * np.tile(features[i], (self.n_mix, 1))
            
            acc_log_likelihood = 0.0
        
            # E-step
            for i in np.arange(self.n_mix):
                pdfs[:, i] = self.log_single_gaussian_pdf(features, i)
            pdfs = np.exp(pdfs)
            sum_pdfs = np.sum(pdfs, 1)
            idx_pdfs_nonzero = (sum_pdfs != 0)
            
            acc_log_likelihood = np.sum(np.log(sum_pdfs[sum_pdfs != 0])) / n_fea
            pdfs[idx_pdfs_nonzero] = pdfs[idx_pdfs_nonzero] / sum_pdfs[idx_pdfs_nonzero][:, np.newaxis]
            var_zero = np.sum(pdfs, 0)
            acc_zero = var_zero
            idx_mix_nonzero = (var_zero != 0)
            for i in np.arange(self.n_mix):
                idx_fea_nonzero = (pdfs[:, i] != 0)
                var_first = pdfs[idx_fea_nonzero, i][:, np.newaxis] * features[idx_fea_nonzero]
                acc_first[i, :] = np.sum(var_first, 0)
                acc_second[i, :] = np.sum(var_first * features[idx_fea_nonzero], 0)
		
            # M-step
            idx_mix_nonzero = (acc_zero != 0)
            self.means[idx_mix_nonzero, :] = acc_first[idx_mix_nonzero, :] / acc_zero[idx_mix_nonzero][:, np.newaxis]
            #print acc_zero
            self.covs[idx_mix_nonzero, :] = acc_second[idx_mix_nonzero, :] / acc_zero[idx_mix_nonzero,][:, np.newaxis] - self.means[idx_mix_nonzero, :] * self.means[idx_mix_nonzero, :]
            self.covs[self.covs < var_floor] = var_floor
            self.covs[self.covs > var_ceiling] = var_ceiling
            self.icovs = 1.0 / self.covs
            self.weights = acc_zero[idx_mix_nonzero] / n_fea
            self.log_weights = np.log(self.weights)
            self.log_const = np.log(np.sqrt(np.prod(self.covs, 1)))

            # Check for convergence
            #print(acc_log_likelihood)
            if np.absolute(old_log_likelihood - acc_log_likelihood) < self.epsilon:
                break

            old_log_likelihood = acc_log_likelihood
            iter -= 1
                
    def kmeans(self, features, n_mix):  
        """
        Brief:
        kmeans for clustering features.
        """
        n_fea = features.shape[0]  
      
        ##init centroids  
        n_fea, n_dim = features.shape  
        centroids = np.vstack(random.sample(features, n_mix))  
        iter = n_kmeans_iteration
        min_frame = 10
        old_sum_distances = 0.0
        
        while iter > 0:  
            distances = np.zeros((n_fea, n_mix))
            for i in np.arange(n_mix):
                distances[:, i] = self.eu_distance(features, centroids[i])
            #print distances[0,:]
            labels = np.argmin(distances, 1)
            # print labels
            n_split = 0
            idx = []
            
            for i in np.arange(n_mix):
                n_centroid = features[labels == i, :].shape[0]
                idx.append((i,n_centroid))
                #print n_centroid
                if n_centroid < min_frame:
                    n_split += 1 
                    continue 
                centroids[i] = np.sum(features[labels == i, :], 0) / n_centroid
            
            if n_split > 0:
                idx = sorted(idx, key = lambda idx: idx[1])
                for i in np.arange(n_split):
                    centroids[idx[i][0]] = centroids[idx[n_mix - 1 - i][0]]*(1 + split_params) 
            
            sum_distances = np.sum(np.min(distances, 1)) 
            print("distances = ",sum_distances)
            if(old_sum_distances - sum_distances) < self.epsilon:
                break
            iter -= 1    
           
        return centroids, labels
        
    def _initialise_parameters(self, features):
        """
        Brief:
        Initialises parameters: means, covariances, and mixing probabilities using kmeans if undefined.
        """
        #print("initialising parameters using kmeans...")
        (fea_num, fea_dim) = features.shape

        if not self.n_dim:
            self.n_dim = fea_dim

        if self.n_dim != fea_dim:
            print("feature dimension does not match!")
            exit()

        if self.means is None or self.icovs is None or self.log_weights is None:

            # use kmeans to initialize the parameters
            centroids, labels = vq.kmeans2(features, self.n_mix, minit = "points", iter = n_kmeans_iteration, thresh = 1e-6)
            #centroids, labels = self.kmeans(features, self.n_mix)
            clusters = [[] for i in np.arange(self.n_mix)]
            for (l,d) in zip(labels, features):
                clusters[l].append(d)

            #for cluster in clusters:
            #    print len(cluster)
            # weights
            if self.log_weights is None:
                self.weights = np.array([len(c) for c in clusters], np.float) / fea_num
            self.log_weights = np.log(self.weights)

            # means, the cluster centers
            if self.means is None:  
                self.means = centroids

            # covariances, using diagonal covariances
            if self.icovs is None:
                self.covs = np.ones((self.n_mix, self.n_dim))
                covariances = []
                for i in np.arange(self.n_mix):
                    cluster = np.array(clusters[i])
                    num = len(cluster)
                    var = np.sum((cluster - centroids[i])**2, 0)/num
                    covariances.append(var)
                self.covs = np.array(covariances)
                self.covs[self.covs < var_floor] = var_floor
                self.covs[self.covs > var_ceiling] = var_ceiling
                self.icovs = 1.0 / self.covs
                self.log_const = 0.5 * np.log((np.prod(self.covs, 1)))
        #print("initialising done...")        
            
    def read(self, fn):
        """
        Brief:
        read gmm file of lab
        """
        fp = open(fn, 'rb')
        flag_gmm = struct.unpack('4s',fp.read(4))[0]
        self.n_dim = struct.unpack('i', fp.read(4))[0]
        self.n_mix = struct.unpack('i', fp.read(4))[0]
        print("dimension = %d, mixture = %d"%(self.n_dim, self.n_mix))
        self.log_weights = np.zeros(self.n_mix)
        
        for i in range(self.n_mix):
            self.log_weights[i] = struct.unpack('f', fp.read(4))[0]
        self.weights = np.exp(self.log_weights)
            
        self.means = np.zeros((self.n_mix, self.n_dim))
        for i in range(self.n_mix):
            for j in range(self.n_dim):
                self.means[i, j] = struct.unpack('f', fp.read(4))[0]
            
        self.icovs = np.zeros((self.n_mix, self.n_dim))
        for i in range(self.n_mix):
            for j in range(self.n_dim):
                self.icovs[i, j] = struct.unpack('f', fp.read(4))[0]
        self.covs = 1.0 / self.icovs       
         
        self.log_const = np.zeros(self.n_mix)
        for i in range(self.n_mix):
            self.log_const[i] = struct.unpack('f', fp.read(4))[0]
        # print self.log_weights, self.means, self.icovs
            
    def write(self, fn):
        """
        Brief:
        write gmm file of lab
        """
        fp = open(fn, 'wb')
        fp.write(struct.pack('4s', flag_gmm))
        fp.write(struct.pack('i', self.n_dim))
        fp.write(struct.pack('i', self.n_mix))
        
        for i in range(self.n_mix):
            fp.write(struct.pack('f', self.log_weights[i]))
            
        for i in range(self.n_mix):
            for j in range(self.n_dim):
                fp.write(struct.pack('f', self.means[i,j]))
            
        for i in range(self.n_mix):
            for j in range(self.n_dim):
                fp.write(struct.pack('f', self.icovs[i,j]))
            
        for i in range(self.n_mix):
            fp.write(struct.pack('f', self.log_const[i]))
