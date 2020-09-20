import numpy as np
import scipy
from random import randint, seed
from math import sqrt
from sklearn.linear_model import LogisticRegression


EPS = 1e-2

fit_idx = 0
predict_idx = 0

class BT:

    def __init__(self, sample_n, MAXIT=50, EPS=1e-5, feature_func = None):
        self.sample_n = sample_n
        self.MAXIT = MAXIT
        self.EPS = EPS
        self.history = []
        self.N = 5
        self.groups = ((1, 1), )
        self.history = None
        self.weights = None
        self.with_ties = True
        self.feature_func = None
        self.feature_func = feature_func



    def _get_groups(self, lst, n):
        from random import shuffle
        shuffle(lst)
        return [lst[i::n] for i in range(n)]


    def _make_s(self, pos_pairs, neg_pairs):
        self.s = np.zeros(shape=(self.features_n, self.features_n), dtype=int)
        T = 0
        self.Ts = np.zeros(shape=(self.features_n, self.features_n), dtype=int)
        # selected_pos = []
        # selected_neg = []
        # for t in range(2 * sample_n//self.features_n):
        #     a = randint(0, pos_pairs.shape[0] - 1)
        #     b = randint(0, neg_pairs.shape[0] - 1)
        for a in range(pos_pairs.shape[0]):
            for b in range(neg_pairs.shape[0]):
                # selected_pos.append(a)
                # selected_neg.append(b)
                #i = choices(range(pos_pairs.shape[1]), weights)
                #j = choices(range(neg_pairs.shape[1]), weights)
                for i in range(pos_pairs.shape[1]):
                    for j in range(i+1, neg_pairs.shape[1]):

                        if pos_pairs[a, i] > neg_pairs[b, i] and pos_pairs[a, j] < neg_pairs[b, j]:
                            self.s[i, j] += 1
                        elif pos_pairs[a, i] < neg_pairs[b, i] and pos_pairs[a, j] > neg_pairs[b, j]:
                            self.s[j, i] += 1
                        elif self.with_ties:
                            self.s[i, j] += 1
                            self.s[j, i] += 1
                            T += 1
                            self.Ts[i, j] += 1
                            self.Ts[j, i] += 1

        return T


    def _fit_indexes(self, features, labels, weights=None):
        labels = np.array(labels)
        # experiment_num = sum([g[0] for g in self.groups])
        all_features = np.array(range(features.shape[1]))
        # pos_pairs = features[np.ix_(labels == 1, [True] * features.shape[1])]
        # neg_pairs = features[np.ix_(labels == 0, [True] * features.shape[1])]
        pos_pairs = features[np.array(labels) == 1]
        pos_pairs = pos_pairs[np.random.randint(0, pos_pairs.shape[0], size=2 * self.sample_n//features.shape[1])]
        neg_pairs = features[np.array(labels) == 0]
        neg_pairs = neg_pairs[np.random.randint(0, neg_pairs.shape[0], size=2 * self.sample_n//features.shape[1])]
        # f_weights = {i: 1 for i in range(features.shape[1])}

        for group_n, select_n  in self.groups:
            groups = self._get_groups(all_features, group_n)
            all_features = []
            for g in groups:
                tmp_pos_pairs = pos_pairs[:, g]
                tmp_neg_pairs = neg_pairs[:, g]

                self.features_n = len(g)
                T = self._make_s(tmp_pos_pairs, tmp_neg_pairs)
                self.T = max(1, T)
                self.main()
                print(self.teta)
                all_features += g[self.gammas.argsort()[::-1][:select_n]].tolist()
                # for i in self.gammas.argsort()[::-1][:select_n]:
                #     f_weights[g[i]] = 1/(i+1)#self.gammas[i]
                print(self.gammas, g[self.gammas.argsort()[::-1]])
            all_features = np.array(all_features)
        # all_features = g
        self.best = all_features

    def fit(self, pos_pairs, neg_pairs, weights=None):
        # print(pos_pairs)
        global fit_idx
        fit_idx += 1
        print(f"{fit_idx} fit: samples_n={self.sample_n}, weights = {weights}")
        self.features_n = pos_pairs.shape[1]
        # samples_num = pos_pairs.shape[0]
        if weights is not None:
            mask1 = np.random.randint(0, pos_pairs.shape[0], size=pos_pairs.shape[0]) <= 2 * self.sample_n//self.features_n
            mask2 = np.random.randint(0, pos_pairs.shape[0], size=neg_pairs.shape[0]) <= 2 * self.sample_n//self.features_n
            mask1[0] = True
            mask2[0] = True
            T = self._make_s(pos_pairs[mask1],
                             neg_pairs[mask2])
        else:
            T = self._make_s(pos_pairs, pos_pairs)
        self.T = max(1, T)
        self.main()
        self.best = self.gammas.argsort()[::-1]#[-1:-4:-1]
        if weights is not None:
            self.log_regressors = []
            mask1 = np.random.randint(0, pos_pairs.shape[0], size=pos_pairs.shape[0]) < 2 * self.sample_n//self.features_n
            mask2 = np.random.randint(0, pos_pairs.shape[0], size=neg_pairs.shape[0]) < 2 * self.sample_n//self.features_n
            mask1[0] = True
            mask2[0] = True
            for idx in self.best:
                clf = LogisticRegression(class_weight=weights)
                clf.fit(np.hstack([pos_pairs[mask1, idx], neg_pairs[mask2, idx]]).reshape((-1, 1)),
                        [1]*(mask1.sum()) + [0]*(mask2.sum()))
                self.log_regressors.append(clf)


    def decision_function(self, test, use_regressors = False):
        # test = -test
        # mask = scipy.sparse.random(1000000, 44, density=0.1, dtype=bool)
        # test = test.multiply(mask)
        global predict_idx
        predict_idx += 1
        print(f"{predict_idx} fit: test.len={len(test)}, use_reg={use_regressors}")

        if use_regressors:
            normed = np.vstack([self.log_regressors[i].decision_function(test[:, idx].reshape(-1, 1)) for i, idx in enumerate(self.best)]).transpose()
            gammas = np.array(self.gammas)[self.best]
            scores = normed*gammas
            score = scores.sum(axis=1)
        else:
            score = test[:, self.best[0]]
        return score

    def set_params(self, **kwargs):
        self.sample_n = kwargs['sample_n']
        # self.density = kwargs['density']
        # self.directory = kwargs['dir']

    def f_gama(self, i):
        numer = 0
        denom = 0
        for j in range(self.features_n):
            if (i != j):
                numer += self.s[i, j]
                denom += self.s[i, j] /(self.gammas[i] + self.teta * self.gammas[j]) + \
                         self.teta * self.s[j, i] / (self.teta * self.gammas[i] + self.gammas[j])
        return numer / denom

    def f_c(self):
        sum = 0
        for i in range(self.features_n):
            for j in range(self.features_n):
                if (i != j):
                    sum += self.gammas[j] * self.s[i, j] / (self.gammas[i] + self.teta * self.gammas[j])
        return 2 * sum / self.T


    def f_teta(self):
        # return 1 / (2 * self.C) + sqrt(1 + 1 / (self.C * self.C) / 4)
        return 1


    def diff(self):
        sum = 0
        for i in range(self.features_n):
            sum += abs(self.gammas_tmp[i] - self.gammas[i])
        return sum

    def main(self):
        it = 0
        self.teta = 0
        self.C = 1
        self.gammas = np.zeros(self.features_n)
        self.gammas.fill(1/self.features_n)
        self.gammas_tmp = np.zeros(self.features_n)
        #std::cout << "***INIT***"
	    #show("gamma_0", gama, FEATURE_NUM)
        while it < 3 or (it < self.MAXIT and self.diff() > self.EPS):
            it = it + 1
        #std::cout << "***Iteration " << it << "***" << std::endl
            gamas_sum = 0
    #update gamas
            for i in range(self.features_n):
                gama_i = self.f_gama(i)
                gamas_sum = gamas_sum + gama_i
                self.gammas_tmp[i] = gama_i #normalize gammas
            for i in range(self.features_n):
                self.gammas_tmp[i] /= gamas_sum
            #count C_k
            self.C = self.f_c()
        #count teta_k
            self.teta = self.f_teta()
            self.gammas, self.gammas_tmp = self.gammas_tmp, self.gammas

        np.nan_to_num(self.gammas, copy=False, nan=1/self.features_n)
        #swap new gama
		#show("gamma_" + std::to_string(it), gama, FEATURE_NUM)
		#show("gamma_tmp_" + std::to_string(it), gama_tmp, FEATURE_NUM)

		#std::cout << "teta_" << it << "=" << teta << std::endl
		#std::cout << "C_" << it - 1 << "=" << C << std::endl
		#std::cout << "DIFF ="<< diff() << std::endl
    def _generate(self, features, n):
        return np.stack([features[np.random.randint(0, features.shape[0], size = n), i]
                         for i in range(features.shape[1])], axis = 1)
