# time ---------------->
# sample_num .....0.....|...1...|...2...|    |.val_N.|...-1...|...-2...|     |...-test_N...|
# for        ...train...|.......validate.............|..............test...................|
# test[i] ~~ samples['sample_num'] == i
# if -timestamp:
#       train[i] ~~ samples['sample_num'] > i
# else:
#       train[i] ~~samles['sample_num'] != i


import sys, os, argparse
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from features_debug import calculate_link_features, get_network, label_independent_features, node2vec_features
from common import save_obj, load_obj
import pandas as pd
import numpy as np
import scipy
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from BT import BT


import math

ks = [1, 5, 10, 100, 1000]#, 1000]

NormMethods = ('autoencoder', 'KNND')
OneClassMethods = ("OCSVM", "IF", "LOF", "KNND", "autoencoder".upper())
TwoClassMethods = ("MCSVM", "RF", "GB", "BT")
WithSelection = ("OCSVM", "LOF","IF", "KNND", "autoencoder".upper(), "MCSVM")
BayesParams = {"OCSVM":{"gamma": (0.00316228, 0.56234133), 'nu': (1e-4, 3.16227766e-01)},
               "MCSVM":{"gamma": (0.00316228, 0.56234133)},
               "GB":{'n_estimators': (10, 200)},
               "RF":{'n_estimators': (10, 200)},
               "IF":{'n_estimators': (100, 200), 'contamination': (1e-4, 3.16227766e-01), "max_samples": (200, 10000)},
               "AUTOENCODER":{'code_size': (2, 5), 'intermediate_dim': (0.25, 0.75)},
               "KNND":{'k': (100, 5000)},
               "BT" : {"pass":(0, 1)}}

GridParams =  {"OCSVM":{"kernel": ["rbf"], "gamma": ["auto"] + list(np.logspace(-2.5, -0.25, num=5, endpoint=True)),
                    'nu': list(np.logspace(-4, -0.5, num=7, endpoint=True)), "cache_size": [4096]},
               "MCSVM":{"kernel": ["rbf"], "gamma": ["auto"] + list(np.logspace(-2.5, -0.25, num=5, endpoint=True)),
                    "cache_size": [4096]},
               "GB":{'n_estimators': [10, 20, 50, 100, 200]},
               "RF":{'n_estimators': [10, 20, 50, 100, 200]},
               "IF":{'n_estimators': [100, 200, 500], "random_state": [1],
                     "contamination": list(np.logspace(-2.5, -0.25, num=5, endpoint=True)),"behaviour": ['new'],
                     "max_samples": [200, 1000, 5000, 10000]},
               "AUTOENCODER":{'code_size': [3, 4, 5], 'intermediate_dim': [0.25, 0.5, 0.75]},
               "KNND":{'k': [[100, 250, 500, 1000]]},
               #"BT": {"sample_n": [5000]}}
"BT": {"sample_n": list(range(100, 10000, 200))}}


sample_weights = None
week_sec = 2*3600*24



def select_threshold(probas):
    from scipy.optimize import minimize_scalar
    from scipy.stats import ks_2samp
    def f(x):
        return ks_2samp(probas[probas > x], probas [probas <=x]).pvalue
    res = minimize_scalar(f, bounds=(probas.min(), probas.max()), method='bounded')
    return res.x


def ndcg_score(labels, probas, with_threshold = True):
    y_true = np.array(labels)
    #y_true[y_true == 0] = -1
    rels = np.array(probas)

    if with_threshold:
        threshold = select_threshold(rels)
        y_true = y_true[rels >= threshold]
        rels = rels[rels >= threshold]

    print("selected len=", len(y_true), len(probas))
    discount = 1 / (np.log(np.arange(y_true.shape[0]) + 2) / np.log(2))
    # print("discount)shape=", discount.shape)
    def dcg(y_true, rels, K):
        y_sorted = y_true[rels.argsort()[::-1][:K]]

        return (y_sorted * discount[:K]).sum()
    res = []
    points = list(np.linspace(1, y_true.shape[0], 1000, endpoint=True, dtype=int))
    for k in points:
        res.append(dcg(y_true, rels, k) / dcg(y_true, y_true, k))

    return dcg(y_true, rels, y_true.shape[0]) / dcg(y_true, y_true, y_true.shape[0]), np.array(points)


def init_classifier(clf):
    if clf == "OCSVM":
        classifier = OneClassSVM(kernel='rbf', cache_size=4096)

    elif clf == "MCSVM":
        classifier = SVC(kernel='rbf',cache_size=4096)

    elif clf == "GB":
        classifier = GradientBoostingClassifier(random_state=1)
    elif clf == "RF":
        classifier = RandomForestClassifier(random_state=1)
    elif clf == 'IF':
        classifier = IsolationForest(random_state=1, contamination='auto')
    elif clf == "autoencoder".upper():
        classifier = None
    elif clf == "KNND":
        classifier = None
    elif clf == "BT":
        classifier = BT(sample_n = 1000)
    else:
        raise Exception("Bad classifier")
    if clf in OneClassMethods:
        generator = oc_gen
    elif clf in TwoClassMethods:
        generator = mc_gen
    return classifier, generator


def init_params(clf, method):
    if method.lower() == 'bayes':
        return BayesParams[clf]
    elif method.lower() == 'grid':
        return GridParams[clf]
    else:
        raise Exception("Bad classifier")


def coverage(labels, probas):
    labels = np.array(labels)
    probas = np.array(probas)
    index = probas.argsort()[::-1]
    return (max(index[labels == 1]) + 1)/labels.sum()


def map_k(labels,probas,ks = (1, 5, 10, 100, 1000)):
    def ap(x, k):
        s = 0
        prev_sum = 0
        for i in range(k):
            prev_sum = prev_sum + x[i]
            s = s + prev_sum/(i+1)*x[i]
        return s/k
    labels = np.array(labels)
    probas = np.array(probas)
    y = labels[probas.argsort()[::-1]]
    aps = []
    for k in ks:
        if k > labels.sum():
            print("too big k, only %d posetive available"%labels.sum())
            break
        tmp = ap(y, k)
        aps.append(tmp)
    return np.array(aps), sum(aps)/len(aps)


def select_features(X, y, default_feature = -1, select_method ="GB"):
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    validation_num = 10
    if select_method == "GB":
        n_estimators = [200]
        if len(n_estimators) == 1:
            classifier = GradientBoostingClassifier(random_state=1, n_estimators=n_estimators[0],
                                                    max_features='sqrt')

    probas = np.random.rand(X.shape[0]) <= 0.5
    classifier.fit(X[probas], y[probas])
    num_of_best_feature = len(list(classifier.feature_importances_[classifier.feature_importances_ > 0.0]))
    return list(np.argsort(classifier.feature_importances_))[::-1][
                   :(num_of_best_feature if default_feature < 0 else min(num_of_best_feature, default_feature))]


def oc_gen(generator, begin = 0, end = None, selected_features = None, x_train_value = 1, select_proba = 1):
    i = -1
    for df_train, df_test, info_df, columns in generator:
        i = i + 1
        if i < begin:
            continue
        if end is not None:
            if i >= end:
                return
        # freq = df_train[df_train['has_link'] == x_train_value]['freq']
        x_train = np.array(df_train.loc[df_train['has_link'] == x_train_value].drop(columns = ["has_link", 'freq']).values)
        x_test = np.array(df_test.drop(columns = ["has_link", 'freq']).values)
        y_test = df_test["has_link"].values
        if selected_features is not None:
            list_feature = selected_features[i]
            #print("selected %d features" %len(list_feature))

            x_train = x_train[:, list_feature]
            x_test = x_test[:, list_feature]
            #print("Train shape before sampling:", x_train.shape)
        if select_proba < 1:
            probas = np.random.rand(x_train.shape[0]) <= select_proba
            x_train = x_train[np.ix_(probas, [True]*x_train.shape[1])]
        yield x_train, None, x_test, list(y_test), info_df, columns


def mc_gen(generator, begin = 0, end = None, selected_features = None, select_proba = 1):
    i = -1
    for df_train, df_test, info_df, columns in generator:
        i = i + 1
        if i < begin:
            continue
        if end is not None:
            if i >= end:
                return
        print(df_train.drop(columns = ["has_link", 'freq']).columns)
        print(df_test.drop(columns = ["has_link", 'freq']).columns)
        x_train = np.array(df_train.drop(columns = ["has_link", 'freq']).values)
        y_train = np.array(df_train['has_link'].values)
        x_test =np.array(df_test.drop(columns = ["has_link", 'freq']).values)
        y_test = np.array(df_test["has_link"])
        if selected_features is not None:
            list_feature = selected_features[i]
            #print("selected %d features" %len(list_feature))
            x_train = x_train[:, list_feature]
            x_test = x_test[:, list_feature]
        if select_proba < 1:
            probas = np.random.rand(x_train.shape[0])<= select_proba
            x_train = x_train[np.ix_(probas, [True]*x_train.shape[1])]
            y_train = y_train[probas]
        yield x_train, list(y_train), x_test, list(y_test), info_df, columns


def significant_features(generator, feature_ns, select_method='GB'):
    res = dict()
    for n in feature_ns:
        res.setdefault(n, [])
    i = 0
    for df_train, df_test, _ in generator:
        x_train = df_train.drop(columns = ["has_link", "freq"])
        y_train = df_train['has_link']
        list_feature = select_features(np.array(x_train.values),
                                       y_train.values, select_method=select_method)
        for n in feature_ns:
            if n == -1:
                res[n].append(list_feature[0:])
            else:
                res[n].append(list_feature[:min(len(list_feature), n)])
        print("Sample " + str(i) + ": " + str(x_train.columns[list_feature]))
        i = i+1
    return res


def autoencoder_experiment(samples, grid):
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from keras.layers import Dense, Input
    from keras.models import Model, load_model
    from keras import regularizers

    code_size = grid.get("code_size", 5)
    batch_size = grid.get("batch_size", 32)
    nb_epoch = grid.get("n_epoch", 500)
    intermediate_part = grid.get("intermediate_dim", 0.5)
    auc_score = 0.0
    aps = np.zeros(len(ks))
    mapk = 0
    covers = 0
    real_n = 0
    normal_value = 1
    # model_name = str(time()).replace(".", "") + ".h5"
    for x_train, _, x_test, y_test in samples:

        real_n = real_n + 1
        input_dim = len(x_train[0])
        intermediate_dim = int(round(intermediate_part*input_dim))
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(intermediate_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
        code = Dense(code_size, activation=None)(encoder)
        decoder = Dense(intermediate_dim, activation='tanh')(code)
        decoder = Dense(input_dim, activation=None)(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adamax',
                           loss='mean_squared_error',
                           metrics=['mse'])


        earlystop = EarlyStopping(monitor='val_loss', patience=5)
        y_test = np.array(y_test)
        x_val = x_test[y_test==normal_value][::2]
        # print(x_test.shape)
        history = autoencoder.fit(x_train, x_train,
                                  epochs=nb_epoch,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  # steps_per_epoch = len(x_train)//batch_size//2,
                                  validation_data=(x_val, x_val),
                                  # validation_steps = len(x_val)//batch_size,
                                  verbose=0,
                                  # use_multiprocessing=True,
                                  callbacks=[earlystop])

        predictions = autoencoder.predict(x_test)
        mse = np.mean(np.power(x_test - predictions, 2), axis=1)
        _probas = (mse - mse.min())/(mse.max()-mse.min())
        _probas = 1 - _probas
        # fpr, tpr, thresholds = roc_curve(y_test, _probas)
        roc_auc = roc_auc_score(y_test, _probas)

        auc_score = auc_score + roc_auc
        a, b = map_k(y_test, _probas, ks)
        c = coverage(y_test, _probas)
        aps = aps + a
        mapk = mapk + b
        covers = covers + c

    mean_aps = aps/real_n
    mean_mapk = mapk/real_n
    mean_covers = covers/real_n
    mean_auc = auc_score/real_n

    return mean_mapk, mean_auc, mean_covers, tuple(mean_aps)


def test_clf(clf, classifier, grid, samples, ret_curve = False, verbose=0, save_res = False, all_features_auc = False):
    # from scipy import interp

    tprs = []
    fprs = []

    if clf == "autoencoder".upper():
        return autoencoder_experiment(samples, grid)
    if clf == 'KNND':
        return fast_knnd(samples, grid)
    classifier.set_params(**grid)
    auc_score = 0.0
    covers = 0
    aps = np.zeros(len(ks))
    mapk = 0
    ndcg = None
    n = 0
    saved_res = None
    for x_train, y_train, x_test, y_test, info, columns in samples:
        #print(x_train.shape, x_test.shape)
        n = n + 1
        # print("test %d"%n)
        if clf == "OCSVM":
            classifier.fit(x_train)
            _probas = [1.0 / (1 + np.exp(-pr)) for pr in classifier.decision_function(x_test)]
        elif clf == "MCSVM":
            classifier.fit(x_train, list(y_train))
            _probas = [1.0 / (1 + np.exp(-pr)) for pr in classifier.decision_function(x_test)]
        elif clf == "RF" or clf == "GB":
            classifier.fit(x_train, list(y_train))
            _probas = classifier.predict_proba(x_test)[:, 1]
        elif clf == 'IF':
            classifier.fit(x_train, sample_weight=sample_weights)
            _probas = classifier.score_samples(x_test)
        elif clf == 'LOF':
            classifier.fit(x_train)
            _probas = classifier.decision_function(x_test)
        elif clf == "KNND":
            classifier.fit(x_train)
            _probas= 1-classifier.decision_function(x_test)
        elif clf == "BT":
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            classifier.fit(x_train[y_train == 1], x_train[y_train == 0], weights={0:1, 1:1.979918791920446e-05})
            #TODO
            print("NEED %s NEED" % columns[classifier.best[0]])
            # density = 0.01
            # x_test  = np.array(x_test)
            # mask = scipy.sparse.random(1000000, 44, density=0.1, dtype=bool)
            _probas = classifier.decision_function(x_test, use_regressors=True)
        if save_res:
            info['probas'] = _probas
            if saved_res is None:
                saved_res = info
            else:
                saved_res = pd.concat([saved_res, info], axis = 0)
        fpr, tpr, thresholds = roc_curve(y_test, _probas)
        tprs.append(tpr)
        tprs.append(tpr)
        roc_auc = roc_auc_score(y_test, _probas)
        auc_score = auc_score + roc_auc
        a, b = map_k(y_test, _probas, ks)
        aps = aps + a
        mapk = mapk + b
        c = ndcg_score(y_test, _probas)[0]
        if ndcg is None:
            ndcg = c
        else:
            ndcg = ndcg + c
        covers = covers + coverage(y_test, _probas)
        if verbose==1:
            print('sample %d: auc = %lf, mapk = %lf, aps=%s, ndcg=' %(n, roc_auc, b, str(a)), c)
            if all_features_auc:
                print({k: roc_auc_score(y_test, x_test[:, k]) for k in range(x_test.shape[1])})
    mean_auc = auc_score / n
    mean_aps = aps / n
    mean_mapk = mapk / n
    mean_covers = covers / n
    mean_ndcg = ndcg / n
    if save_res:
        return mean_mapk, mean_auc, mean_ndcg, mean_covers, tuple(mean_aps), saved_res
    return mean_mapk, mean_auc, mean_ndcg, mean_covers, tuple(mean_aps)


def fast_knnd(samples, K):
    K = K['k']
    if hasattr(K, "__iter__"):
        last = max(K) + 1
    else:
        last = K + 1
        K = [K]
    auc_score = dict.fromkeys(K, 0)
    covers = dict.fromkeys(K, 0)
    aps = dict.fromkeys(K, np.zeros(len(ks)))
    mapk = dict.fromkeys(K, 0)
    n = 0
    for x_train, _, x_test, y_test, _ in samples:
        n = n+1
        probas_ = 1-np.array([np.sort(np.partition(np.linalg.norm(x_train - t, axis=1), last)[:last])[K]
                     for t in x_test])
        for idx, k in enumerate(K):
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, idx])
            roc_auc = roc_auc_score(y_test, probas_[:, idx])
            auc_score[k] = auc_score[k] + roc_auc
            a, b = map_k(y_test, probas_[:,idx], ks)
            aps[k] = aps[k] + a
            mapk[k] = mapk[k] + b
            covers[k] = covers[k] + coverage(y_test, probas_[:, idx])
    for idx, k in enumerate(K):
        auc_score[k] = auc_score[k] / n
        aps[k] = aps[k] / n
        mapk[k] = mapk[k] / n
        covers[k] = covers[k] / n
    res = list([(mapk[k], auc_score[k], covers[k], aps[k]) for k in K])
    if len(res) == 1:
        return res[0]
    else:
        return res


def prepare_params(params):
    feature_n = int(round(params.pop('feature_num', 0)))
    for name in ('n_estimators', 'code_size', 'k', "max_samples"):
        if params.get(name, None):
            params[name] = int(round(params[name]))
    return feature_n, params


def make_function(clf, classifier, generator, selected_features, sample_proba, score):
    from time import time
    def f(**kwargs):
        t0 = time()
        feature_n, kwargs = prepare_params(kwargs)
        if clf == "KNND":
            kwargs['k'] = [kwargs['k']]
        res = test_clf(clf, classifier, kwargs, generator(sample_generator(samples),
                                                  begin=-samples['sample_num'].min(), end=samples['sample_num'].max()-samples['sample_num'].min(),
                                                  selected_features=[x[:feature_n] for x in selected_features]
                                                  if selected_features is not None else None,
                                                  select_proba=sample_proba))
        print("Time: %lf" %(time() - t0))
        if score == "mapk":
            return res[0]
        elif score == "auc":
            return res[1]
        elif score == "ndcg":
            return res[2]
        else:
            raise Exception("Bad score function")
    return f


def bayes_experiment(clf, samples, samples_num, select_method, sample_proba = 1, iters = 10, save_file = None):
    from bayes_opt import BayesianOptimization

    classifier, generator = init_classifier(clf)
    params = init_params(clf, 'bayes')

    if clf in WithSelection:
        feature_ns = [-1]
        selected_features = significant_features(sample_generator(samples),
                                                 select_method=select_method,
                                                 feature_ns=feature_ns)[-1]
        params["feature_num"] = (1, min((len(x) for x in selected_features)))
    else:
        selected_features = None

    for score_func in ['auc', 'mapk']:
        print("*" *10 + score_func.upper() + " OPTIMIZATION" + "*"*10)
        optimizer = BayesianOptimization(
            f=make_function(clf, classifier, generator, selected_features, sample_proba, score_func),
            pbounds=params,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=2,
        )
        optimizer.maximize(
            init_points=3,
            n_iter=15,
        )
        best_params = optimizer.max['params']
        feature_n, best_params = prepare_params(best_params)
        print(best_params, "feature_n = %d"%feature_n)

        res = test_clf(clf, classifier, best_params, generator(sample_generator(samples),
                                                      begin=0, end=-samples['sample_num'].min(),
                                                      selected_features=[x[:feature_n] for x in selected_features]
                                                      if selected_features is not None else None,
                                                      select_proba=sample_proba), verbose=1, save_res=save_file is not None)
        if save_file is not None:
            save_obj(res[-1], save_file + "_" + score_func)
            res = res[:-1]
        print("BEST: ", res)
        print("*"*10 + "AVERAGE_RES" + "*"*10)
        print('avg=', np.average(np.array([r['target'] for r in optimizer.res])),
              " std=", np.std(np.array([r['target'] for r in optimizer.res])))
    return 0


def grid_experiment(clf, samples, samples_num, select_method ="GB", sample_proba = 1):
    from sklearn.model_selection import ParameterGrid
    from time import time
    import itertools

    classifier, generator = init_classifier(clf)
    params = init_params(clf, 'grid')

    if clf in WithSelection:
        feature_ns = [1, 3, 5, 7, 10, 20, -1]
        selected_features = significant_features(sample_generator(samples),
                                                 select_method=select_method,
                                                 feature_ns=feature_ns)
    else:
        feature_ns = [None]
        selected_features = None

    # grid_result = {} #(grid, feature_n):(mapk, auc, coverage)
    #
    # for grid, feature_n in itertools.product(ParameterGrid(params), feature_ns):
    #     t0 = time()
    #     result = test_clf(clf, classifier, grid, generator(sample_generator(samples),
    #                                                 begin=-samples['sample_num'].min(), end=samples['sample_num'].max() - samples['sample_num'].min(),
    #                                                   selected_features=selected_features[feature_n]
    #                                                   if selected_features is not None else None,
    #                                                   select_proba=sample_proba))
    #     if clf == 'KNND':
    #         for idx, i in enumerate(grid['k']):
    #             grid_result.update({(frozenset({'k':i}.items()), feature_n): result[idx]})
    #     else:
    #         grid_result.update({(frozenset(grid.items()), feature_n): result})
    #     print("Params: feature_n=" + str(feature_n) + ", " + str(grid) + "\nTime %lf"%(time() - t0))
    #
    # print("*"*20 + "RESULTS" + "*"*20)
    # print(grid_result)
    #
    # for idx, m in enumerate(["mapk", "auc"]):
    #     best_grid, best_feature_n = max(grid_result, key = lambda x: grid_result.get(x)[idx])
    #     best_grid = dict(best_grid)
    #     result = test_clf(clf, classifier, best_grid, generator(sample_generator(samples),
    #                                                             begin=0, end=-samples['sample_num'].min(),
    #                                                        selected_features=selected_features[best_feature_n]
    #                                                        if selected_features is not None else None))
    #     print("*" * 20 + "BEST_" + m.upper() + "*" * 20)
    #     print("Best params: %s, feature_n = %d" % (str(best_grid), best_feature_n if best_feature_n is not None else -1))
    #     print("%s = %f"%(m.upper(), result[idx]))
    #     if m == "mapk":
    #         print("aps = %s" % str(result[-1]))

    print("*" * 20 + "AVERAGE ON ALL PARAMETER GRID" + "*" * 20)

    grid_result = {} #(grid, feature_n):(mapk, auc, coverage)
    import random
    random.seed(1)
    for grid, feature_n in itertools.product(ParameterGrid(params), feature_ns):
        t0 = time()
        mean_res = None#np.zeros(shape = (4, ))
        for i in range(10):
        # result = test_clf(clf, classifier, grid, generator(sample_generator(samples),
        #                                                 begin = 0, end=1, #-samples['sample_num'].min(),
        #                                               selected_features=selected_features[feature_n]
        #                                               if selected_features is not None else None,
        #                                               select_proba=sample_proba), verbose=1, all_features_auc=True)
            result = test_clf(clf, classifier, grid, generator(sample_generator(samples),
                                                            begin = 1, end=2, #-samples['sample_num'].min(),
                                                          selected_features=selected_features[feature_n]
                                                          if selected_features is not None else None,
                                                          select_proba=sample_proba), verbose=1, all_features_auc=True)[:-1]
            result = np.array(result)
            if mean_res is None:
                mean_res = result
            else:
                mean_res += result
        result = mean_res/10
        if clf == 'KNND':
            for idx, i in enumerate(grid['k']):
                grid_result.update({(frozenset({'k':i}.items()), feature_n): result[idx]})
        else:
            grid_result.update({(frozenset(grid.items()), feature_n): result})
        # print("Params: feature_n=" + str(feature_n) + ", " + str(grid) + "\nTime %lf"%(time() - t0))
    if clf == "BT":
        print(classifier.history)
    print(grid_result)
    for idx, m in enumerate(["mapk", "auc"]):
        results =np.array([grid_result.get(x)[idx] for x in grid_result])
        print("%s: avg = %f, std = %f" % (m.upper(), np.mean(results), np.std(results)))
    return 0


def _link_features_from_users(user_features, pairs, raw=False):
    features = list(user_features.columns)
    link_features = pd.DataFrame(pairs, columns=['left', 'right'])
    link_features = link_features.join(user_features.add_suffix("_left"), on='left')
    link_features = link_features.join(user_features.add_suffix("_right"), on='right')
    if not raw:
        for feature in features:
            link_features[feature + "_mean"] = (link_features[feature + "_right"] + link_features[feature + "_left"])/2
            link_features[feature + "_abs"] = (link_features[feature + "_right"] - link_features[feature + "_left"]).abs()
            link_features.drop([feature + "_right", feature + "_left"], axis=1, inplace=True)
    return link_features.set_index(['left', 'right'], verify_integrity=True)




def get_features(links, socialnet, env_dir, pairs, remove_list=None, type='ours', max_timestamp = None):
    import networkx as nx
    import pandas as pd
    filenames = [os.path.join(env_dir, socialnet + "_" + y) for y in links]
    filenames = [csv_file for csv_file in filenames if os.path.exists(csv_file + '.csv')]
    valid_links = [link for link in links if os.path.exists(os.path.join(env_dir, socialnet + "_" + link + '.csv'))]
    Graphs = []

    remove_set = set(remove_list)

    for csv_file in filenames:
        df = pd.read_csv(csv_file + '.csv', sep=';', error_bad_lines=True, index_col=False)
        if max_timestamp is not None:
            df = df[df['time'] <= max_timestamp]
            df = df.drop(columns='time')
            df = df.groupby(by=['left', 'right']).size().reset_index()
            df.rename(columns={0:'Weight'}, inplace=True)

        # dtype={"left":'string', "right":'string', "Weight":"float"})
        df["Weight_reversed"] = 1 / df["Weight"]

        if remove_list:
            indexes_for_remove = set()

            for row in df.itertuples():
                if (row.left, row.right) in remove_set or (row.right, row.left) in remove_set:
                    indexes_for_remove.add(row.Index)

            df.drop(indexes_for_remove, axis=0, inplace=True)

        G = nx.from_pandas_edgelist(df,
                                    source='left',
                                    target='right',
                                    edge_attr=['Weight', 'Weight_reversed'],
                                    create_using=nx.DiGraph())

        print("*****Graph statistic*****")
        print("Nodes: %d\nEdges: %d" % (G.number_of_nodes(), G.number_of_edges()))
        Graphs.append(G)
        del df

    p = Pool(len(Graphs))
    if type=='ours' or type=="all":

        user_features_res = p.starmap(label_independent_features, zip([None] * len(Graphs), valid_links,
                                                                      [None] * len(Graphs), [False] * len(Graphs),
                                                                      Graphs))
        user_features_res = [x for x in user_features_res if x is not None]
        if len(user_features_res) == 0:
            return None
        user_features = user_features_res[0]
        for x in user_features_res[1:]:
            user_features = user_features.join(x, how='outer')
        user_features.fillna(0.0, inplace=True)
        user_features = _link_features_from_users(user_features, pairs)
        # calculate_link_features(filename, prefix, pairs, parallel=False):
        print("after indeoendent user_features ",user_features.shape)
        del user_features_res
        p.close()
        p.join()

        p = ThreadPool(len(Graphs))
        print(len(pairs))
        link_features_res = p.starmap(calculate_link_features, zip(Graphs, [link + "_" for link in valid_links],
                                                         [pairs] * len(Graphs), [True] * len(Graphs)))
        # link_features_res = calculate_link_features(Graphs[0], "all_",pairs,True)
        print(len(Graphs))
        del Graphs
        for x in link_features_res:
            print("link features shape ", x.shape)
        link_features_res = [x for x in link_features_res if x is not None]
        if len(link_features_res) == 0:
            return None
        link_features = link_features_res[0]
        for x in link_features_res[1:]:
            link_features = link_features.join(x, how = 'outer')
        p.close()
        p.join()
        print("user features ",user_features.shape)


        del link_features_res

        print("join link_features ", link_features.shape, " and user_features", user_features.shape)
        save_obj(link_features, "link_features")
        save_obj(user_features, "user_features")
        res = user_features.join(link_features)
    if type == 'node2vec' or type=='all':
        p = Pool(len(Graphs))
        embeddings = p.starmap(node2vec_features, zip(Graphs, [link + "_" for link in valid_links]))
        del Graphs
        embeddings = [x for x in embeddings if x is not None]
        user_features = embeddings[0]
        for x in embeddings[1:]:
            user_features = user_features.join(x, how='outer')
        res1 = _link_features_from_users(user_features, pairs, raw=True)
        if type=='all':
            res = res.join(res1)
        else:
            res = res1
        del embeddings
        p.close()
        p.join()


    #def label_independent_features(env_id, link, filename, save_paths=True, G=None):
    return res


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s_f')
    parser.add_argument('-save', default=None)
    parser.add_argument('-s_n', type=int, default=10)
    parser.add_argument('-env_dir')
    parser.add_argument('-clf', default="OCSVM", choices = ["OCSVM", "MCSVM", "GB", "RF", "IF", "LOF", "KNND", "autoencoder", "BT"])
    parser.add_argument('-select', default='GB')
    parser.add_argument('-metric', default='mapk', choices = ["mapk", 'auc'])
    parser.add_argument('-norm', action = 'store_true')
    parser.add_argument('-timestamp', action='store_true')
    parser.add_argument('-remove_new', action='store_true')
    parser.add_argument('-code', type=int, default=5)
    parser.add_argument('-sample_proba', type=float, default=1)
    parser.add_argument('-bayes', action='store_true')
    parser.add_argument('-simple', action='store_true')
    parser.add_argument('-features', default='ours', choices=["node2vec", "ours", 'all', 'individual'])
    parser.add_argument('-user', type=int, default=-1)
    return parser




def get_neg_links(links_samples_df, distribution, i, already_negative, user_id=None):
    from random import choice
    import itertools

    already_negative = set(already_negative)
    if i == 0:
        pos_pairs = list(links_samples_df[distribution == 0][["left", "right"]].itertuples(index=False))
    elif i > 0:
        pos_pairs = list(
            links_samples_df[(distribution < i) & (distribution >= 0)][["left", "right"]].itertuples(index=False))
    else:
        pos_pairs = list(links_samples_df[distribution > i][["left", "right"]].itertuples(index=False))

    pos_pairs = set(map(lambda x: tuple(x), pos_pairs))
    nodes = list(set(itertools.chain(*pos_pairs)))
    all_pos_pairs = set(map(lambda x: tuple(x), list(links_samples_df[['left', 'right']].itertuples(index=False))))
    test_df = links_samples_df[distribution == i]
    num = (test_df['left'].isin(nodes) & test_df['right'].isin(nodes)).sum()
    print("%d posetive at %d (with removing new - %d)"%(len(pos_pairs), i, num))
    cur_no_links = set()
    if user_id is not None:
        while len(cur_no_links) < num:
            pair = (choice(user_id), choice(nodes))
            if (pair not in all_pos_pairs) and pair[0] != pair[1] and \
                    ((pair[1], pair[0]) not in all_pos_pairs) and ((pair[1], pair[0]) not in cur_no_links) and \
                (pair not in already_negative) and ((pair[1], pair[0]) not in already_negative):
                cur_no_links.add(pair)
    else:
        while len(cur_no_links) < num:
            pair = (choice(nodes), choice(nodes))
            if (pair not in all_pos_pairs) and pair[0] != pair[1] and \
                    ((pair[1], pair[0]) not in all_pos_pairs) and ((pair[1], pair[0]) not in cur_no_links) and \
                    (pair not in already_negative) and ((pair[1], pair[0]) not in already_negative):
                cur_no_links.add(pair)
    return list(cur_no_links)



def get_samples(env_dir, samples_num, samples_path, train_path=None, test_path=None, with_timestamp = False, type='ours', user_id=None):
    from random import choice
    import networkx as nx
    import itertools

    graph_file = os.path.join(env_dir, "VK_time")
    distribution = None
    if with_timestamp:
        links_samples_df = pd.read_csv(graph_file + ".csv", sep=';', error_bad_lines=True, index_col=False)
        print(len(links_samples_df))
        links_samples_df.sort_values(by='time', inplace=True)
        if user_id is not None:
            only_one_df = links_samples_df[(np.array(links_samples_df['left'].isin(user_id)) | np.array(links_samples_df['right'].isin(user_id)))
            & ~(np.array(links_samples_df['left'].isin(user_id)) & np.array(links_samples_df['right'].isin(user_id)))]

            links_samples_df = pd.concat([links_samples_df[(np.array(links_samples_df['left'].isin(user_id)) & np.array(links_samples_df['right'].isin(user_id)))],
                                          only_one_df[only_one_df['left'].isin(user_id)],
                                          only_one_df[only_one_df['right'].isin(user_id)].rename({'right':'left', 'left':'right'})])

        timestamps = np.array(links_samples_df['time'])
        max_time = timestamps[-1]
        test_week = int((max_time - timestamps[int(0.75*len(links_samples_df))])//week_sec)
        validate_week = int((timestamps[int(0.75*len(links_samples_df))] - timestamps[int(0.6*len(links_samples_df))])//week_sec)
        distribution = np.array([0]*len(links_samples_df))
        test_week = 7
        validate_week = 5
        for i in range(test_week):
            end = max_time - i*week_sec
            start = max_time - (1 + i)*week_sec
            distribution[(timestamps <= end) & (timestamps > start)] = -test_week + i
        for i in range(validate_week):
            end = max_time - test_week*week_sec - i*week_sec
            start = max_time - test_week*week_sec - (1 + i)*week_sec
            distribution[(timestamps <= end) & (timestamps > start)] = validate_week- i
        print("test share=%lf, validate share=%lf" %((distribution < 0).sum()/len(links_samples_df),
                                                      (distribution > 0).sum()/len(links_samples_df)))
        links_samples_df['sample_num'] = distribution
        links_samples_df['has_link'] = 1
        links_samples_df.drop(labels=['time'], axis=1, inplace=True)
        links_samples_df['neigbours_num'] = -1
        # del timestamps

    else:
    # return None
        train_G = get_network(os.path.join(env_dir, 'train1')).to_undirected()

        links_samples_df = pd.DataFrame(
            [(i, j, len(list(nx.common_neighbors(train_G, i, j))))
             for (i, j) in list(train_G.edges)],
            columns=['left', 'right', 'neigbours_num'])
        links_samples_df.sort_values(by='neigbours_num', inplace=True)
        links_samples_df['sample_num'] = [i%samples_num for i in range(len(list(train_G.edges)))]
        links_samples_df['has_link'] = 1
        del train_G


    no_links = []
    no_links_distribution = []
    print("linked samples made")
    order = np.array([0] + list(range(1, distribution.max() + 1)) + list(range(-1, distribution.min() -1, -1)))
    for i in order:

        new_no_links = list(get_neg_links(links_samples_df, distribution, i, no_links, user_id))
        no_links = no_links + new_no_links
        no_links_distribution = no_links_distribution + [i]*len(new_no_links)


    G = get_network(graph_file).to_undirected()

    no_links_samples_df = pd.DataFrame(
                [(i, j, len(list(nx.common_neighbors(G, i, j))))
                for (i, j) in no_links],
                columns=['left', 'right', 'neigbours_num'])
    if not with_timestamp:
        no_links_samples_df.sort_values(by='neigbours_num', inplace=True)

    no_links_samples_df['sample_num'] = no_links_distribution
    no_links_samples_df['has_link'] = 0
    samples = pd.concat([links_samples_df, no_links_samples_df], sort=True)

    print("len of samples", len(samples))

    print("not_linked_samples_made", len(no_links))
    samples.drop('neigbours_num', axis=1, inplace=True)
    if not with_timestamp:

        test_G = get_network(os.path.join(env_dir, 'test1')).to_undirected()
        test_df = pd.DataFrame(list(test_G.edges), columns=['left', 'right'])
        test_df['sample_num'] = -1
        test_df['has_link'] = 1
        samples = pd.concat([samples, test_df])

    del G
    del no_links_samples_df
    del links_samples_df
    # del distribution
    del no_links
    all_links = list(map(lambda x: tuple(x), list(samples[["left", "right"]].itertuples(index=False))))
    print(len(all_links))
    gc.collect()

    test_df = samples[samples.sample_num == -test_week]
    if with_timestamp:
        max_timestamp = timestamps[distribution == -test_week].max()
    else:
        max_timestamp = None
    test_set = set([(x.left, x.right) for x in test_df[test_df.has_link == 1][['left', 'right']].itertuples()])
    features_samples = get_features(['like', 'friend', 'repost', 'all'], 'VK', args.env_dir,
                                    all_links, test_set, type=type, max_timestamp=max_timestamp)
    print("feature samples ", len(features_samples))
    features_samples = features_samples.add_suffix('_sample' + str(-test_week))
    features_samples.fillna(0, inplace=True)
    # del test_df
    remove_set=test_set

    print(str(-test_week) + " sample features counted, len of feature_samples =", len(features_samples))
    if samples_path is not None:
        # tmp = samples

        tmp = samples.set_index(['left', 'right'], verify_integrity=True).join(features_samples)

        # tmp.fillna(0.0, inplace=True)
        save_obj(tmp, os.path.join(samples_path, '_sample' + str(-test_week)))
        # del tmp

    prev_i = -test_week
    for i in order[-2::-1]:
        if i == 0:
            print("All test sample_features made")
            continue
        test_df = samples[samples.sample_num == i]
        remove_set = set([(x.left, x.right) for x in test_df[test_df.has_link == 1][['left', 'right']].itertuples()]).union(remove_set)
        # del test_df
        tmp = get_features(['like', 'friend', 'repost', 'all'], 'VK', args.env_dir,
                           all_links,remove_set, type=type,
                           max_timestamp = timestamps[distribution == -test_week].max() if with_timestamp else None).add_suffix("_sample" + str(i))
        tmp.fillna(0.0, inplace=True)
        features_samples = features_samples.join(tmp, how='outer')
        print(str(i) + " sample features counted, len of features_samples =", len(features_samples))

        if samples_path is not None:
            # tmp = samples.set_index(['left', 'right']).join(features_samples).fillna(0.0)
            save_obj(samples.set_index(['left', 'right'], verify_integrity=True).join(features_samples), os.path.join(samples_path, 'sample_' + str(i)))
            os.remove(os.path.join(samples_path, 'sample_' + str(prev_i)))
            # del tmp
        prev_i = i

    samples = samples.set_index(['left', 'right'], verify_integrity=True)
    samples = samples.join(features_samples)
    samples.fillna(0.0, inplace=True)

    if samples_path is not None:
        save_obj(samples, os.path.join(samples_path, 'sample_all'))

    return samples


def sample_generator(samples, features=None):
    columns = list(samples.columns)
    samples['sample_num'].min()
    for i in range(samples['sample_num'].min(), samples['sample_num'].max()+1):
        if i == 0:
            continue
        delete_list = []
        rename_list = {}
        for col in columns:
            if col in {'freq', 'has_link'}:
                continue
            if features is not None:
                feature_ok = False
                for feature in features:
                    feature_ok=feature_ok or (feature in col)
                if not feature_ok:
                    delete_list.append(col)
                    continue

            index = col.rfind("_")
            if index is not None:
                if col[index:] == "_sample" + str(i):
                    rename_list[col] = col[:index]
                    continue
                else:
                    delete_list.append(col)
        # print(columns, delete_list)
        if len(rename_list) == 0:
            continue
        if with_timestamp:
            if i == 0:
                continue
            elif i > 0: #validation
                mask = (samples['sample_num'] < i) & (samples['sample_num'] >= 0)
                train_df = samples.loc[mask].\
                    drop(delete_list,axis=1).rename(columns=rename_list)
            else:#test
                train_df = samples.loc[samples['sample_num'] > i].drop(delete_list, axis=1).rename(columns=rename_list)
            test_df = samples.loc[samples['sample_num'] == i].drop(delete_list, axis=1).rename(columns=rename_list)
        else:
            train_df = samples.loc[samples['sample_num'] != i].loc[samples['sample_num'] > 0].drop(delete_list, axis=1).rename(columns=rename_list)
            test_df = samples.loc[samples['sample_num'] == i].drop(delete_list, axis=1).rename(columns=rename_list)
            print (i, len(train_df), len(test_df))
        if remove_new:
            exist_links = train_df[train_df['has_link'] == 1]
            exist_links = exist_links.reset_index()
            all_users = set(list(exist_links['left']) + list(exist_links['right']))
            test_df_no_index = test_df.reset_index()
            mask = np.array(test_df_no_index['left'].isin(all_users)) & np.array(test_df_no_index['right'].isin(all_users))
            test_df = test_df[mask]
            # print("length of after removing:%d"%len(test_df))
        delete_list = list(test_df.columns)
        delete_list.remove('has_link')
        info_df = test_df.drop(columns = delete_list)
        info_df['sample_num'] = i
        yield train_df.reindex(sorted(train_df.columns), axis=1), \
                test_df.reindex(sorted(test_df.columns), axis=1), \
                info_df.reindex(sorted(info_df.columns), axis=1), \
                sorted(test_df.columns)


def val_select(samples, sample_proba):
    validate_sample_num = samples["sample_num"].max()
    samples.drop(columns='freq', inplace=True)
    features = set()
    for col in list(samples.columns):
        if col in {'freq', 'has_link'}:
            continue

        index = col.rfind("_")
        if index is not None:
            if col[index:] == "_sample-1":
                features.add(col[:index])
    best_mapk = 0
    best_auc = 0
    best_ndcg = 0
    for feature in features:
        print('*' * 10 + feature + "*" * 10)
        cur_auc = 0
        cur_mapk = 0
        cur_ndcg = 0
        for i in range(1, validate_sample_num + 1):
            test_df = samples[samples["sample_num"] == i]
            mask = (samples['sample_num'] < i) & (samples['sample_num'] >= 0)
            train_df = samples.loc[mask]
            exist_links = train_df[train_df['has_link'] == 1].reset_index()
            all_users = set(list(exist_links['left']) + list(exist_links['right']))
            test_df_no_index = test_df.reset_index()
            mask = np.array(test_df_no_index['left'].isin(all_users)) & np.array(
                test_df_no_index['right'].isin(all_users))
            test_df = test_df[mask]
            X = np.array(test_df[feature + "_sample" + str(i)])
            y = np.array(test_df["has_link"])

            # fpr, tpr, thresholds = roc_curve(y, X)
            cur_auc += roc_auc_score(y, X)
            tmp = map_k(y, X, ks)
            cur_mapk += tmp[1]
            cur_ndcg += ndcg_score(y, X)
        cur_mapk = cur_mapk / validate_sample_num
        cur_auc = cur_auc / validate_sample_num
        cur_ndcg = cur_ndcg / validate_sample_num
        if cur_auc > best_auc:
            best_auc_col = feature
            best_auc = cur_auc
        if cur_mapk > best_mapk:
            best_mapk = cur_mapk
            best_mapk_col = feature
        if cur_mapk > best_ndcg:
            best_ndcg = cur_mapk
            best_ndcg_col = feature

        print("AUC:", cur_auc)
        print("MAPK:", cur_mapk)
        print("NDCG:", cur_ndcg)
    print("*" * 20)
    print("BEST AUC - %s, BEST_MAPK - %s, BEST_NDCG - %s" %
          (best_auc_col, best_mapk_col, best_ndcg_col))
    for i in range(samples['sample_num'].min(), 0):
        test_df = samples[samples["sample_num"] == i]
        if remove_new:
            train_df = samples.loc[samples['sample_num'] > i]
            print("*" * 10 + "i = " + str(i) + "*" * 10)
            print("length of before removing:%d" % len(test_df))
            exist_links = train_df[train_df['has_link'] == 1].reset_index()
            all_users = set(list(exist_links['left']) + list(exist_links['right']))
            test_df_no_index = test_df.reset_index()
            mask = np.array(test_df_no_index['left'].isin(all_users)) & np.array(
                test_df_no_index['right'].isin(all_users))
            test_df = test_df[mask]
            print("length of after removing:%d" % len(test_df))
        X = np.array(test_df[best_auc_col + "_sample" + str(i)])
        y = np.array(test_df["has_link"])
        # fpr, tpr, thresholds = roc_curve(y, X)

        X = np.array(test_df[best_mapk_col + "_sample" + str(i)])
        a, b = map_k(y, X, ks)

        X = np.array(test_df[best_ndcg_col + "_sample" + str(i)])
        c = ndcg_score(y, X)
        print("sample %d, auc = %lf, mapk = %lf, ndcg=%lf, aps= %s" %
              (i, roc_auc_score(y, X), c, b, str(a)))


def simple_experiment(samples, sample_proba = 1, way="train", BT_eq =True, N =10):
    if way == "val":
        val_select(samples, sample_proba)
        return
    else: #way == train
        from SimpleClf import SimpleClf
        for metric, func in [('ndcg', lambda x, y: ndcg_score(x, y)[0]),
                             ('mapk',lambda x, y:map_k(x, y)[1]),
                             ('auc', roc_auc_score)]:
            all_res = []
            for n in GridParams['BT']['sample_n']:
                for x_train, y_train, x_test, y_test, info in mc_gen(sample_generator(samples),
                                                                     begin=2, end=3, selected_features=None,
                                                        select_proba=sample_proba):
                    score = None
                    if metric == 'ndcg':
                        score_func = lambda x, y: ndcg_score(x, y)[0]
                    else:
                        score_func = func
                    for i in range(N):
                        clf = SimpleClf(func, n)
                        clf.fit(x_train, y_train)
                        _probas = clf.decision_function(x_test)
                        # fpr, tpr, thresholds = roc_curve(y_test, _probas)

                        if score is None:
                            score = score_func(y_test, _probas)
                        else:
                            score += score_func(y_test, _probas)
                    #print("points:", ndcg_score(y_test, _probas)[1])
                    # res = roc_auc_score(y_test, res)
                    score /= N
                    all_res.append(score)
                    #print("BEST FEATURE - %d, %s= " % (clf.best_i, metric), score)
            print("all_res", metric, all_res)
            #all_res = []


def check_params(clf, samples, best_params, feature_n, selected_features, save_file, sample_proba = 1):
    classifier, generator = init_classifier(clf)

    res = test_clf(clf, classifier, best_params, generator(sample_generator(samples),
                                                           begin=0, end=-samples['sample_num'].min(),
                                                           selected_features=[x[:feature_n] for x in selected_features]
                                                           if selected_features is not None else None,
                                                           select_proba=sample_proba), verbose=1,save_res=save_file is not None)[-1]

    save_obj(res, save_file)


def call_from_string(clf, s_f, best_params, feature_n, save_file, sample_proba):
    best_params = eval(best_params)
    samples = load_obj(s_f)
    samples['freq'] = 1  # /freq
    if "Weight" in samples.columns:
        samples.drop(columns='Weight', inplace=True)
    if clf in WithSelection:
        feature_ns = [-1]
        selected_features = significant_features(sample_generator(samples),
                                                 select_method="GB",
                                                 feature_ns=feature_ns)[-1]
    else:
        selected_features = None

    check_params(clf, samples, best_params, feature_n, selected_features, save_file, sample_proba)



if __name__ == "__main__":
    user_id = None

    if False:
        clf = "IF"
        s_f = "E:\\terror_project\\MMCSnapIn\\python_utility\experiment\\facebook_timestamp\\sample_all"
        best_params = "{'contamination': 0.31149629594315725, 'max_samples': 654, 'n_estimators': 100}"
        feature_n = 12
        save_file = "C:\\Users\\alina\\Desktop\\эксперименты правильные\\facebook\\bayes\\if_AUC"
        sample_proba = 0.1
        with_timestamp = True
        remove_new = False
        call_from_string(clf, s_f, best_params, feature_n, save_file, sample_proba)
    else:
        parser = createParser()
        args = parser.parse_args(sys.argv[1:])
        with_timestamp = args.timestamp
        remove_new = args.remove_new
        if args.s_f is None and args.env_dir is None:
            print("No model_dir or sample file")
        if args.env_dir is not None:
            samples = get_samples(args.env_dir, args.s_n, args.s_f, with_timestamp= args.timestamp, type=args.features,
                                  user_id=None)
        else:
            samples = pd.read_pickle(args.s_f + ".pkl")#load_obj(args.s_f)
        samples['freq'] = 1 #/freqs

        if args.features == 'individual':
            samples = samples.reset_index()
            samples = samples[np.array(samples['left'].isin(user_id)) | np.array(samples['right'].isin(user_id))]
            print(samples['has_link'].sum())
            print(len(samples))
            samples.set_index(['left', 'right'], verify_integrity=True)
        print(len(samples))
        if args.norm or args.clf in NormMethods:
            scaler = StandardScaler(copy=False)
            train_df = samples.drop(['has_link', 'sample_num'], axis = 1)
            train_array = train_df.values
            df = scaler.fit_transform(train_array)
            samples.iloc[:, 2:] = df

        # probas = np.random.rand(samples.shape[0]) <= 0.5
        # samples = samples[probas]

        # if args.clf == "BT":
        #     bt(samples)
        if args.simple:
            simple_experiment(samples, args.sample_proba)
            exit(0)
        if args.bayes:
            bayes_experiment(args.clf.upper(), samples, args.s_n, select_method=args.select,
                        sample_proba=args.sample_proba, save_file=args.save)
        else:
            grid_experiment(args.clf.upper(), samples, args.s_n, select_method=args.select,
                    sample_proba=args.sample_proba)#, "friend_common_neigbors_num")
