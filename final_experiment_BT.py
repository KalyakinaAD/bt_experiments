from features import calculate_link_features, label_independent_features, links_feature, find_graph_features
import numpy as np
import pandas as pd
import os
from common import load_obj
import networkx as nx
import random
from BT import BT
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier
from itertools import combinations, product
from sklearn.svm import SVC
from time import time

BT_flag = False

K = 100
N = 5000

test_week = 7
validate_week = 5
week_sec = 2*3600*24
number_of_features = 3*10

bt_sample_n = [1000, 5000, 7000, 10000]#list(range(100, 5000, 200))


directory = "E:\\terror_project\\MMCSnapIn\\python_utility\\experiment\\digg_f+v\\"
# link_features_file = os.path.join(directory, 'sample1.pkl')
like_links= os.path.join(directory, "VK_like")
friend_links= os.path.join(directory, "VK_like")
all_links = os.path.join(directory, "VK_time")


friend_links_df = pd.read_csv(friend_links+".csv", sep=';', error_bad_lines=True, index_col=False, dtype={'left':int, 'right':int})
friend_links_df['Weight'] = 1
friend_links_df['Weight_reversed'] = 1


links_samples_df = pd.read_csv(all_links + ".csv", sep=';', error_bad_lines=True, index_col=False, dtype={'left':int, 'right':int})

links_samples_df.sort_values(by='time', inplace=True)

timestamps = np.array(links_samples_df['time'])
max_time = timestamps[-1]
# test_week = int((max_time - timestamps[int(0.75*len(links_samples_df))])//week_sec)
# validate_week = int((timestamps[int(0.75*len(links_samples_df))] - timestamps[int(0.6*len(links_samples_df))])//week_sec)
distribution = np.array([0]*len(links_samples_df))

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
# links_samples_df['has_link'] = 1
# links_samples_df.drop(labels=['time'], axis=1, inplace=True)
links_samples_df['Weight'] = 1
links_samples_df['Weight_reversed'] = 1

# experiment weeks sample get features
for i in range(-7, -1, 1):
    print("week %d"%i)
    G = nx.from_pandas_edgelist(links_samples_df, source='left',
                                                                          target='right',
                                                                          edge_attr=['Weight', 'Weight_reversed'],
                                                                          create_using=nx.DiGraph())

    G_train = nx.from_pandas_edgelist(links_samples_df[links_samples_df.sample_num >= i + 2],
                                                                          source='left',
                                                                          target='right',
                                                                          edge_attr=['Weight', 'Weight_reversed'],
                                                                          create_using=nx.DiGraph())

    G_test = nx.from_pandas_edgelist(links_samples_df[links_samples_df.sample_num >= i + 1],
                                                                         source='left',
                                                                         target='right',
                                                                         edge_attr=['Weight', 'Weight_reversed'],
                                                                         create_using=nx.DiGraph())


    cur_graph = G_test.to_undirected()

    prev_graph = G_train.to_undirected()

    next_graph = nx.from_pandas_edgelist(links_samples_df[links_samples_df.sample_num >= i],
                                        source='left',
                                        target='right',
                                        edge_attr=['Weight', 'Weight_reversed'],
                                        create_using=nx.DiGraph()).to_undirected()

    max_time = links_samples_df[links_samples_df.sample_num == i]['time'].min()

    friend_graph = nx.from_pandas_edgelist(friend_links_df[friend_links_df.time < max_time],
                                    source='left',
                                    target='right',
                                    edge_attr=['Weight', 'Weight_reversed'],
                                    create_using=nx.DiGraph())
    cur_nodes = list(map(int, cur_graph.nodes))
    prev_nodes = list(map(int, prev_graph.nodes))

    cur_edges = list(map(lambda x: (min(int(x[0]), int(x[1])), max(int(x[0]), int(x[1]))), cur_graph.edges))
    next_edges = list(map(lambda x: (min(int(x[0]), int(x[1])), max(int(x[0]), int(x[1]))), next_graph.edges))

    roc_auc_results = []
    pr_auc_results = []

    for bt_n in bt_sample_n: # for each number of samples
        final_pr_auc = 0
        final_roc_auc = 0
        time0 = time()
        for j in range(5): #repeat 5 times for smooth

            links_train = links_samples_df[links_samples_df.sample_num == i + 1]
            users_train = set(links_train.left) | set(links_train.right)

            df_train = find_graph_features(G_train, '')
            df_train['got link'] = np.vectorize(lambda x: x in users_train)(df_train.reset_index(['MemberName'])["MemberName"])

            links_test = links_samples_df[links_samples_df.sample_num  == i]
            users_test = set(links_test.left) | set(links_test.right)

            df_test = find_graph_features(G_test, '')
            df_test['got link'] = np.vectorize(lambda x: x in users_test)(df_test.reset_index(['MemberName'])["MemberName"])


            model = GradientBoostingClassifier()#, class_weight={1: df_train['got link'].sum(), 0:len(cur_graph.nodes) - df_train['got link'].sum()})


            model.fit(df_train.drop(columns=['got link']), df_train['got link'])
            link_users = model.predict(df_test.drop(columns=['got link']))
            link_probas = model.decision_function(df_test.drop(columns=['got link']))

            print("real users = %d, predicted users = %d" % (df_test['got link'].sum(), link_users.sum()))
            a = roc_auc_score(df_test['got link'], link_probas)
            b = average_precision_score(df_test['got link'], link_probas)
            print("use roc auc = %lf"%a)

            predicted_users = (df_test.reset_index(['MemberName']))["MemberName"][link_probas.argsort()][::-1]
            final_pairs = list(filter(lambda x: x[0] != x[1] and not cur_graph.has_edge(*x),
                                      product(predicted_users[:K], predicted_users[:N])))
            #do bt
            if BT_flag:
                # select posetive and negative pairs

                pos_pairs = random.sample(list(cur_graph.edges), 2 * bt_n // number_of_features)
                pos_pairs_set = set(pos_pairs)
                neg_pairs = set()
                while len(neg_pairs) < (2 * bt_n // number_of_features):
                    left, right = random.choice(cur_nodes), random.choice(cur_nodes)
                    left, right = right, left if (right < left) else (left, right)
                    if not cur_graph.has_edge(left, right) and left != right and (left, right) not in neg_pairs:
                        neg_pairs.add((left, right))

                neg_pairs = list(neg_pairs)
                # count features for selected nodes
                friend_pos_features = calculate_link_features(
                    nx.from_pandas_edgelist(links_samples_df[links_samples_df.sample_num >= i + 1],
                                            source='left',
                                            target='right',
                                            edge_attr=['Weight', 'Weight_reversed'],
                                            create_using=nx.DiGraph()), "friend_", pos_pairs, parallel=False)
                friend_neg_features = calculate_link_features(
                    nx.from_pandas_edgelist(links_samples_df[links_samples_df.sample_num >= i + 1],
                                            source='left',
                                            target='right',
                                            edge_attr=['Weight', 'Weight_reversed'],
                                            create_using=nx.DiGraph()), "friend_", neg_pairs, parallel=False)

                all_pos_features = calculate_link_features(G_test, "", pos_pairs, parallel=False)
                all_neg_features = calculate_link_features(G_test, "", neg_pairs, parallel=False)

                neg_features = all_neg_features.join(friend_neg_features, how='left')
                pos_features = all_pos_features.join(friend_pos_features, how='left')
                bt = BT(bt_n)
                bt.fit(np.array(neg_features), np.array(pos_features))

                #select feature to count ... somehow...
                selected_feature = bt.best
                feature_name = neg_features.columns[bt.best]
                print("best feature = %s"%feature_name)
                idx_ = feature_name.find("_")
                prefix = ''
                if idx_ >= 0:
                    prefix = feature_name[:idx_]
                if prefix == 'friend':
                    final_graph = friend_graph
                    feature_name = feature_name[idx_+1:]
                else:
                    final_graph = G_test
                    feature_name = feature_name

                print("len of final pairs %d"%len(final_pairs))
                final_index = links_feature(cur_graph, final_pairs, feature_name)

                final_index = final_index.iloc[:, 0]

            else:
                pos_pairs = random.sample(list(prev_graph.edges), 2 * bt_n // number_of_features)
                pos_pairs_set = set(pos_pairs)
                neg_pairs = set()
                while len(neg_pairs) < (2 * bt_n // number_of_features):
                    left, right = random.choice(prev_nodes), random.choice(prev_nodes)
                    left, right = right, left if (right < left) else (left, right)
                    if not prev_graph.has_edge(left, right) and left != right and (left, right) not in neg_pairs:
                        neg_pairs.add((left, right))

                neg_pairs = list(neg_pairs)
                # count features for selected nodes
                friend_pos_features = calculate_link_features(
                    nx.from_pandas_edgelist(links_samples_df[links_samples_df.sample_num >= i + 2],
                                            source='left',
                                            target='right',
                                            edge_attr=['Weight', 'Weight_reversed'],
                                            create_using=nx.DiGraph()), "friend_", pos_pairs, parallel=False)
                friend_neg_features = calculate_link_features(
                    nx.from_pandas_edgelist(links_samples_df[links_samples_df.sample_num >= i + 2],
                                            source='left',
                                            target='right',
                                            edge_attr=['Weight', 'Weight_reversed'],
                                            create_using=nx.DiGraph()), "friend_", neg_pairs, parallel=False)

                all_pos_features = calculate_link_features(G_train, "", pos_pairs, parallel=False)
                all_neg_features = calculate_link_features(G_train, "", neg_pairs, parallel=False)

                neg_features = all_neg_features.join(friend_neg_features, how='left')
                pos_features = all_pos_features.join(friend_pos_features, how='left')


                model = GradientBoostingClassifier()
                model.fit(np.array(neg_features.append(pos_features)), [0]*len(neg_features) + [1]*len(pos_features))

                all_features = calculate_link_features(G_test, "", final_pairs, parallel=False)
                friend_features = calculate_link_features(
                    nx.from_pandas_edgelist(links_samples_df[links_samples_df.sample_num >= i + 1],
                                            source='left',
                                            target='right',
                                            edge_attr=['Weight', 'Weight_reversed'],
                                            create_using=nx.DiGraph()), "friend_", final_pairs, parallel=False)
                features = all_features.join(friend_features, how='left')
                final_index = model.decision_function(features)

            #from test set
            final_true = []
            final_scores = []
            for idx, pair in zip(final_index, final_pairs):
                final_true.append(next_graph.has_edge(*pair))
                final_scores.append(idx)

            a = average_precision_score(final_true, final_scores)
            b = roc_auc_score(final_true, final_scores)
            final_pr_auc += a
            final_roc_auc += b

            print("final roc_auc = %lf"%b)
            print("final pr_auc = %lf"%a)


        print("%d bt_n: pr_auc=%lf, roc_auc=%lf, time = %lf" % (bt_n, final_pr_auc / 5, final_roc_auc/5, (-time0 + time()) / 5))
        roc_auc_results.append(final_roc_auc / 5)
        pr_auc_results.append(final_pr_auc / 5)
        if not BT_flag:
            break

    print("FOR week %d:"%i, "pr_aucs =", pr_auc_results)
    print("FOR week %d:"%i, "roc_aucs =", roc_auc_results)