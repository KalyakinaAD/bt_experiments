import pandas as pd
from common import get_cursor, update_db, save_obj
import networkx as nx
import numpy as np
#import node2vec



def links_feature(G, pairs, feature):
    #from itertools import product
    from time import time
    t0 = time()
    res = None
    if feature == 'preferrential_attachment':
        res = pd.DataFrame(
            nx.preferential_attachment(G, pairs),
            columns=['left', 'right', 'preferrential_attachment']).set_index(['left', 'right'])
    elif feature == 'adamic_adar_index':
        res = pd.DataFrame(
            nx.adamic_adar_index(G, pairs),
            columns=['left', 'right', 'adamic_adar_index']).set_index(['left', 'right'])
    elif feature == 'jaccard_coefficient':
        res =  pd.DataFrame(
            nx.jaccard_coefficient(G, pairs),
            columns=['left', 'right', 'jaccard_coefficient']).set_index(['left', 'right'])
    elif feature == 'common_neigbors_num':
        res = pd.DataFrame(
        [(i, j, len(list(nx.common_neighbors(G, i, j))))
         for (i, j) in pairs],
        columns=['left', 'right', 'common_neigbors_num']).set_index(['left', 'right'])
    print("%s takes %f" % (feature, (time() - t0)))
    print(feature, ".shape = ", res.shape, "pairs.shape", len(pairs))

    return res


def _calculate_link_features(G, pairs, prefix='', parallel = False, filename = 'test'):
    from itertools import product
    import random
    from time import time
    graph_size = len(G.nodes())

    if pairs:
        # p = Pool(4)
        # res = p.map()
        if not parallel:
            t0 = time()
            preferential_attachment = pd.DataFrame(
                nx.preferential_attachment(G, pairs),
                columns=['left', 'right', 'preferrential_attachment']).set_index(['left', 'right'])
            print("%s preferrential_attachment takes %f" % (filename, (time() - t0)))

            t0 = time()
            adamic_adar_index = pd.DataFrame(
                nx.adamic_adar_index(G, pairs),
                columns=['left', 'right', 'adamic_adar_index']).set_index(['left', 'right'])
            print("%s adamic_adar_index takes %f" % (filename, (time() - t0)))

            t0 = time()
            jaccard_coefficient = pd.DataFrame(
                nx.jaccard_coefficient(G, pairs),
                columns=['left', 'right', 'jaccard_coefficient']).set_index(['left', 'right'])
            print("%s jaccard_coefficient takes %f" % (filename, (time() - t0)))

            t0 = time()
            common_neighbors_num = pd.DataFrame(
                [(i, j, len(list(nx.common_neighbors(G, i, j))))
                 for (i, j) in pairs],
                columns=['left', 'right', 'common_neigbors_num']).set_index(['left', 'right'])
            print("%s common_neigbors_num takes %f" % (filename, (time() - t0)))

            link_features_df = preferential_attachment.join(adamic_adar_index) \
                .join(jaccard_coefficient) \
                .join(common_neighbors_num)

        else: #parallel
            from multiprocessing import Pool
            p = Pool(4)
            print("parallel link features for %s" % filename)
            res = p.starmap(links_feature, zip([G]*4, [pairs]*4, ['preferrential_attachment', 'adamic_adar_index',
                                           'jaccard_coefficient', 'common_neigbors_num']))
            link_features_df = res[0]
            # for x in res:
            #     print("in parallel shape=", x.shape, "(", x.columns, ")")
            for df in res[1:]:
                link_features_df = link_features_df.join(df)
                # print("join:", link_features_df.shape)
            p.close()
            p.join()

        link_features_df['preferrential_attachment'] = link_features_df['preferrential_attachment'] / graph_size ** 2
        link_features_df['adamic_adar_index'] = link_features_df['adamic_adar_index'] / graph_size * np.log(graph_size)
        link_features_df['jaccard_coefficient'] = link_features_df['jaccard_coefficient']
        link_features_df['common_neigbors_num'] = link_features_df['common_neigbors_num'] / graph_size
        return link_features_df.add_prefix(prefix)
    else:
        return None


def calculate_link_features(filename, prefix, pairs, parallel = False):
    from multiprocessing.pool import ThreadPool
    if filename is None:
        return None
    try:
        if isinstance(filename, str):
            G = get_network(filename)
        else:
            G = filename
    except Exception as e:
        print(e)
        return None
    pairs = _check_pairs(G, pairs)
    if not pairs:
        return None
    if parallel:
        p = ThreadPool(2)
        res = p.starmap(_calculate_link_features, [(G.to_undirected(reciprocal=True), pairs, 'reciprocal_', parallel, filename),
                         (G.to_undirected(reciprocal=False), pairs, '', parallel, filename)])
        # print("in calculate link features res0.shape =", res[0].shape, "res1.shape =", res[1].shape, "len of pairs =", len(pairs))
        link_features_df = res[0].join(res[1])
        p.close()
        p.join()
    else:
        link_features_df = _calculate_link_features(G.to_undirected(reciprocal=True), pairs, 'reciprocal_', parallel, filename)
        link_features_df = link_features_df.join(_calculate_link_features(G.to_undirected(reciprocal=False), pairs, '', parallel, filename))
    link_features_df = link_features_df.add_prefix(prefix)
    return link_features_df


def _check_pairs(G, pairs):
    users = set(G.nodes())
    res_pairs = []
    for pair in pairs:
        if pair[0] in users and pair[1] in users and pair[0] != pair[1]:
            res_pairs.append(pair)
    return res_pairs


def get_connectivity(filenames, pairs):
    import pandas as pd
    df = pd.DataFrame(pairs, columns= ['left', 'right']).set_index(['left', 'right'])
    #df["has_link"] = pd.Series(np.zeros(len(df)), index=df.index)
    edges = []
    for f in filenames:
        try:
            G = get_network(f).to_undirected()
            edges = edges + list(G.edges)
        except Exception as e:
            print(e)
    has_link = []
    edges = set(edges)
    for pair in pairs:
        if pair in edges:
            print("linked %d and %d" %pair)
            has_link.append(1)
        else:
            has_link.append(-1)
    df["has_link"] = has_link
    return df


def calculate_dangerous_features(df, users, prefix):
    #users = list(map(str, users))
    df_max_features = (df.loc[:, users] == 1).max(axis=1).astype(int)
    df_count_features = (df.loc[:, users] == 1).sum(axis=1).astype(int)
    df_part_features = (df.loc[:, users] == 1).sum(axis=1) / (df == 1).sum(axis=1)

    df_max_features = pd.DataFrame(
        data=df_max_features,
        columns=['Max']
    ).add_prefix(prefix)
    df_count_features = pd.DataFrame(
        data=df_count_features,
        columns=['Count']
    ).add_prefix(prefix)
    df_part_features = pd.DataFrame(
        data=df_part_features,
        columns=['Part']
    ).add_prefix(prefix)

    return df_part_features.join(df_count_features, how='left').join(df_max_features, how='left')


def label_dependent_features(env_id, shortest_paths, feature_types, top=5):
    new_features = pd.DataFrame()
    conn, cursor = get_cursor()
    good_users, bad_users = good_bad_users_from_db(env_id, cursor)
    # good_users = set(map(str, good_users))
    # bad_users = set(map(str, bad_users))

    for df, prefix in shortest_paths:
        try:
            df = load_paths(df)
        except Exception as e:
            print(e)
            continue
        dangerous_users_df = list(set(df.index).intersection(bad_users))
        safe_users_df = list(set(df.index).intersection(good_users))
        unknown_users_df = list(set(df.index) - set(bad_users) - set(good_users))
        for users, users_type in [(dangerous_users_df, 'bad_'),
                                      (safe_users_df, 'good_'),
                                      (unknown_users_df, 'unknown_')]:
            if len(users) == 0:
                continue
            users_str = ', '.join(list(map(str, users)))
            users_for_column = list(map(str, users))
            df_path_features = df.loc[:, users_for_column] \
                    .T.agg(['min']) \
                    .T.add_prefix(users_type + prefix)
            new_features = new_features.join(df_path_features, how='outer')
            df_path_features = df.loc[users, :] \
                .agg(['min'], axis=0) \
                .T.add_prefix(users_type + prefix + 'rev_')
            new_features = new_features.join(df_path_features, how='outer')
            for feature in feature_types:
                command = "SELECT user_id " \
                          "FROM data.user_features " \
                          "WHERE environment_id = %s AND %s IS NOT NULL " \
                          "AND user_id IN(%s) " \
                          "ORDER BY %s DESC " \
                          "LIMIT %d" % (env_id, feature, users_str, feature, top)
                cursor.execute(command)
                top_users = cursor.fetchall()
                top_users_df = [x for t in top_users for x in t]
                if len(set(df.index).intersection(top_users_df)) == 0:
                    continue
                df_path_features_top = df.loc[:, users_for_column] \
                    .T.agg(['min']) \
                    .T.add_prefix(users_type + prefix) \
                    .add_suffix('_top{}_'.format(top) + feature)
                new_features = new_features.join(df_path_features_top, how='outer')

            if 'weight' not in prefix:
                    #### count?
                new_features = new_features.join(calculate_dangerous_features(df, users_for_column, users_type + prefix))
                new_features = new_features.join(calculate_dangerous_features(df, users_for_column, users_type + prefix + 'rev_'))

    update_db(env_id, new_features, cursor)
    conn.close()
    del new_features
    return


def good_bad_users_from_db(env_id, cursor):
    command = "SELECT user_id " \
              "FROM data.user_features " \
              "WHERE user_type = %s " \
              "AND environment_id = " + str(env_id)
    try:
        cursor.execute(command % "'good'")
        good_users = cursor.fetchall()
        cursor.execute(command % "'bad'")
        bad_users = cursor.fetchall()
        return set([x for t in good_users for x in t]), set([x for t in bad_users for x in t])
    except Exception as e:
        print(e)


def find_users_distances(G):
    #pay attention on orient!!!!!
    shortest_path = dict(nx.all_pairs_shortest_path_length(G))
    df_shortest_path = pd.DataFrame.from_dict(shortest_path, orient = "index")
    df_shortest_path.index = df_shortest_path.index.astype('int64')
    df_shortest_path.columns = df_shortest_path.columns.astype('int64')

    shortest_path_weighted = dict(nx.all_pairs_dijkstra_path_length(G, weight='Weight_reversed'))
    df_shortest_path_weighted = pd.DataFrame.from_dict(shortest_path_weighted, orient = "index")
    df_shortest_path_weighted.index = df_shortest_path_weighted.index.astype('int64')
    df_shortest_path_weighted.columns = df_shortest_path_weighted.columns.astype('int64')

    for user in df_shortest_path_weighted.index:
        df_shortest_path_weighted.loc[user, user] = np.nan
        df_shortest_path.loc[user, user] = np.nan
    return df_shortest_path, df_shortest_path_weighted


def find_graph_features(G, prefix):
    from time import time

    t0 = time()
    sociablility = {i: G.degree()[i] / (2 * len(set(nx.all_neighbors(G, i)))) for i in G.nodes()}
    print("sociability takes %lf " %(time() - t0))
    t0 = time()
    degree = pd.DataFrame(list(nx.in_degree_centrality(G).items()),
                          columns=['MemberName', 'deg_prestige']).set_index('MemberName')
    print("in degree takes %lf " %(time() - t0))


    sociablility = pd.DataFrame(list(sociablility.items()),
                                columns=['MemberName', 'sociability']).set_index('MemberName')
    t0 = time()
    pagerank_weighted = pd.DataFrame(list(nx.pagerank(G, weight='Weight_reversed').items()),
                                     columns=['MemberName', 'pagerank_weight']).set_index('MemberName')
    print("pagerank weighted takes %lf " % (time() - t0))

    t0 = time()
    pagerank = pd.DataFrame(list(nx.pagerank(G).items()),
                            columns=['MemberName', 'pagerank']).set_index('MemberName')

    print("pagerank takes %lf " % (time() - t0))
    t0 = time()
    df_vertices_features = degree.join(sociablility).join(pagerank).join(pagerank_weighted)
    t0 = time()
    try:
        hubs, authorities = nx.hits(G, max_iter=1000)
        hubs = pd.DataFrame(list(hubs.items()),
                            columns=['MemberName', 'hub_measure']).set_index('MemberName')
        authorities = pd.DataFrame(list(authorities.items()),
                                   columns=['MemberName', 'author_measure']).set_index('MemberName')
        df_vertices_features = df_vertices_features.join(hubs).join(authorities)
    except Exception as e:
        print("Exception with HITS: " + str(e))
    print("HITs takes %lf " % (time() - t0))
    if len(G.nodes()) < 100:
        closeness = pd.DataFrame(list(nx.closeness_centrality(G).items()),
                                 columns=['MemberName', 'prox_prestige']).set_index('MemberName')
        betweenness = pd.DataFrame(list(nx.betweenness_centrality(G).items()),
                                   columns=['MemberName', 'between_central']).set_index('MemberName')
        df_vertices_features= df_vertices_features.join(closeness).join(betweenness)

    return df_vertices_features.add_prefix(prefix)


def load_paths(filename):
    filename = filename + ".csv"
    print("reading " + filename)
    df = pd.read_csv(filename, sep=';', error_bad_lines=True, index_col=False)
    #df.columns =
    df = df.set_index('who')
    return df


def get_network(csv_file):
    #import igraph
    print("reading", csv_file)
    df = pd.read_csv(csv_file+'.csv', sep=';', error_bad_lines=True, index_col=False)#,
                     #dtype={"left":'string', "right":'string', "Weight":"float"})
    # if "Weight" not in df.columns:
    #     df["Weight"] = 1
    df["Weight_reversed"] = 1/df["Weight"]



    G = nx.from_pandas_edgelist(df,
                                source='left',
                                target='right',
                                edge_attr=['Weight', 'Weight_reversed'],
                                create_using=nx.DiGraph())


    print("*****Graph statistic*****")
    print("Nodes: %d\nEdges: %d" % (G.number_of_nodes(), G.number_of_edges()))
    return G


def find_shortest_paths(csv_file):
    import subprocess
    df = pd.read_csv(csv_file + '.csv', sep=';', error_bad_lines=True, index_col=False)
    df["Weight_reversed"] = 1 / df["Weight"]
    df["Length"] = 1
    filename1 = csv_file + "_weights_reversed"
    filename2 = csv_file + "_length"
    df.to_csv(filename1 + ".csv", sep=";", columns=['left', 'right', "Weight_reversed"], index = False)
    df.to_csv(filename2 + ".csv", sep=";", columns=['left', 'right', "Length"], index  = False)
    # params1 = " -f " + filename1 + " -f_out " + csv_file + '_shortest_path_weighted.csv'
    # params2 = " -f " + filename2 + " -f_out " + csv_file + '_shortest_path.csv'

    p1 = subprocess.Popen(["E:\\terror_project\\MMCSnapIn\\python_utility\\count_paths.exe", "-f", filename1, "-max_time", "2.5", "-f_out",
                           (csv_file + '_shortest_path_weighted.csv')])

    p2 = subprocess.Popen(["E:\\terror_project\\MMCSnapIn\\python_utility\\count_paths.exe", "-f", filename2, "-max_time", "2.5", "-f_out",
                           (csv_file + '_shortest_path.csv')])

    return p1, p2


def label_independent_features(env_id, link, filename, save_paths=True, G=None):
    import os
    p1 = None
    p2 = None
    df_vertices_features_G = None
    try:
        if filename is not None:
            if save_paths:
                p1, p2 = find_shortest_paths(filename)
            G = get_network(filename)
        df_vertices_features_G = find_graph_features(G, link + '_')
        del G
        if env_id is not None:
            conn, cur = get_cursor()
            update_db(env_id, df_vertices_features_G, cur)
            conn.close()
    except Exception as e:
        print(e)
        exit(0)
    #save_obj(df_vertices_features_G, filename + '_vertices_features')
    if filename is not None:
        filename1 = filename + "_weights_reversed.csv"
        filename2 = filename + "_length.csv"
        try:
            if p1 is not None:
                p1.wait()
                os.remove(filename1)
        except Exception as e:
            print(e)
        try:
            if p2 is not None:
                p2.wait()
                os.remove(filename2)
        except Exception as e:
            print(e)
    return df_vertices_features_G


def get_commands(column_names):
    commands = set([])
    for column in column_names:
        commands.add('ALTER TABLE data.user_features ADD COLUMN ' + column + ' double precision')
    return commands


def new_columns():
    column_names = set([])
    for type in ['bad_', 'good_', 'unknown_']:
        for path_link in ['like_', 'repost_', 'friend_', 'all_']:
            for weight in ['', 'weight_']:
                for reverse in ['', 'rev_']:
                    column_names.add(type + 'path_' + path_link + weight + reverse + 'min')
                for feature in [link + "_" + feature for link in ['like', 'friend', 'repost', 'all'] for feature in
                                ['deg_prestige', 'prox_prestige', 'between_central', 'hub_measure',
                                 'author_measure', 'sociability', 'pagerank_weight', 'pagerank']]:
                    column_names.add(type + 'path_' + path_link + weight + 'min_top5_' + feature)
                    column_names.add(feature)
                if 'weight' not in weight:
                    for reverse in ['', 'rev_']:
                        for agg in ['Max', 'Count', 'Part']:
                            column_names.add(type + 'path_' + path_link + weight + reverse + agg)
    return column_names


def node2vec_features(G, prefix, dim=16):
    from node2vec import  Node2Vec
    import tempfile
    import uuid
    import os
    node2vec = Node2Vec(G, dimensions=dim, walk_length=10, num_walks=50, workers=1)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    tmp_dir = tempfile.gettempdir()
    tmp_file = os.path.join(tmp_dir, str(uuid.uuid4()))
    model.wv.save_word2vec_format(tmp_file, binary=False)
    df = pd.read_csv(tmp_file, sep = ' ', names=['node'] + [str(i) for i in range(dim)], skiprows=1)
    os.remove(tmp_file)
    return df.set_index('node').add_prefix(prefix)


if __name__ == "__main__":
    commands = get_commands(new_columns())
    conn, cursor = get_cursor()
    for command in commands:
        try:
            cursor.execute(command)
        except Exception as e:
            print(e)

# for type in ['bad_', 'good_', 'unknown_']:
#     for path_link in ['like_', 'repost_', 'friend_']:
#         for weight in ['', 'weighted_']:
#             for reverse in ['', 'reversed_']:
#                 column_names.append(type+'shortest_path_' + path_link + weight +reverse + 'min')
#             for feature in [link + "_" + feature for link in ['like', 'friend', 'repost'] for feature in ['degree_prestige', 'proximity_prestige', 'betweenness_centrality','hub_measure', 'authority_measure', 'sociability', 'pagerank_weighted', 'pagerank']]:
#                 column_names.append(type+'shortest_path_' + path_link + weight + 'min_top_5_' + feature)
#             if 'weighted' not in weight:
#                 for reverse in ['', 'reversed_']:
#                     for agg in ['Max', 'Count', 'Part']:
#                         column_names.append(type + 'shortest_path_' + path_link + weight + reverse + agg)