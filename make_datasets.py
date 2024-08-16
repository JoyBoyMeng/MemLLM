import numpy as np
import random
import pandas as pd


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def get_data(dataset_name):
    ### Load data and train val test split
    if dataset_name == 'yelpv2':
        graph_df = pd.read_csv(f'../datasets/{dataset_name}/ml_{dataset_name}.csv')
        # review_texts = np.load(f'../datasets/{dataset_name}/text.npy', mmap_mode='r')
        # graph_texts = np.load(f'../datasets/{dataset_name}/talkasgraph_unify.npy', mmap_mode='r')
        #
        # review_texts = review_texts.tolist()
        # graph_texts = graph_texts.tolist()
        # new_string = 'This is a start str that will never be used'
        # review_texts = [new_string] + review_texts
        # graph_texts = [new_string] + graph_texts
        edge_raw_features = np.load(f'../datasets/{dataset_name}/embeddings.npy')
        zero_vector = np.zeros((1, 4096))  # 4096 768
        edge_raw_features = np.vstack((zero_vector, edge_raw_features))
    elif dataset_name == 'recipev2' or dataset_name == 'clothing':
        graph_df = pd.read_csv(f'../datasets/{dataset_name}/ml_{dataset_name}.csv')
        # review_texts = np.load(f'../datasets/{dataset_name}/text.npy', mmap_mode='r')
        # graph_texts = np.load(f'../datasets/{dataset_name}/talkasgraph_unify.npy', mmap_mode='r')
        # review_texts = review_texts.tolist()
        # graph_texts = graph_texts.tolist()
        # new_string = 'This is a start str that will never be used'
        # review_texts = [new_string] + review_texts
        # graph_texts = [new_string] + graph_texts
        edge_raw_features = np.load(f'../datasets/{dataset_name}/embeddings.npy')
        zero_vector = np.zeros((1, 4096))  # 4096 768
        edge_raw_features = np.vstack((zero_vector, edge_raw_features))
    elif dataset_name == 'reddit':
        graph_df = pd.read_csv(f'../datasets/{dataset_name}/ml_{dataset_name}.csv')
        review_texts = []
        graph_texts = np.load(f'../datasets/{dataset_name}/talkasgraph_unify.npy', mmap_mode='r')
        graph_texts = graph_texts.tolist()
        # 在列表的前面插入新的字符串
        new_string = 'This is a start str that will never be used'
        graph_texts = [new_string] + graph_texts

    # node_features = np.load(f'../datasets/{dataset_name}/ml_{dataset_name}_node.npy')

    if dataset_name == 'yelpv2' or dataset_name == 'recipev2' or dataset_name == 'clothing':
        val_time, test_time = list(np.quantile(graph_df.ts, [0.7, 0.85]))
    elif dataset_name == 'reddit':
        val_time, test_time, end_time = list(np.quantile(graph_df.ts, [0.035, 0.0425, 0.05]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    random.seed(2024)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # Compute nodes which appear at test time
    test_node_set = set(sources[timestamps > val_time]).union(
        set(destinations[timestamps > val_time]))
    # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

    # Mask saying for each source and destination whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # For train we keep edges happening before the validation time which do not involve any new node
    # used for inductiveness
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.sources).union(train_data.destinations)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    if dataset_name == 'yelpv2' or dataset_name == 'recipev2' or dataset_name == 'clothing':
        test_mask = timestamps > test_time
    elif dataset_name == 'reddit':
        test_mask = np.logical_and(timestamps <= end_time, timestamps > test_time)

    edge_contains_new_node_mask = np.array(
        [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask],
                             edge_idxs[new_node_val_mask], labels[new_node_val_mask])

    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              labels[new_node_test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
        len(new_test_node_set)))
    graph_texts, review_texts = [], []

    return graph_texts, review_texts, full_data, train_data, val_data, test_data, \
        new_node_val_data, new_node_test_data, edge_raw_features