import numpy as np


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:

            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, max_node_idx=None):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                        data.edge_idxs,
                                                        data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    return NeighborFinder(adj_list)


class NeighborFinder:
    def __init__(self, adj_list):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []

        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # We sort the list based on timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[
                                                                                             src_idx][:i]

    def get_temporal_neighbor(self, nodes, timestamps, n_samples, dataset, n_neighbors=10):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(nodes) == len(timestamps))
        texts = []
        neighbors_all = []
        edge_times_all = []
        edge_idxs_all = []

        for i, (node, timestamp) in enumerate(zip(nodes, timestamps)):
            neighbors, edge_idxs, edge_times = self.find_before(node,
                                                                timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

            if i < n_samples:
                if dataset == 'yelpv2':
                    prompt = 'User [user] has posted ' + str(len(neighbors)) + ' comments recently.\n'
                elif dataset == 'reddit':
                    prompt = 'Reddit user [user] has posted ' + str(len(neighbors)) + ' threads recently.\n'
                elif dataset == 'recipev2':
                    prompt = 'User [user] has posted ' + str(len(neighbors)) + ' comments recently.\n'
                elif dataset == 'clothing':
                    prompt = 'User [user] has posted ' + str(len(neighbors)) + ' comments recently.\n'
            else:
                if dataset == 'yelpv2':
                    prompt = 'Business [business] has received ' + str(len(neighbors)) + ' comments recently.\n'
                elif dataset == 'reddit':
                    prompt = 'The subreddit [subreddit] recently received ' + str(len(neighbors)) + ' threads.\n'
                elif dataset == 'recipev2':
                    prompt = 'Recipe [recipe] has received ' + str(len(neighbors)) + ' comments recently.\n'
                elif dataset == 'clothing':
                    prompt = 'Clothes [clothes] has received ' + str(len(neighbors)) + ' comments recently.\n'
            if len(neighbors) > 0 and n_neighbors > 0:
                # Take most recent interactions
                edge_times = edge_times[-n_neighbors:]
                neighbors = neighbors[-n_neighbors:]
                edge_idxs = edge_idxs[-n_neighbors:]

                assert (len(neighbors) <= n_neighbors)
                assert (len(edge_times) <= n_neighbors)
                assert (len(edge_idxs) <= n_neighbors)
                assert (len(neighbors) == len(edge_idxs))

                # node_memory = memory[[node] * len(neighbors)]   # (neighbor_num, memdim)
                # neighbors_memory = memory[neighbors]
                # node_memory = node_memory.unsqueeze(1).expand(-1, token_len, -1)    # (neighbor_num, token_len, memdim)
                # neighbors_memory = neighbors_memory.unsqueeze(1).expand(-1, token_len, -1)
                neighbors_all.append(neighbors)
                edge_times_all.append(edge_times)
                edge_idxs_all.append(edge_idxs)

                if dataset == 'yelpv2':
                    graph_texts_1node = ['User [user] commented on business [business] at time [tij].\n'] * len(neighbors)
                elif dataset == 'reddit':
                    graph_texts_1node = ['Reddit user [user] posts a thread in the subreddit [subreddit] at time [tij].\n'] * len(neighbors)
                elif dataset == 'recipev2':
                    graph_texts_1node = ['User [user] commented on recipe [recipe] at time [tij].\n'] * len(
                        neighbors)
                elif dataset == 'clothing':
                    graph_texts_1node = ['User [user] commented on clothes [clothes] at time [tij].\n'] * len(
                        neighbors)
                emb_texts_1node = ''.join(graph_texts_1node)
                if i < n_samples:
                    emb_texts_1node = prompt + emb_texts_1node
                else:
                    if dataset == 'yelpv2':
                        emb_texts_1node = prompt + emb_texts_1node + 'Will user [user] comment on business [business] at time [tij]?\n'
                    elif dataset == 'reddit':
                        emb_texts_1node = prompt + emb_texts_1node + 'Will user [user] post a thread in the subreddit [subreddit] at time [tij]?\n'
                    elif dataset == 'recipev2':
                        emb_texts_1node = prompt + emb_texts_1node + 'Will user [user] comment on recipe [recipe] at time [tij]?\n'
                    elif dataset == 'clothing':
                        emb_texts_1node = prompt + emb_texts_1node + 'Will user [user] comment on clothes [clothes] at time [tij]?\n'
            elif len(neighbors) == 0:
                if i < n_samples:
                    emb_texts_1node = prompt
                else:
                    if dataset == 'yelpv2':
                        emb_texts_1node = prompt + 'Will user [user] comment on business [business] at time [tij]?\n'
                    elif dataset == 'reddit':
                        emb_texts_1node = prompt + 'Will user [user] post a thread in the subreddit [subreddit] at time [tij]?\n'
                    elif dataset == 'recipev2':
                        emb_texts_1node = prompt + 'Will user [user] comment on recipe [recipe] at time [tij]?\n'
                    elif dataset == 'clothing':
                        emb_texts_1node = prompt + 'Will user [user] comment on clothes [clothes] at time [tij]?\n'
                neighbors_all.append([])
                edge_times_all.append([])
                edge_idxs_all.append([])
            else:
                print('Something is going crazy in neighbor finder!!!')
                exit()
            texts.append(emb_texts_1node)
        texts = np.array(texts)
        return texts, neighbors_all, edge_times_all, edge_idxs_all


def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round
