import torch
from collections import defaultdict
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from memory import Memory
from modules import TimeEncode, ExpandMemory, FixEmbed, ShrinkMessage, MergeLayer, ReductionLayer
from memory_updater import get_memory_updater
from message_aggregator import get_message_aggregator
from embedding_module import get_embedding_module, get_llm_embedding


class MemLLM(torch.nn.Module):
    def __init__(self, llm_path, device, num_nodes, num_users, num_items, memory_dimension, message_dimension,
                 token_dimension, link_texts, graph_texts, mean_time_shift_src, std_time_shift_src, mean_time_shift_dst,
                 std_time_shift_dst, neighbor_finder, node_raw_features, edge_raw_features, dataset_name):
        super(MemLLM, self).__init__()
        self.device = device
        self.dataset = dataset_name
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.LLM = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True,
                                                        torch_dtype=torch.bfloat16).to(
            device)
        # self.LLM = torch.nn.DataParallel(self.LLM)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
        self.tokenizer.pad_token = '[PAD]'
        max_length = self.tokenizer.model_max_length
        pad_token_id = self.tokenizer.pad_token_id
        # add tokens
        assert num_nodes == num_users + num_items, 'Wrong! num_nodes != num_users + num_items'
        new_tokens = []
        # for i in range(0, num_users + 1):
        #     new_token = '[' + 'user_' + str(i) + ']'
        #     new_tokens.append(new_token)
        # for i in range(1, num_items + 1):
        #     new_token = '[' + 'item_' + str(i) + ']'
        #     new_tokens.append(new_token)
        if dataset_name == 'yelpv2':
            new_tokens.append('[user]')
            new_tokens.append('[business]')
            new_tokens.append('[tij]')
        elif dataset_name == 'reddit':
            new_tokens.append('[user]')
            new_tokens.append('[subreddit]')
            new_tokens.append('[tij]')
        elif dataset_name == 'recipev2':
            new_tokens.append('[user]')
            new_tokens.append('[recipe]')
            new_tokens.append('[tij]')
        elif dataset_name == 'clothing':
            new_tokens.append('[user]')
            new_tokens.append('[clothes]')
            new_tokens.append('[tij]')
        num_added_tokens = self.tokenizer.add_tokens(new_tokens)
        print(f'Added {num_added_tokens} tokens')
        print(f"New vocabulary size: {len(self.tokenizer)}")
        self.LLM.resize_token_embeddings(len(self.tokenizer))

        self.num_nodes = num_nodes + 1
        self.memory_dimension = memory_dimension
        self.message_dimension = message_dimension
        self.memory = Memory(n_nodes=self.num_nodes,
                             memory_dimension=self.memory_dimension,
                             message_dimension=self.message_dimension,
                             device=device)
        # self.time_dimension = time_dimension
        self.time_dimension = memory_dimension  # token_dimension
        self.token_dimension = token_dimension
        self.time_encoder = TimeEncode(dimension=self.time_dimension)
        self.expand_memory_1 = ExpandMemory(input_dim=self.memory_dimension, output_dim=self.token_dimension)
        self.expand_memory_2 = ExpandMemory(input_dim=self.memory_dimension, output_dim=self.token_dimension)
        self.expand_memory_3 = ExpandMemory(input_dim=self.time_dimension, output_dim=self.token_dimension)
        self.shrink_message = ShrinkMessage(input_dim=token_dimension, output_dim=self.message_dimension)
        self.fix_embedding = FixEmbed(input_dim=token_dimension * 4, hidden_dim=token_dimension,
                                      output_dim=token_dimension)
        self.reduction = ReductionLayer(token_dimension, memory_dimension)
        self.link_texts = link_texts
        self.graph_texts = graph_texts
        if dataset_name == 'yelpv2' or dataset_name == 'recipev2' or dataset_name == 'clothing':
            assert len(self.link_texts) == len(
                self.graph_texts), f'wrong! len(self.link_texts){len(self.link_texts)} != len(self.graph_texts){len(self.graph_texts)}'
        elif dataset_name == 'reddit':
            assert len(self.link_texts) == 0, 'wrong! len(self.link_texts) != 0'

        # self.mean_time_shift_src = mean_time_shift_src
        # self.std_time_shift_src = std_time_shift_src
        # self.mean_time_shift_dst = mean_time_shift_dst
        # self.std_time_shift_dst = std_time_shift_dst
        self.neighbor_finder = neighbor_finder
        # self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        # self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        # self.n_node_features = self.node_raw_features.shape[1]
        # self.n_edge_features = self.edge_raw_features.shape[1]
        # self.time_encoder1 = TimeEncode(dimension=self.n_node_features)
        # self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
        #                                  self.n_node_features, 1)
        self.memory_updater = get_memory_updater(module_type="gru",
                                                 memory=self.memory,
                                                 message_dimension=self.memory_dimension * 4,
                                                 memory_dimension=self.memory_dimension,
                                                 device=device)
        self.message_aggregator = get_message_aggregator(aggregator_type='last',
                                                         device=device)
        # self.embedding_module = get_embedding_module(module_type='graph_attention',
        #                                              node_features=self.node_raw_features,
        #                                              edge_features=self.edge_raw_features,
        #                                              memory=self.memory,
        #                                              neighbor_finder=self.neighbor_finder,
        #                                              time_encoder=self.time_encoder1,
        #                                              n_layers=1,
        #                                              n_node_features=self.n_node_features,
        #                                              n_edge_features=self.n_edge_features,
        #                                              n_time_features=self.n_node_features,
        #                                              embedding_dimension=self.n_node_features,
        #                                              device=device,
        #                                              n_heads=2, dropout=0.1,
        #                                              use_memory=True,
        #                                              n_neighbors=10)
        self.embedding_module = get_llm_embedding(self.LLM, self.tokenizer, self.neighbor_finder, self.expand_memory_1,
                                                  self.expand_memory_2, self.expand_memory_3,
                                                  token_dimension, self.time_dimension, self.fix_embedding,
                                                  self.shrink_message, self.message_dimension, self.time_encoder,
                                                  device, dataset_name)

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                    edge_idxs, n_neighbors=20):
        memory, last_update = self.get_updated_memory(list(range(self.num_nodes)),
                                                      self.memory.messages)

        # Compute the embeddings using the embedding module
        # source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        #     source_nodes].long()
        # source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
        # destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        #     destination_nodes].long()
        # destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
        # negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        #     negative_nodes].long()
        # negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
        #
        # time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
        #                        dim=0)
        nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])
        n_samples = len(source_nodes)
        pos_score, neg_score = self.embedding_module.compute_embedding(memory, nodes, timestamps, n_samples,
                                                                       n_neighbors=n_neighbors)

        self.update_memory(positives, self.memory.messages)
        assert torch.allclose(memory[positives], self.memory.get_memory(positives),
                              atol=1e-3), "Something wrong in how the memory was updated"
        self.memory.clear_messages(positives)

        unique_sources, source_id_to_messages = self.compute_raw_messages(source_nodes,
                                                                          destination_nodes,
                                                                          edge_times, edge_idxs)
        unique_destinations, destination_id_to_messages = self.compute_raw_messages(destination_nodes,
                                                                                    source_nodes,
                                                                                    edge_times, edge_idxs)

        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)

        return pos_score, neg_score

    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                   edge_idxs, n_neighbors=20):
        n_samples = len(source_nodes)
        pos_score, neg_score = self.compute_temporal_embeddings(source_nodes, destination_nodes, negative_nodes,
                                                                edge_times, edge_idxs, n_neighbors)

        # score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
        #                             torch.cat([destination_node_embedding,
        #                                        negative_node_embedding])).squeeze(dim=0)
        # pos_score = score[:n_samples]
        # neg_score = score[n_samples:]

        return pos_score, neg_score

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes, messages)

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                          timestamps=unique_timestamps)

    def get_updated_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes, messages)

        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)

        return updated_memory, updated_last_update

    def preprocess_text(self, text):
        token_ids = self.tokenizer(text)['input_ids']
        token_ids = torch.tensor(token_ids).to(self.device)
        return token_ids

    def get_messages(self, source_nodes, destination_nodes, edge_times, edge_idxs):
        with torch.no_grad():
            edge_times = torch.from_numpy(edge_times).float().to(self.device)
            source_memory = self.memory.get_memory(source_nodes)
            destination_memory = self.memory.get_memory(destination_nodes)
            source_time_delta = edge_times - self.memory.last_update[source_nodes]
            source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(source_nodes),
                                                                                                    -1)

            edge_idxs = np.array(edge_idxs, dtype=int)
            graph_texts = np.array(self.graph_texts)[edge_idxs]
            if self.dataset == 'yelpv2' or self.dataset == 'recipev2' or self.dataset == 'clothing':
                link_texts = np.array(self.link_texts)[edge_idxs]
                # prompt + text
                texts = np.core.defchararray.add(graph_texts, link_texts)
            elif self.dataset == 'reddit':
                texts = graph_texts
            token_ids = [self.preprocess_text(text) for text in texts]
            token_ids = pad_sequence(token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            # mask
            attention_masks = (token_ids != self.tokenizer.pad_token_id).type(torch.float).to(self.device)
            # LLM
            token_embeddings = self.LLM.model.get_input_embeddings()(token_ids)

            # expand memory dim as token dim
            source_num = source_memory.shape[0]
            memory_input = torch.concat([source_memory, destination_memory], dim=0)
            memory_output = self.expand_memory(memory_input)
            source_memory = memory_output[0:source_num]
            destination_memory = memory_output[source_num:]
            # Integrate token, memory, and time.
            source_memory = source_memory.unsqueeze(1)  # Shape: (2, 1, 4096)
            destination_memory = destination_memory.unsqueeze(1)  # Shape: (2, 1, 4096)
            source_time_delta_encoding = source_time_delta_encoding.unsqueeze(1)  # Shape: (2, 1, 4096)
            # Expand the new dimension to match token_embeddings' second dimension
            dimm = token_embeddings.shape[1]
            source_memory = source_memory.expand(-1, dimm, -1)  # Shape: (2, 43, 4096)
            destination_memory = destination_memory.expand(-1, dimm, -1)  # Shape: (2, 43, 4096)
            source_time_delta_encoding = source_time_delta_encoding.expand(-1, dimm, -1)  # Shape: (2, 43, 4096)
            token_embeddings_bias = self.fix_embedding(
                torch.concat([token_embeddings, source_memory, destination_memory, source_time_delta_encoding], dim=2))
            # input LLM
            embeddings = \
                self.LLM.model(inputs_embeds=token_embeddings_bias.to(torch.bfloat16), attention_mask=attention_masks)[
                    0]

            # Obtain the average embedding of a sentence.
            attention_masks_sum = attention_masks.sum(dim=1, keepdim=True)
            sentence_embeddings = (embeddings * attention_masks.unsqueeze(-1)).sum(dim=1) / attention_masks_sum

            #  use sentence embedding as message
            sentence_messages = self.shrink_message(sentence_embeddings.to(torch.float32))

        # store message
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)
        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((sentence_messages[i], edge_times[i]))

        return unique_sources, messages

    def compute_raw_messages(self, src_node_ids, dst_node_ids, node_interact_times, edge_ids):
        """
        compute new raw messages for nodes in src_node_ids
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :return:
        """
        node_interact_times = torch.from_numpy(node_interact_times).float().to(self.device)
        edge_features = self.edge_raw_features[torch.from_numpy(edge_ids)]
        edge_features = self.reduction(edge_features)

        # Tensor, shape (batch_size, memory_dim)
        src_node_memories = self.memory.get_memory(src_node_ids)
        dst_node_memories = self.memory.get_memory(dst_node_ids)

        # Tensor, shape (batch_size, )
        src_node_delta_times = node_interact_times - \
                               self.memory.last_update[torch.from_numpy(src_node_ids)]
        # Tensor, shape (batch_size, time_feat_dim)
        src_node_delta_time_features = self.time_encoder(src_node_delta_times.unsqueeze(dim=1)).reshape(
            len(src_node_ids), -1)

        # Tensor, shape (batch_size, message_dim = memory_dim + memory_dim + time_feat_dim + edge_feat_dim)
        new_src_node_raw_messages = torch.cat(
            [src_node_memories, dst_node_memories, src_node_delta_time_features, edge_features], dim=1)

        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        new_node_raw_messages = defaultdict(list)
        # ndarray, shape (num_unique_node_ids, )
        unique_node_ids = np.unique(src_node_ids)

        for i in range(len(src_node_ids)):
            new_node_raw_messages[src_node_ids[i]].append((new_src_node_raw_messages[i], node_interact_times[i]))

        return unique_node_ids, new_node_raw_messages

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
