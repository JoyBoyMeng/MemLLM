import torch
from torch import nn
import numpy as np
import math

from temporal_attention import TemporalAttentionLayer
from torch.nn.utils.rnn import pad_sequence
from modules import Score

torch.set_printoptions(threshold=float('inf'))


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 dropout):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        # self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        return NotImplemented


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
        super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                            neighbor_finder, time_encoder, n_layers,
                                            n_node_features, n_edge_features, n_time_features,
                                            embedding_dimension, device, dropout)

        class NormalLinear(nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.n_node_features)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

        return source_embeddings


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                             neighbor_finder, time_encoder, n_layers,
                                             n_node_features, n_edge_features, n_time_features,
                                             embedding_dimension, device, dropout)

        self.use_memory = use_memory
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        """Recursive implementation of curr_layers temporal graph attention layers.

        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """

        assert (n_layers >= 0)

        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
            timestamps_torch))

        source_node_features = self.node_features[source_nodes_torch, :]

        if self.use_memory:
            source_node_features = memory[source_nodes, :] + source_node_features

        if n_layers == 0:
            return source_node_features
        else:

            source_node_conv_embeddings = self.compute_embedding(memory,
                                                                 source_nodes,
                                                                 timestamps,
                                                                 n_layers=n_layers - 1,
                                                                 n_neighbors=n_neighbors)

            neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
                source_nodes,
                timestamps,
                n_neighbors=n_neighbors)

            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

            edge_deltas = timestamps[:, np.newaxis] - edge_times

            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            neighbors = neighbors.flatten()
            neighbor_embeddings = self.compute_embedding(memory,
                                                         neighbors,
                                                         np.repeat(timestamps, n_neighbors),
                                                         n_layers=n_layers - 1,
                                                         n_neighbors=n_neighbors)

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
            edge_time_embeddings = self.time_encoder(edge_deltas_torch)

            edge_features = self.edge_features[edge_idxs, :]

            mask = neighbors_torch == 0

            source_embedding = self.aggregate(n_layers, source_node_conv_embeddings,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_features,
                                              mask)

            return source_embedding

    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        return NotImplemented


class GraphSumEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                                edge_features=edge_features,
                                                memory=memory,
                                                neighbor_finder=neighbor_finder,
                                                time_encoder=time_encoder, n_layers=n_layers,
                                                n_node_features=n_node_features,
                                                n_edge_features=n_edge_features,
                                                n_time_features=n_time_features,
                                                embedding_dimension=embedding_dimension,
                                                device=device,
                                                n_heads=n_heads, dropout=dropout,
                                                use_memory=use_memory)
        self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                             n_edge_features, embedding_dimension)
                                             for _ in range(n_layers)])
        self.linear_2 = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                             embedding_dimension) for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                       dim=2)
        neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
        neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

        source_features = torch.cat([source_node_features,
                                     source_nodes_time_embedding.squeeze()], dim=1)
        source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
        source_embedding = self.linear_2[n_layer - 1](source_embedding)

        return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                      neighbor_finder, time_encoder, n_layers,
                                                      n_node_features, n_edge_features,
                                                      n_time_features,
                                                      embedding_dimension, device,
                                                      n_heads, dropout,
                                                      use_memory)

        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            n_edge_features=n_edge_features,
            time_dim=n_time_features,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=n_node_features)
            for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        attention_model = self.attention_models[n_layer - 1]

        source_embedding, _ = attention_model(source_node_features,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_features,
                                              mask)

        return source_embedding


class LLMEmbedding(nn.Module):
    def __init__(self, LLM, tokenizer, neighbor_finder, expand_memory_1, expand_memory_2, expand_memory_3,
                 token_dimension, time_dimension, fix_embedding,
                 shrink_message, message_dim, time_encoder, device, dataset):
        super(LLMEmbedding, self).__init__()
        self.LLM = LLM
        self.tokenizer = tokenizer
        self.neighbor_finder = neighbor_finder
        self.expand_memory_1 = expand_memory_1
        self.expand_memory_2 = expand_memory_2
        self.expand_time = expand_memory_3
        self.token_dimension = token_dimension
        self.time_dimension = time_dimension
        self.time_encoder = time_encoder
        self.fix_embedding = fix_embedding
        self.shrink_message = shrink_message
        self.score = Score(input_dim=message_dim, hidden_dim=64, output_dim=1)
        self.device = device
        self.dataset = dataset

    def preprocess_text(self, text):
        token_ids = self.tokenizer(text)['input_ids']
        token_ids = torch.tensor(token_ids).to(self.device)
        return token_ids

    def reshape(self, source_nodes, destination_nodes, source_neighbors, destination_neighbors, memory,
                len_token_ids, source_timestamps, source_edge_times, destination_edge_times):
        for i in range(len(source_nodes)):
            source_node = source_nodes[i]
            destination_node = destination_nodes[i]
            timestamp = source_timestamps[i]

            source_neighbor = source_neighbors[i]
            source_edge_time = source_edge_times[i]
            destination_neighbor = destination_neighbors[i]
            destination_edge_time = destination_edge_times[i]

            len_source_neighbor = len(source_neighbor)
            len_destination_neighbor = len(destination_neighbor)
            if self.dataset == 'yelpv2':
                p1, p2, p3, p4, p5 = 11, 14, 12, 14, 15
            elif self.dataset == 'reddit':
                p1, p2, p3, p4, p5 = 13, 21, 13, 21, 20
            elif self.dataset == 'recipev2':
                p1, p2, p3, p4, p5 = 11, 15, 12, 15, 16
            elif self.dataset == 'clothing':
                p1, p2, p3, p4, p5 = 11, 14, 13, 14, 15

            if len_source_neighbor < 10:
                len_tokens_prompt1 = p1
            else:
                len_tokens_prompt1 = p1 + 1
            if len_destination_neighbor < 10:
                len_tokens_prompt3 = p3
            else:
                len_tokens_prompt3 = p3 + 1
            len_tokens_prompt2 = p2
            len_tokens_prompt4 = p4
            len_tokens_prompt5 = p5

            tmpnodes_1 = np.array([source_node])
            tmpnodes_2 = np.array([destination_node])
            tmptimes_ago = np.array([timestamp])
            if len_source_neighbor != 0:
                tmpnodes_2 = np.concatenate([tmpnodes_2, source_neighbor])
                tmptimes_ago = np.concatenate([tmptimes_ago, source_edge_time])
            if len_destination_neighbor != 0:
                tmpnodes_1 = np.concatenate([tmpnodes_1, destination_neighbor])
                tmptimes_ago = np.concatenate([tmptimes_ago, destination_edge_time])

            memory_input_1 = memory[tmpnodes_1]
            memory_output_1 = self.expand_memory_1(memory_input_1)
            memory_input_2 = memory[tmpnodes_2]
            memory_output_2 = self.expand_memory_2(memory_input_2)

            tmptimes_now = np.full(len(tmptimes_ago), timestamp)
            time_delta = tmptimes_now - tmptimes_ago
            time_delta = torch.from_numpy(time_delta).float().to(self.device)
            time_delta_encoding = self.time_encoder(time_delta.unsqueeze(dim=1)).view(len(tmptimes_ago), -1)
            time_delta_encoding = self.expand_time(time_delta_encoding)

            # bls
            memory_bls_src = memory_output_1[0].unsqueeze(0)
            memory_bls_dst = memory_output_2[0].unsqueeze(0)
            time_bls_encoding = time_delta_encoding[0].unsqueeze(0)
            # print(f'BLS size is {memory_bls_src.shape[0]}')

            # prompt1
            memory_prompt1_src = memory_prompt1_dst = memory_output_1[0].unsqueeze(0).expand(len_tokens_prompt1, -1)
            time_prompt1_encoding = time_delta_encoding[0].unsqueeze(0).expand(len_tokens_prompt1, -1)
            # print(f'prompt1 size is {memory_prompt1_src.shape[0]}')

            # prompt2
            if len_source_neighbor != 0:
                memory_prompt2_dst = memory_output_2[1:1 + len_source_neighbor]
                memory_prompt2_dst = memory_prompt2_dst.unsqueeze(1).expand(-1, len_tokens_prompt2, -1).reshape(-1,
                                                                                                                self.token_dimension)
                memory_prompt2_src = memory_output_1[0].unsqueeze(0).expand(memory_prompt2_dst.shape[0], -1)
                time_prompt2_encoding = time_delta_encoding[1:1 + len_source_neighbor]
                time_prompt2_encoding = time_prompt2_encoding.unsqueeze(1).expand(-1, len_tokens_prompt2, -1).reshape(
                    -1, self.token_dimension)
                # print(f'prompt2 size is {memory_prompt2_src.shape[0]}')

            # prompt3
            memory_prompt3_src = memory_prompt3_dst = memory_output_2[0].unsqueeze(0).expand(len_tokens_prompt3, -1)
            time_prompt3_encoding = time_delta_encoding[0].unsqueeze(0).expand(len_tokens_prompt3, -1)
            # print(f'prompt3 size is {memory_prompt3_src.shape[0]}')

            # prompt4
            if len_destination_neighbor != 0:
                memory_prompt4_dst = memory_output_1[1:1 + len_destination_neighbor]
                memory_prompt4_dst = memory_prompt4_dst.unsqueeze(1).expand(-1, len_tokens_prompt4, -1).reshape(-1,
                                                                                                                self.token_dimension)
                memory_prompt4_src = memory_output_2[0].unsqueeze(0).expand(memory_prompt4_dst.shape[0], -1)
                time_prompt4_encoding = time_delta_encoding[1 + len_source_neighbor:]
                time_prompt4_encoding = time_prompt4_encoding.unsqueeze(1).expand(-1, len_tokens_prompt4, -1).reshape(
                    -1, self.token_dimension)
                # print(f'prompt4 size is {memory_prompt4_src.shape[0]}')

            # prompt5
            memory_prompt5_src = memory_output_1[0].unsqueeze(0).expand(len_tokens_prompt5, -1)
            memory_prompt5_dst = memory_output_2[0].unsqueeze(0).expand(len_tokens_prompt5, -1)
            time_prompt5_encoding = time_delta_encoding[0].unsqueeze(0).expand(len_tokens_prompt5, -1)
            # print(f'prompt5 size is {memory_prompt5_src.shape[0]}')

            memory_src = torch.cat((memory_bls_src, memory_prompt1_src), dim=0)
            memory_dst = torch.cat((memory_bls_dst, memory_prompt1_dst), dim=0)
            time_encoding = torch.cat((time_bls_encoding, time_prompt1_encoding), dim=0)
            if len_source_neighbor != 0:
                memory_src = torch.cat((memory_src, memory_prompt2_src), dim=0)
                memory_dst = torch.cat((memory_dst, memory_prompt2_dst), dim=0)
                time_encoding = torch.cat((time_encoding, time_prompt2_encoding), dim=0)
            memory_src = torch.cat((memory_src, memory_prompt3_src), dim=0)
            memory_dst = torch.cat((memory_dst, memory_prompt3_dst), dim=0)
            time_encoding = torch.cat((time_encoding, time_prompt3_encoding), dim=0)
            if len_destination_neighbor != 0:
                memory_src = torch.cat((memory_src, memory_prompt4_src), dim=0)
                memory_dst = torch.cat((memory_dst, memory_prompt4_dst), dim=0)
                time_encoding = torch.cat((time_encoding, time_prompt4_encoding), dim=0)
            memory_src = torch.cat((memory_src, memory_prompt5_src), dim=0)
            memory_dst = torch.cat((memory_dst, memory_prompt5_dst), dim=0)
            time_encoding = torch.cat((time_encoding, time_prompt5_encoding), dim=0)

            dim_mem_src = memory_src.shape[0]
            dim_mem_dst = memory_dst.shape[0]
            dim_mem_time = time_encoding.shape[0]
            assert dim_mem_src == dim_mem_dst, f'dim_mem_src is {dim_mem_src}, dim_mem_dst is {dim_mem_dst}'
            assert dim_mem_src == dim_mem_time, f'dim_mem_src is {dim_mem_src}, dim_mem_time is {dim_mem_time}'
            assert dim_mem_src <= len_token_ids, f'reshape wrong: dim_mem {dim_mem_src}, len_token_ids {len_token_ids}'

            if dim_mem_src < len_token_ids:
                fill = len_token_ids - dim_mem_src
                memory_prompt6_src = memory_output_1[0].unsqueeze(0).expand(fill, -1)
                memory_prompt6_dst = memory_output_2[0].unsqueeze(0).expand(fill, -1)
                time_prompt6_encoding = time_delta_encoding[0].unsqueeze(0).expand(fill, -1)
                memory_src = torch.cat((memory_src, memory_prompt6_src), dim=0)
                memory_dst = torch.cat((memory_dst, memory_prompt6_dst), dim=0)
                time_encoding = torch.cat((time_encoding, time_prompt6_encoding), dim=0)

            if i == 0:
                memory_src_all = memory_src.unsqueeze(0)
                memory_dst_all = memory_dst.unsqueeze(0)
                time_encoding_all = time_encoding.unsqueeze(0)
            else:
                memory_src_all = torch.cat((memory_src_all, memory_src.unsqueeze(0)), dim=0)
                memory_dst_all = torch.cat((memory_dst_all, memory_dst.unsqueeze(0)), dim=0)
                time_encoding_all = torch.cat((time_encoding_all, time_encoding.unsqueeze(0)), dim=0)

        return memory_src_all, memory_dst_all, time_encoding_all

    def compute_embedding(self, memory, nodes, timestamps, n_samples, n_neighbors=10):
        texts, neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(nodes, timestamps,
                                                                                             n_samples,
                                                                                             self.dataset,
                                                                                             n_neighbors=n_neighbors)
        assert (len(nodes) == len(texts))
        assert (len(nodes) == len(neighbors))
        assert (len(nodes) == len(edge_idxs))
        assert (len(nodes) == len(edge_times))
        assert (len(nodes) == len(timestamps))

        # prompt_user = 'User [user] has posted ' + str(len(neighbors)) + ' comments recently.\n'
        # history_user = 'User [user] commented on business [business] at time [tij].\n'
        # prompt_bus = 'Business [business] has received ' + str(len(neighbors)) + ' comments recently.\n'
        # history_bus = 'User [user] commented on business [business] at time [tij].\n'
        # prompt_qs = 'Will user [user] comment on business [business] at time [tij]?\n'

        source_texts = texts[:n_samples]
        destination_texts = texts[n_samples:2 * n_samples]
        sample_texts = texts[2 * n_samples:]

        source_neighbors = neighbors[:n_samples]
        destination_neighbors = neighbors[n_samples:2 * n_samples]
        sample_neighbors = neighbors[2 * n_samples:]

        source_edge_times = edge_times[:n_samples]
        destination_edge_times = edge_times[n_samples:2 * n_samples]
        sample_edge_times = edge_times[2 * n_samples:]

        source_nodes = nodes[:n_samples]
        destination_nodes = nodes[n_samples:2 * n_samples]
        sample_nodes = nodes[2 * n_samples:]

        source_timestamps = timestamps[:n_samples]
        # destination_timestamps = timestamps[n_samples:2 * n_samples]
        # sample_timestamps = timestamps[2 * n_samples:]

        assert len(source_texts) == len(destination_texts), 'source_texts != destination_texts in length'
        assert len(source_texts) == len(sample_texts), 'source_texts != negative_texts in length'

        positive_texts = np.core.defchararray.add(source_texts, destination_texts)
        # print(positive_texts)
        negative_texts = np.core.defchararray.add(source_texts, sample_texts)

        positive_token_ids = [self.preprocess_text(text) for text in positive_texts]
        # print(positive_token_ids)
        positive_token_ids = pad_sequence(positive_token_ids, batch_first=True,
                                          padding_value=self.tokenizer.pad_token_id)
        len_positive_token_ids = positive_token_ids.shape[1]
        positive_attention_masks = (positive_token_ids != self.tokenizer.pad_token_id).type(torch.float).to(self.device)
        positive_token_embeddings = self.LLM.model.get_input_embeddings()(positive_token_ids)
        positive_memory_src, positive_memory_dst, positive_time_encoding = self.reshape(source_nodes,
                                                                                        destination_nodes,
                                                                                        source_neighbors,
                                                                                        destination_neighbors,
                                                                                        memory,
                                                                                        len_positive_token_ids,
                                                                                        source_timestamps,
                                                                                        source_edge_times,
                                                                                        destination_edge_times)
        positive_token_embeddings_bias = self.fix_embedding(
            torch.concat([positive_token_embeddings, positive_memory_src, positive_memory_dst, positive_time_encoding],
                         dim=2))
        positive_embeddings = \
            self.LLM.model(inputs_embeds=positive_token_embeddings_bias.to(torch.bfloat16),
                           attention_mask=positive_attention_masks)[0]
        positive_attention_masks_sum = positive_attention_masks.sum(dim=1, keepdim=True)
        positive_sentence_embeddings = (positive_embeddings * positive_attention_masks.unsqueeze(-1)).sum(
            dim=1) / positive_attention_masks_sum
        positive_sentence_messages = self.shrink_message(positive_sentence_embeddings.to(torch.float32))

        negative_token_ids = [self.preprocess_text(text) for text in negative_texts]
        negative_token_ids = pad_sequence(negative_token_ids, batch_first=True,
                                          padding_value=self.tokenizer.pad_token_id)
        len_negative_token_ids = negative_token_ids.shape[1]
        negative_attention_masks = (negative_token_ids != self.tokenizer.pad_token_id).type(torch.float).to(self.device)
        negative_token_embeddings = self.LLM.model.get_input_embeddings()(negative_token_ids)
        negative_memory_src, negative_memory_dst, negative_time_encoding = self.reshape(source_nodes,
                                                                                        sample_nodes,
                                                                                        source_neighbors,
                                                                                        sample_neighbors,
                                                                                        memory,
                                                                                        len_negative_token_ids,
                                                                                        source_timestamps,
                                                                                        source_edge_times,
                                                                                        sample_edge_times)
        negative_token_embeddings_bias = self.fix_embedding(
            torch.concat([negative_token_embeddings, negative_memory_src, negative_memory_dst, negative_time_encoding],
                         dim=2))
        negative_embeddings = \
            self.LLM.model(inputs_embeds=negative_token_embeddings_bias.to(torch.bfloat16),
                           attention_mask=negative_attention_masks)[0]
        negative_attention_masks_sum = negative_attention_masks.sum(dim=1, keepdim=True)
        negative_sentence_embeddings = (negative_embeddings * negative_attention_masks.unsqueeze(-1)).sum(
            dim=1) / negative_attention_masks_sum
        negative_sentence_messages = self.shrink_message(negative_sentence_embeddings.to(torch.float32))

        sentence_messages = torch.cat([positive_sentence_messages, negative_sentence_messages], dim=0)
        score = self.score(sentence_messages)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        return pos_score.sigmoid(), neg_score.sigmoid()


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(node_features=node_features,
                                       edge_features=edge_features,
                                       memory=memory,
                                       neighbor_finder=neighbor_finder,
                                       time_encoder=time_encoder,
                                       n_layers=n_layers,
                                       n_node_features=n_node_features,
                                       n_edge_features=n_edge_features,
                                       n_time_features=n_time_features,
                                       embedding_dimension=embedding_dimension,
                                       device=device, n_heads=n_heads,
                                       dropout=dropout, use_memory=use_memory)
    elif module_type == "graph_sum":
        return GraphSumEmbedding(node_features=node_features,
                                 edge_features=edge_features,
                                 memory=memory,
                                 neighbor_finder=neighbor_finder,
                                 time_encoder=time_encoder,
                                 n_layers=n_layers,
                                 n_node_features=n_node_features,
                                 n_edge_features=n_edge_features,
                                 n_time_features=n_time_features,
                                 embedding_dimension=embedding_dimension,
                                 device=device,
                                 n_heads=n_heads, dropout=dropout, use_memory=use_memory)

    elif module_type == "identity":
        return IdentityEmbedding(node_features=node_features,
                                 edge_features=edge_features,
                                 memory=memory,
                                 neighbor_finder=neighbor_finder,
                                 time_encoder=time_encoder,
                                 n_layers=n_layers,
                                 n_node_features=n_node_features,
                                 n_edge_features=n_edge_features,
                                 n_time_features=n_time_features,
                                 embedding_dimension=embedding_dimension,
                                 device=device,
                                 dropout=dropout)
    elif module_type == "time":
        return TimeEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout,
                             n_neighbors=n_neighbors)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))


def get_llm_embedding(LLM, tokenizer, neighbor_finder, expand_memory_1, expand_memory_2, expand_memory_3,
                      token_dimension, time_dimension, fix_embedding,
                      shrink_message, message_dim, time_encoder, device, dataset):
    return LLMEmbedding(LLM, tokenizer, neighbor_finder, expand_memory_1, expand_memory_2, expand_memory_3,
                        token_dimension, time_dimension, fix_embedding,
                        shrink_message, message_dim, time_encoder, device, dataset)
