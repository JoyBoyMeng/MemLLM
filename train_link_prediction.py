import torch
import numpy as np
import argparse
import sys
import math
import time
from pathlib import Path

from make_datasets import get_data
from utils import get_neighbor_finder, RandEdgeSampler, compute_time_statistics, EarlyStopMonitor
from memllm import MemLLM
from evaluation import eval_edge_prediction


def get_memory_size(tensor):
    """Returns the size of the tensor in bytes."""
    return tensor.element_size() * tensor.nelement()


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser('LLM for social network link prediction training')
    parser.add_argument('-d', '--data', type=str, help='Dataset name',
                        default='yelp')
    parser.add_argument('--bs', type=int, default=2, help='Batch_size')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--llm_path', type=str, default='../LLM/shakechen/Llama-2-7b-hf', help='Pretrained LLM path')
    parser.add_argument('--mem_dim', type=int, default=172, help='Dimensions of the memory')
    parser.add_argument('--mess_dim', type=int, default=100, help='Dimensions of the message')
    parser.add_argument('--token_dim', type=int, default=4096, help='Dimensions of the message')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                      'backprop')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    dataset_name = args.data
    GPU = args.gpu
    llm_path = args.llm_path
    memory_dimension = args.mem_dim
    message_dimension = args.mess_dim
    token_dimension = args.token_dim
    batch_size = args.bs
    n_epoch = args.n_epoch
    n_neighbor = args.n_degree
    backprop_every = args.backprop_every

    # Path("./saved_models/").mkdir(parents=True, exist_ok=True)
    # Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    # MODEL_SAVE_PATH = f'./saved_models/{args.data}.pth'
    # get_checkpoint_path = lambda \
    #         epoch: f'./saved_checkpoints/{args.data}-{epoch}.pth'

    graph_texts, review_texts, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, edge_raw_features = get_data(
        dataset_name)

    train_ngh_finder = get_neighbor_finder(train_data)
    full_ngh_finder = get_neighbor_finder(full_data)

    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations, seed=4)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    # Compute time statistics
    # mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    #     compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = [], [], [], []

    num_users = len(set(full_data.sources))
    num_items = len(set(full_data.destinations))
    num_nodes = full_data.n_unique_nodes
    num_edges = full_data.n_interactions
    # node_raw_features = np.zeros((num_nodes + 1, 172))
    # edge_raw_features = np.zeros((num_edges + 1, 172))
    node_raw_features = []
    # edge_raw_features = []

    memllm = MemLLM(llm_path, device, num_nodes, num_users, num_items, memory_dimension, message_dimension,
                    token_dimension, review_texts, graph_texts, mean_time_shift_src, std_time_shift_src,
                    mean_time_shift_dst,
                    std_time_shift_dst, train_ngh_finder, node_raw_features, edge_raw_features, dataset_name)
    memllm.to(device)


    print(memllm)
    # state_dict = memllm.state_dict()
    # for param_name, param_tensor in state_dict.items():
    #     param_size = param_tensor.size()
    #     memory_size = get_memory_size(param_tensor) / (1024 ** 2)  #
    #     print(f"Parameter Name: {param_name}\tSize: {param_size}\tMemory Size: {memory_size:.2f} MB")
    # exit()


    # for name, param in memllm.named_parameters():
    #     for i in range(0, 32):
    #         layer_th = 'layers.' + str(i) + '.'
    #         if layer_th in name:
    #             param.requires_grad = False
    #     if 'model.norm.weight' in name:
    #         param.requires_grad = False
    for name, param in memllm.LLM.named_parameters():
        # if '.0.' in name:
        #     continue
        # if '.1.' in name:
        #     continue
        # if '.2.' in name:
        #     continue
        # if '.3.' in name:
        #     continue
        param.requires_grad = False


    print("\nFrozen parameters:")
    for name, param in memllm.named_parameters():
        if not param.requires_grad:
            print(name)
    # params_before_update = {name: param.clone() for name, param in memllm.named_parameters()}


    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, memllm.parameters()), lr=args.lr)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / batch_size)
    new_nodes_val_aps = []
    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []

    early_stopper = EarlyStopMonitor(max_round=args.patience)

    for epoch in range(n_epoch):
        start_epoch = time.time()
        memllm.memory.__init_memory__()
        memllm.set_neighbor_finder(train_ngh_finder)
        m_loss = []

        for k in range(0, num_batch, backprop_every):
            loss = 0
            optimizer.zero_grad()
            for j in range(backprop_every):
                batch_idx = k + j
                if batch_idx >= num_batch:
                    continue
                print(f'{batch_idx} / {num_batch - 1}')
                start_idx = batch_idx * batch_size
                end_idx = min(num_instance, start_idx + batch_size)
                sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], train_data.destinations[
                                                                                           start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]

                size = len(sources_batch)
                _, negatives_batch = train_rand_sampler.sample(size)

                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=device)

                memllm = memllm.train()

                pos_prob, neg_prob = memllm.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                       negatives_batch,
                                                                       timestamps_batch, edge_idxs_batch, n_neighbor)

                loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
            loss /= args.backprop_every
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

            memllm.memory.detach_memory()

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        # Validation uses the full graph
        memllm.set_neighbor_finder(full_ngh_finder)
        train_memory_backup = memllm.memory.backup_memory()
        val_ap, val_auc = eval_edge_prediction(model=memllm,
                                               negative_edge_sampler=val_rand_sampler,
                                               data=val_data,
                                               n_neighbors=n_neighbor,
                                               batch_size=batch_size)
        val_memory_backup = memllm.memory.backup_memory()
        memllm.memory.restore_memory(train_memory_backup)
        # Validate on unseen nodes
        nn_val_ap, nn_val_auc = eval_edge_prediction(model=memllm,
                                                     negative_edge_sampler=val_rand_sampler,
                                                     data=new_node_val_data,
                                                     n_neighbors=n_neighbor,
                                                     batch_size=batch_size)
        memllm.memory.restore_memory(val_memory_backup)

        new_nodes_val_aps.append(nn_val_ap)
        val_aps.append(val_ap)
        train_losses.append(np.mean(m_loss))

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        print('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        print('Epoch mean loss: {}'.format(np.mean(m_loss)))
        print('val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
        print('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

        if early_stopper.early_stop_check(val_auc):
            print('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            print(f'The best model is at epoch {early_stopper.best_epoch}')
            break

        memllm.eval()
        val_memory_backup = memllm.memory.backup_memory()
        memllm.embedding_module.neighbor_finder = full_ngh_finder
        test_ap, test_auc = eval_edge_prediction(model=memllm,
                                                 negative_edge_sampler=test_rand_sampler,
                                                 data=test_data,
                                                 n_neighbors=n_neighbor,
                                                 batch_size=batch_size)
        memllm.memory.restore_memory(val_memory_backup)
        nn_test_ap, nn_test_auc = eval_edge_prediction(model=memllm,
                                                       negative_edge_sampler=nn_test_rand_sampler,
                                                       data=new_node_test_data,
                                                       n_neighbors=n_neighbor,
                                                       batch_size=batch_size)
        print('Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
        print('Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))

    print('Saving memllm model')
    memllm.memory.restore_memory(val_memory_backup)
    Path("./saved_models/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f'./saved_models/{args.data}.pth'
    torch.save(memllm.state_dict(), MODEL_SAVE_PATH)
    print('memllm model saved')
