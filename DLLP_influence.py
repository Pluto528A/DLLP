import numpy as np
import math
import random
import argparse
import time
import copy
import warnings
import torch
from torch.nn import BCEWithLogitsLoss

SIGMOID_BOUND = 500
SIGMOID_TABLE_SIZE = 625000
sigmoid_table = [0.0] * (SIGMOID_TABLE_SIZE + 1)

embedding_size = 0
node_size = 0


def initSigmoidTable():
    global sigmoid_table
    for k in range(SIGMOID_TABLE_SIZE + 1):
        x = 2.0 * SIGMOID_BOUND * k / SIGMOID_TABLE_SIZE - SIGMOID_BOUND
        sigmoid_table[k] = 1.0 / (1.0 + math.exp(-x))

def fastSigmoid(e1, e2):
    x = np.dot(np.array(e1, dtype=np.float64), np.array(e2, dtype=np.float64))
    if x > SIGMOID_BOUND:
        return 1.0
    elif x < -SIGMOID_BOUND:
        return 0.0
    idx = int((x + SIGMOID_BOUND) * SIGMOID_TABLE_SIZE / (2 * SIGMOID_BOUND))
    return sigmoid_table[idx]


def loadGraph(graph):
    links, weights = [], []
    with open(graph, 'r') as f:
        for l in f:
            l = l.strip().split(' ')
            u, v = int(l[0]), int(l[1])
            links.append((u, v))
            if len(l) < 3:
                weights.append(1.0)
            else:
                weights.append(np.float64(l[2]))
    return links, weights


def loadEmbedding(path):
    global node_size, embedding_size
    with open(path, 'r') as f:
        header = f.readline().strip().split()
        node_size, embedding_size = map(int, header)
        emb = [[] for _ in range(node_size)]
        for line in f:
            parts = line.strip().split()
            nid = int(parts[0])
            emb[nid] = [np.float64(x) for x in parts[1:]]
    return emb


def compute_high_order_influence(emb, links, weights, target_edge, daweights, sample_iters=2000, burn_in=1000, agg_interval=20):
    global node_size, embedding_size
    v1, v2 = target_edge

    vlist = [v1, v2]

    loss = np.float64(daweights) * (fastSigmoid(emb[v1], emb[v2]) - 1.0)

    g_loss = np.zeros(node_size * embedding_size, dtype=np.float64)
    for i in range(embedding_size):
        g_loss[v1 * embedding_size + i] = loss * emb[v2][i]
        g_loss[v2 * embedding_size + i] = loss * emb[v1][i]

    Hi_g_loss = g_loss.copy()
    ave_Hi_g_loss = np.zeros(node_size * embedding_size, dtype=np.float64)
    iters = 0

    while True:
        sample_id = int(random.random() * len(links))
        sv1, sv2 = links[sample_id]
        if sv1 not in vlist and sv2 not in vlist:
            continue

        s = fastSigmoid(emb[sv1], emb[sv2])
        sample_loss = np.float64(weights[sample_id]) * s * (1.0 - s)

        sample_H_g_loss = np.zeros(node_size * embedding_size, dtype=np.float64)
        sv1_w2 = np.dot(
            np.array(emb[sv2], dtype=np.float64),
            Hi_g_loss[sv1 * embedding_size:(sv1 + 1) * embedding_size]
        )
        sv2_w1 = np.dot(
            np.array(emb[sv1], dtype=np.float64),
            Hi_g_loss[sv2 * embedding_size:(sv2 + 1) * embedding_size]
        )

        if sv1 in vlist:
            for i in range(embedding_size):
                sample_H_g_loss[sv2 * embedding_size + i] += sample_loss * emb[sv1][i] * sv1_w2
                sample_H_g_loss[sv1 * embedding_size + i] += sample_loss * emb[sv2][i] * sv1_w2

        if sv2 in vlist:
            for i in range(embedding_size):
                sample_H_g_loss[sv2 * embedding_size + i] += sample_loss * emb[sv1][i] * sv2_w1
                sample_H_g_loss[sv1 * embedding_size + i] += sample_loss * emb[sv2][i] * sv2_w1

        original_Hi_g_loss = copy.deepcopy(Hi_g_loss)
        warning_flag = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            Hi_g_loss = g_loss + Hi_g_loss - sample_H_g_loss

            warning_occurred = False
            for warning in w:
                if (issubclass(warning.category, RuntimeWarning) and
                        "invalid value encountered in subtract" in str(warning.message)):
                    warning_occurred = True
                    break

            if warning_occurred:
                Hi_g_loss = original_Hi_g_loss
                warning_flag = True

        if warning_flag:
            continue
        iters += 1

        if iters % 20 == 0 and iters > 1000:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error', category=RuntimeWarning, message='overflow encountered in add')

                    ave_Hi_g_loss += Hi_g_loss
            except RuntimeWarning as e:
                return copy.deepcopy(emb)
            except Exception as e:
                return copy.deepcopy(emb)
        if iters >= 2000:
            break

    ave_Hi_g_loss /= np.float64((iters - 1000) / 20 * len(links))

    daflag = -1.0
    processed_emb = []
    for i in range(node_size):
        new_vec = (
            np.array(emb[i], dtype=np.float64) +
            daflag * ave_Hi_g_loss[i * embedding_size:(i + 1) * embedding_size]
        ).tolist()
        processed_emb.append(new_vec)

    return processed_emb


def standardize_edge(u, v):
    return (min(u, v), max(u, v))


def sort_key(x):
    if math.isnan(x[2]):
        return (float('-inf'), x[0], x[1])
    else:
        return (x[2], x[0], x[1])


def get_scores(z, edge_list, device):
    u = torch.tensor([e[0] for e in edge_list], dtype=torch.long, device=device)
    v = torch.tensor([e[1] for e in edge_list], dtype=torch.long, device=device)

    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=torch.float64, device=device)
    else:
        z = z.to(dtype=torch.float64)

    return (z[u] * z[v]).sum(dim=1)


def load_edge_list(path):
    edges = []
    with open(path) as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            u, v = parts[0], parts[1]
            edges.append((u, v))
    return edges


def main():
    parser = argparse.ArgumentParser(description='Calculate only the high-order influence values of the presence of edges (float64 precision)')
    parser.add_argument('--dataname', default="Italypowergrid", type=str, help='graph name')
    parser.add_argument('--method', default="gcn", type=str, help='graph name')
    parser.add_argument('-iters', type=int, default=2000, help='The total number of higher-order sampling iterations')
    parser.add_argument('-burn', type=int, default=1000, help='burn-in')
    parser.add_argument('-interval', type=int, default=20, help='Sampling accumulation interval')
    args = parser.parse_args()

    graph = f"./data/{args.dataname}/{args.dataname}_train_embedding_graph.txt"
    embedding = f"./output/{args.method}/{args.dataname}/node_embeddings_{args.dataname}_{args.method}.txt"
    out_file = f"./output/{args.method}/{args.dataname}/exist_{args.dataname}_noabs_influence_sorted_{args.method}.txt"

    initSigmoidTable()

    links, weights = loadGraph(graph)

    emb = loadEmbedding(embedding)

    exist_infl = []

    start_time = time.time()
    batch_start = start_time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    E_pos_train = load_edge_list(graph)
    E_neg_train = load_edge_list(f"./data/{args.dataname}/{args.dataname}_link_predicate_graph_noexist_train.txt")

    loss_time_start = time.time()
    loss_fn = BCEWithLogitsLoss()
    emb_tensor = torch.tensor(emb, dtype=torch.float64, device=device)
    pos_scores = get_scores(emb_tensor, E_pos_train, device)
    neg_scores = get_scores(emb_tensor, E_neg_train, device)
    scores = torch.cat([pos_scores, neg_scores]).to(dtype=torch.float64)
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).to(dtype=torch.float64)
    loss_orignal = loss_fn(scores, labels)
    loss_time_end = time.time()

    batch_max_infl = -float('inf')
    batch_max_edge = None
    batch_min_infl = float('inf')
    batch_min_edge = None
    for idx, (u, v) in enumerate(links):
        std_edge = (u, v)
        w = np.float64(1.0)

        emb_new = compute_high_order_influence(
            emb, links, weights, std_edge, w,
            sample_iters=args.iters,
            burn_in=args.burn,
            agg_interval=args.interval
        )

        emb_new_tensor = torch.tensor(emb_new, dtype=torch.float64, device=device)

        pos_scores = get_scores(emb_new_tensor, E_pos_train, device)
        neg_scores = get_scores(emb_new_tensor, E_neg_train, device)
        scores = torch.cat([pos_scores, neg_scores]).to(dtype=torch.float64)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).to(dtype=torch.float64)
        loss_new = loss_fn(scores, labels)

        infl = (loss_new - loss_orignal).item()
        exist_infl.append((u, v, infl))

        if infl > batch_max_infl:
            batch_max_infl = infl
        if infl < batch_min_infl:
            batch_min_infl = infl

    exist_infl_sorted = sorted(exist_infl, key=sort_key)

    with open(out_file, 'w') as f:
        for u, v, infl in exist_infl_sorted:
            f.write(f"{u} {v} {infl}\n")


if __name__ == '__main__':
    # python .\near_new_influence.py
    main()