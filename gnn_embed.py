
from tqdm import tqdm
import argparse
import networkx as nx
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import from_networkx
from sklearn.metrics import roc_auc_score, average_precision_score

def load_graph(path):
    G = nx.Graph()
    with open(path) as f:
        for line in f:
            u, v, w = line.strip().split()
            G.add_edge(int(u), int(v), weight=float(w))
    return G


def load_edge_list(path):
    edges = []
    with open(path) as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            u, v = parts[0], parts[1]
            edges.append((u, v))
    return edges


def get_scores(z, edge_list, device):
    u = torch.tensor([e[0] for e in edge_list], dtype=torch.long, device=device)
    v = torch.tensor([e[1] for e in edge_list], dtype=torch.long, device=device)
    return (z[u] * z[v]).sum(dim=1)


def save_embeddings(z, path):
    N, D = z.size()
    with open(path, 'w') as f:
        f.write(f"{N} {D}\n")
        for idx in range(N):
            vec = " ".join(f"{v:.6f}" for v in z[idx].cpu().tolist())
            f.write(f"{idx} {vec}\n")


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid_ch)
        self.conv2 = GCNConv(hid_ch, out_ch)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class GAEncoder(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, 2*out_ch)
        self.conv2 = GCNConv(2*out_ch, out_ch)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


def main(args):
    G = load_graph(args.graph)
    E_pos_train = load_edge_list(args.graph)
    E_pos_test = load_edge_list(args.test_pos)
    E_neg_train = load_edge_list(args.neg_train)
    E_neg_test = load_edge_list(args.neg_test)

    data = from_networkx(G)
    num_nodes = G.number_of_nodes()
    data.x = torch.eye(num_nodes, dtype=torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    if args.method == 'gcn':
        model = GCNEncoder(data.num_features, args.hidden_dim, args.embed_dim).to(device)
        use_gae = False
    elif args.method == 'gae':
        model = GAE(GAEncoder(data.num_features, args.embed_dim)).to(device)
        use_gae = True
    else:
        raise ValueError("method must be 'gcn' or 'gae'")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = BCEWithLogitsLoss()

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training Epochs"):
        model.train()
        optimizer.zero_grad()

        if use_gae:
             z = model.encode(data.x, data.edge_index)
        else:
            z = model(data.x, data.edge_index)

        pos_scores = get_scores(z, E_pos_train, device)
        neg_scores = get_scores(z, E_neg_train, device)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()

        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                if use_gae:
                    z = model.encode(data.x, data.edge_index)
                else:
                    z = model(data.x, data.edge_index)
                pos_s = torch.sigmoid(get_scores(z, E_pos_test, device)).cpu().numpy()
                neg_s = torch.sigmoid(get_scores(z, E_neg_test, device)).cpu().numpy()
                y_true = [1]*len(pos_s) + [0]*len(neg_s)
                y_score = list(pos_s) + list(neg_s)
                auc = roc_auc_score(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
            print(f"[{args.method.upper()}] Epoch {epoch} | loss={loss.item():.6f} | test_auc={auc:.4f} | test_ap={ap:.4f}")

    if use_gae:
        z = model.encode(data.x, data.edge_index)
    else:
        z = model(data.x, data.edge_index)
    save_embeddings(z, args.out_embedding)
    print(f"Embedding saved to {args.out_embedding}")


if __name__ == "__main__":
    # python .\gnn_embed.py --method gcn
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default="Italypowergrid", type=str, help='graph name')
    parser.add_argument('--delete', type=int, default=30)

    parser.add_argument('--method', choices=['gcn', 'gae'], required=True,
                        help="Select a model:'gcn' 或 'gae'")
    parser.add_argument('--graph', type=str, required=False,
                        help="Training graph path（u v weight）")
    parser.add_argument('--test_pos', type=str, required=False,
                        help="Test set positive path")
    parser.add_argument('--neg_train', type=str, required=False,
                            help="The negative edge path of the training set")
    parser.add_argument('--neg_test', type=str, required=False,
                        help="Test set negative edge path")
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--out_embedding', type=str, required=False,
                        help="Save the embedded file path (.txt)")
    args = parser.parse_args()

    args.graph = f"./data/{args.dataname}/{args.dataname}_train_embedding_graph.txt"
    args.test_pos = f"./data/{args.dataname}/{args.dataname}_test_link_predicate_graph.txt"
    args.neg_test = f"./data/{args.dataname}/{args.dataname}_link_predicate_graph_noexist_test.txt"
    args.neg_train = f"./data/{args.dataname}/{args.dataname}_link_predicate_graph_noexist_train.txt"

    args.out_embedding = f"./output/{args.method}/{args.dataname}/node_embeddings_{args.dataname}_{args.method}.txt"

    main(args)