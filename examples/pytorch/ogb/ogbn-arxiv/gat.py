import argparse
import math
import time
import os
from tqdm import tqdm
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from torch.utils.tensorboard import SummaryWriter
import dgl
import sys
from models import GAT
from utils import *

set_logging_format()
version = None
epsilon = 1 - math.log(2)
device = None

n_node_feats, n_classes = 0, 0


def load_data(dataset):
    global n_node_feats, n_classes

    data = DglNodePropPredDataset(name=dataset, root=os.path.join(os.environ['HOME'], 'data', 'OGB'))
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph):
    global n_node_feats

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    return graph


def gen_model(args):
    model = GAT(
        n_node_feats + n_classes if args.use_labels else n_node_feats,
        n_classes,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        attn_drop=args.attn_drop,
        edge_drop=args.edge_drop,
        use_attn_dst=not args.no_attn_dst,
        use_symmetric_norm=args.use_norm,
    )

    return model


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]


def add_labels(feat, labels, idx):
    onehot = th.zeros([feat.shape[0], n_classes]).to(device)
    onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(model, graph, labels, train_idx, val_idx, test_idx, optimizer, use_labels, n_label_iters, args):
    model.train()

    feat = graph.ndata["feat"]

    if use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat)

    # randomly sample edges to calculate topology consistency
    mask = torch.randn(graph.num_edges(), device=pred.device) < args.topo_mask_threshold
    srcs = torch.masked_select(graph.edges()[0], mask)
    dsts = torch.masked_select(graph.edges()[1], mask)
    pred_srcs = pred[srcs]
    pred_dsts = pred[dsts]
    topo_consistency = torch.nn.functional.cosine_similarity(pred_srcs, pred_dsts).mean()
    topo_loss = 1 - topo_consistency

    if n_label_iters > 0:
        unlabel_idx = torch.cat([train_pred_idx, val_idx, test_idx])
        for _ in range(n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx].detach(), dim=-1)
            pred = model(graph, feat)

    ce_loss = custom_loss_function(pred[train_pred_idx], labels[train_pred_idx])
    loss = ce_loss + args.topo_loss_ratio * topo_loss

    loss.backward()
    optimizer.step()

    return ce_loss.item(), pred.detach(), topo_loss.item()


@th.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx, use_labels, n_label_iters, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)

    if n_label_iters > 0:
        unlabel_idx = torch.cat([val_idx, test_idx])
        for _ in range(n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat)

    train_loss = custom_loss_function(pred[train_idx], labels[train_idx])
    val_loss = custom_loss_function(pred[val_idx], labels[val_idx])
    test_loss = custom_loss_function(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running, log_file,
        tensorboard_writer):
    # define model and optimizer
    model = gen_model(args)
    model = model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training loop
    total_time = 0
    best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in tqdm(range(1, args.n_epochs + 1), desc=f'running {n_running}', ncols=80):
        tic = time.time()

        adjust_learning_rate(optimizer, args.lr, epoch)

        ce_loss, pred, topo_loss = train(
            model, graph, labels, train_idx, val_idx, test_idx, optimizer, args.use_labels,
            args.n_label_iters, args
        )
        acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = evaluate(
            model, graph, labels, train_idx, val_idx, test_idx, args.use_labels, args.n_label_iters, evaluator
        )

        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % args.log_every == 0:
            log_file.write(f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}\n")
            log_file.write(
                f"CELoss: {ce_loss:.4f}, TopoLoss: {topo_loss:.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Best test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{best_test_acc:.4f}\n"
            )
        tensorboard_writer.add_scalar(f'RUN{n_running}_train/loss', ce_loss, epoch)
        tensorboard_writer.add_scalar(f'RUN{n_running}_train/acc', acc, epoch)
        tensorboard_writer.add_scalars(f'RUN{n_running}_eval/loss', {'train': train_loss,
                                                                     'val': val_loss,
                                                                     'test': test_loss}, epoch)
        tensorboard_writer.add_scalars(f'RUN{n_running}_eval/acc', {'train': train_acc,
                                                                    'val': val_acc,
                                                                    'test': test_acc}, epoch)
        tensorboard_writer.add_scalars(f'RUN{n_running}_eval/best_acc', {'val': best_val_acc,
                                                                         'test': best_test_acc}, epoch)

        for l, e in zip(
                [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
                [acc, train_acc, val_acc, test_acc, ce_loss, train_loss, val_loss, test_loss],
        ):
            l.append(e)

    log_file.write("*" * 50)
    log_file.write(f'running {n_running}\n')
    log_file.write(f"Average epoch time: {total_time / args.n_epochs}, Test acc: {best_test_acc}\n")

    if args.plot_curves:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip([accs, train_accs, val_accs, test_accs],
                            ["acc", "train acc", "val acc", "test acc"]):
            plt.plot(range(args.n_epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_acc_{n_running}.png")

        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
                [losses, train_losses, val_losses, test_losses],
                ["loss", "train loss", "val loss", "test loss"]
        ):
            plt.plot(range(args.n_epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_loss_{n_running}.png")

    return best_val_acc, best_test_acc


def count_parameters(args):
    model = gen_model(args)
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device, n_node_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser("GAT on OGBN-Arxiv",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, help="running times", default=10)
    argparser.add_argument("--n-epochs", type=int, help="number of epochs", default=2000)

    argparser.add_argument("--use-labels", type=str2bool, default=True)
    argparser.add_argument("--n-label-iters", type=int, help="number of label iterations", default=0)
    argparser.add_argument("--no-attn-dst", type=str2bool, default=True)
    argparser.add_argument("--use-norm", type=str2bool, default=True)
    argparser.add_argument('--topo-loss-ratio', type=float, default=1.0)
    argparser.add_argument('--topo-mask-threshold', type=float, default=0.2)

    argparser.add_argument("--n-layers", type=int, help="number of layers", default=3)
    argparser.add_argument("--n-heads", type=int, help="number of heads", default=3)
    argparser.add_argument("--n-hidden", type=int, help="number of hidden units", default=250)

    argparser.add_argument("--dropout", type=float, help="dropout rate", default=0.75)
    argparser.add_argument("--input-drop", type=float, help="input drop rate", default=0.25)
    argparser.add_argument("--attn-drop", type=float, help="attention dropout rate", default=0.0)
    argparser.add_argument("--edge-drop", type=float, help="edge drop rate", default=0.3)

    argparser.add_argument("--lr", type=float, help="learning rate", default=0.002)
    argparser.add_argument("--wd", type=float, help="weight decay", default=0)
    argparser.add_argument("--log-every", type=int, help="log every LOG_EVERY epochs", default=1)
    argparser.add_argument("--plot-curves", help="plot learning curves", action="store_true")

    argparser.add_argument("--version", type=str, required=True)
    args = argparser.parse_args()

    if not args.use_labels and args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")

    path = 'log'
    if not os.path.exists(path):
        os.mkdir(path)
    version = f'{args.version}_{formatted_time()}'
    backup_code(path=path, version=version)
    log_file = open(os.path.join(path, version, 'log'), 'w', buffering=1)
    write_dict(args.__dict__, log_file, prefix=hint_line('args'))
    tensorboard_writer = SummaryWriter(log_dir=os.path.join(path, version, 'tensorboard'))

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # load data
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data('ogbn-arxiv')
    print(graph, file=log_file)
    graph = preprocess(graph)
    log_file.write(f"Number of params: {count_parameters(args)}\n")

    graph = graph.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    # run
    val_accs = []
    test_accs = []

    for n_running in range(1, args.n_runs + 1):
        # seed from 0~9 according to the OGB leaderboard instruction.
        seed = n_running - 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)

        val_acc, test_acc = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running,
                                log_file, tensorboard_writer)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    log_file.write(f"Runned {args.n_runs} times\n")
    log_file.write(f"Val Accs: {val_accs}\n")
    log_file.write(f"Test Accs: {test_accs}\n")
    log_file.write(f"Average val accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}\n")
    log_file.write(f"Average test accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}\n")
    log_file.write(f"Number¢∞ of params: {count_parameters(args)}\n")


if __name__ == "__main__":
    # set random seed=42
    # seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    print(' '.join(sys.argv))
    main()

"""
Run this file with command: 
python -u gat.py --gpu=0 --n-label-iters=1 --dropout=0.75 --input-drop=0.25 --edge-drop=0.3 --topo-mask-threshold=0.2 --topo-loss-ratio=1.0 --version=gat_topo

Final result:
Val Accs: [0.7520386590154032, 0.7520386590154032, 0.7498573777643545, 0.7533474277660324, 0.7516359609382866, 0.7510990301687976, 0.7505285412262156, 0.7508976811302392, 0.7508976811302392, 0.7509983556495184]
Test Accs: [0.7418677859391396, 0.7384317840462523, 0.741662037322799, 0.7410036417505093, 0.7392959282348827, 0.7384317840462523, 0.739439952266321, 0.7407155936876324, 0.7396251260210275, 0.7388844310022015]
Average val accuracy: 0.7513 ± 0.0009
Average test accuracy: 0.7399 ± 0.0012
Number of params: 1441580
"""
