# GAT+label+reuse+topo loss OGB submission

This repository contains the code to reproduce the performance of "GAT+label+reuse+topo" on ogbn-arxiv dataset.Most code of this repo is modified based on [DGL example](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv) and [Espylapiza's GAT implement](https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv). 

Many tricks that we use include 'reuse' please refer to [Espylapiza's GAT implement](https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv). Most hyperparameters follows the setting of previous works, hyperparameters about 'topo loss' were hand tuned. All experiments were runned with a Tesla V100 with 16GB memory.

## topo loss

When training GAT model, we observed that node prediction has a large variance among training epochs. We proposed topo loss in order to stabilize graph model training. Givin a Graph G=(V,E) with n nodes V=[v1,v2,...,vn] and m edges E=[e1,e2,...,em]. ei=<vis,vid>$ where vis is the source node and vid is the destination node. Topo loss can be expressed as 

<img src="http://chart.googleapis.com/chart?cht=tx&chl= \mathcal{L}_{topo}=\frac{1}{|E'|}\sum_{i\in E'}\cos(p_{v_i^s},p_{v_i^d})" style="border:none;">

where pv is GAT prediction vector of node v and E' is a sampled subset of E.

## usage

```shell
pip install -r requirements.txt
python -u gat.py --gpu=0 --n-label-iters=1 --dropout=0.75 --input-drop=0.25 --edge-drop=0.3 --topo-mask-threshold=0.2 --topo-loss-ratio=1.0 --version=gat_topo
```

Training log and performance will be written into file "./log/[version]_[time_stamp]/log"

## result

To validate the reproduction of this repo, we first randomly use seed 42 at first,  and the result:

```
Val Accs:, [0.7514346118997282, 0.7514010537266351, 0.7509983556495184, 0.7520722171884963, 0.7505620993993087, 0.7514681700728212, 0.7522064498808685, 0.7523406825732407, 0.750729890264774, 0.7510654719957045]
Test Accs: [0.7384317840462523, 0.7397485751908318, 0.7407155936876324, 0.7413945641215562, 0.7408184679958028, 0.7386169578009588, 0.7408390428574367, 0.7423615826183569, 0.7400571981153427, 0.7398308746373681]
Average val accuracy: 0.7514 ± 0.0006
Average test accuracy: 0.7403 ± 0.0011
Number of params: 1441580
```

Then we follow the instruction of OGB rules and set the random seed from 0~9 and get the following result:

```
Val Accs: [0.7520386590154032, 0.7520386590154032, 0.7498573777643545, 0.7533474277660324, 0.7516359609382866, 0.7510990301687976, 0.7505285412262156, 0.7508976811302392, 0.7508976811302392, 0.7509983556495184]
Test Accs: [0.7418677859391396, 0.7384317840462523, 0.741662037322799, 0.7410036417505093, 0.7392959282348827, 0.7384317840462523, 0.739439952266321, 0.7407155936876324, 0.7396251260210275, 0.7388844310022015]
Average val accuracy: 0.7513 ± 0.0009
Average test accuracy: 0.7399 ± 0.0012
Number of params: 1441580
```