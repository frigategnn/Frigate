import pickle
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
import multiprocessing as mp
from scipy.sparse import coo_array


def sssplr(G, nodes):
    result = {}
    for n in nodes:
        result[n] = nx.single_source_dijkstra_path_length(G, n, cutoff=None, weight='haversine')
    return result


def lipschitz_node_embeddings(G, nodes, k):
    G_temp = G.reverse(copy=True)
    anchor_nodes = np.random.choice(nodes, size=k, replace=False)
    num_workers = 16 if k > 16 else k
    results = []
    per_worker = k/num_workers
    pool = mp.Pool(processes=num_workers)
    for n in range(num_workers):
        start, end = int(per_worker*n), int(per_worker*(n+1))
        results.append(
                pool.apply_async(
                    sssplr, args=[
                        G_temp, anchor_nodes[start:end]]))
    lips_dist_list = [result.get() for result in results]
    pool.close()
    pool.join()
    lips_dist = {}
    for d in lips_dist_list:
        lips_dist.update(d)
    embeddings = np.zeros((len(nodes), k))
    for i, node_i in enumerate(anchor_nodes):
        sd = lips_dist[node_i]
        for j, node_j in enumerate(nodes):
            dist = sd.get(node_j, -1)
            if dist!=-1:
                embeddings[node_j, i] = 1/(dist+1)
    embeddings = (embeddings - embeddings.mean(axis=0))/embeddings.std(axis=0)
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str,
                        help="Path to the directory having adj_mx.pkl file, eg: Chengdu")
    parser.add_argument('-k', type=int, default=16,
                        help="Number of Lipschitz anchor nodes")
    parser.add_argument('--output_filename', required=True, type=str,
                        help="Output file name")
    pargs = parser.parse_args()
    # ---------------------------------------------------
    adj_path = Path(pargs.dataset,"adj_mx.pkl")
    with open(adj_path, "rb") as f:
        adj_data = pickle.load(f)
    adj = coo_array((adj_data['v'],adj_data['ij']),shape=adj_data['shape']).todense()
    G = nx.DiGraph()
    G.add_nodes_from(range(adj_data['shape'][0]))
    for i,j in zip(adj_data['ij'][0],adj_data['ij'][1]):
        G.add_edge(i, j, haversine=adj[i,j])
    nodes = list(range(adj_data['shape'][0]))
    embeddings = lipschitz_node_embeddings(G, nodes, pargs.k)
    np.savez_compressed(pargs.output_filename+'.npz', lipschitz=embeddings)


if __name__=="__main__":
    main()
