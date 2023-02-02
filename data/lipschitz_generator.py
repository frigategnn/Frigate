import os
import sys
import random
import pickle
import argparse
import numpy as np
import multiprocessing as mp

import networkx as nx
np.set_printoptions(formatter={'float':'{: >+5.2f}'.format})
import logging

logger = logging.getLogger('lipschitz')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
fm = logging.Formatter("%(process)s - %(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(fm)
ch.flush = sys.stdout.flush
logger.addHandler(ch)

def sssplr(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_dijkstra_path_length(graph, node, cutoff=cutoff, weight='haversine')
    return dists_dict
def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result
def lipschitz_node_embeddings(nodes_forward, G, k):
    nodes = list(nodes_forward.keys())
    G_temp = G.reverse(copy=True)
    anchor_nodes = random.sample(nodes, k)
    logger.info("Starting dijkstra")
    num_workers = 32
    cutoff = None
    pool = mp.Pool(processes = num_workers)
    results = [pool.apply_async(sssplr,\
            args=(G_temp, anchor_nodes[int(k/num_workers*i):int(k/num_workers*(i+1))], cutoff))\
            for i in range(num_workers)]
    outputs = [p.get() for p in results]
    dists_dict = merge_dicts(outputs)
    pool.close()
    pool.join()
    logger.info('Dijkstra done')
    embeddings = np.zeros((len(nodes),k))
    for i, node_i in enumerate(anchor_nodes):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(nodes):
            dist = shortest_dist.get(node_j, -1)
            if dist != -1:
                embeddings[nodes_forward[node_j], i] = 1/(dist+1)
    embeddings = (embeddings - embeddings.mean(axis=0))/embeddings.std(axis=0)
    return embeddings, nodes_forward

parser = argparse.ArgumentParser()
parser.add_argument("-d","--data-home",default="chengdu_data")
parser.add_argument("-g","--graph",default="map/graph_with_haversine.pkl")
parser.add_argument("-o","--output-dir",default="outputs")
args = parser.parse_args()
logger.debug(f"OPTIONS: {vars(args)}")
# load graph
with open(os.path.join(os.path.dirname(os.path.abspath('.')),args.data_home,args.graph),'rb') as f:
    g = pickle.load(f)

nodes_used = set()
for i, e in enumerate(g.edges(data=True)):
    nodes_used.add(e[0])
    nodes_used.add(e[1])

nodes_used = list(nodes_used)
logger.debug(f"Number of nodes {len(nodes_used)}")

nodes_forward = {node:i for i,node in enumerate(nodes_used)}
k = 16
logger.info(f"k={k} anchor nodes used for lipschitz embeddings.")
em,nf=lipschitz_node_embeddings(nodes_forward, g, k)
logger.info("Embeddings are\n{}\n{}".format(em,em.shape))
with open(os.path.join(os.path.dirname(os.path.abspath('.')),args.output_dir,"{df}_lipschitz_data.npz".format(df=args.data_home)),"wb") as f:
    save_dict = {'embeddings':em,'node_mappings_used':nf}
    np.savez_compressed(f,**save_dict)

logger.info("Saved output at: {}".format(os.path.join(os.path.dirname(os.path.abspath('.')),args.output_dir,"{df}_lipschitz_data.npz".format(df=args.data_home))))
logger.info("Look for keys \"embeddings\" and \"node_mappings_used\"")
