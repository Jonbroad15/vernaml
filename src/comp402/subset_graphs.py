import networkx as nx

def slice_graph(g, subset):
    """
    In:
        - graph
        - list of subset of nodes
    Out:
        - subgraph
        - complement of subgraph
    """
    g_prime = g.copy()

    if len(subset) > 0:
        g_subgraph = g_prime.subgraph(subset).copy()
        g_prime.remove_nodes_from([n for n in g_prime if n in set(subset)])
    else:
        g_subgraph = None

    return g_subgraph, g_prime

