import math

import jax
import jax.numpy as jnp

from .block_management import Block
from .pgm import SpinNode


def make_graph(
    side_len: int,
    torus: bool,
) -> tuple:
    jumps = [(1, 0), (2, 1), (3, 2), (1, 4)]
    side_len = math.ceil(side_len / 2) * 2
    size = side_len**2

    def get_idx(i, j):
        if torus:
            i = (i + 10 * side_len) % side_len
            j = (j + 10 * side_len) % side_len

        cond = (i >= side_len) | (j >= side_len) | (i < 0) | (j < 0)
        return jnp.where(cond, -1, i * side_len + j)

    def get_coords(idx):
        return idx // side_len, (idx + side_len) % side_len

    @jax.jit
    def make_edge_single(idx, di, dj):
        i, j = get_coords(idx)
        return jnp.array([idx, get_idx(i + di, j + dj)])

    make_edge_arr = jax.jit(jax.vmap(make_edge_single, in_axes=(0, None, None), out_axes=0))

    indices = jnp.arange(size)
    edge_arrs_list = []

    for dx, dy in jumps:
        edges_pos = make_edge_arr(indices, dx, dy)
        edges_neg = make_edge_arr(indices, -dx, -dy)
        edge_arrs_list.append(edges_pos)
        edge_arrs_list.append(edges_neg)

    edge_array = jnp.concatenate(edge_arrs_list, axis=0)

    nodes_upper = []
    nodes_lower = []
    all_nodes = []
    for i in range(size):
        new_node = SpinNode()
        all_nodes.append(new_node)
        if (i // side_len + i % side_len) % 2 == 0:
            nodes_upper.append(new_node)
        else:
            nodes_lower.append(new_node)

    edges = set()
    edge_array = edge_array.tolist()
    for i, j in edge_array:
        if i == -1 or j == -1:
            continue
        edges.add((all_nodes[i], all_nodes[j]))

    edges = list(edges)

    upper_block = Block(nodes_upper)
    lower_block = Block(nodes_lower)

    return all_nodes, edges, upper_block, lower_block
