#!/usr/bin/env python3
"""Independent checks for the zero-forcing referee report.

Run with Python 3.12+ and NetworkX 3.6.1:
    python zero_forcing_extended_audits.py --package /path/to/unzipped/submission

The closure simulator, subset enumeration, marker tests, clique transfer, and
polynomial determinant below are written independently of the submission.
The submission's general block transfer is used only as a comparison target.
These finite checks supplement the proofs in the report; they are not proofs.
"""
from __future__ import annotations
import argparse
import itertools as it
import json
import random
import sys
import time
from pathlib import Path
import networkx as nx


def trim(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p or [0]


def add(*ps):
    out = [0] * max(map(len, ps), default=1)
    for p in ps:
        for i, a in enumerate(p):
            out[i] += a
    return trim(out)


def neg(p):
    return [-a for a in p]


def mul(p, q):
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i+j] += a*b
    return trim(out)


def shift(p):
    return trim([0] + list(p))


def determinant(matrix):
    n = len(matrix)
    out = [0]
    for perm in it.permutations(range(n)):
        parity = sum(perm[i] > perm[j] for i in range(n) for j in range(i+1,n))
        term = [(-1)**parity]
        for i, j in enumerate(perm):
            term = mul(term, matrix[i][j])
        out = add(out, term)
    return out


def adjacency(graph):
    assert set(graph) == set(range(len(graph)))
    return tuple(sum(1 << v for v in graph[u]) for u in range(len(graph)))


def closure(adj, initial, forbidden=0):
    """Synchronous bit-mask closure, independently implemented."""
    blue = initial
    while True:
        next_blue = blue
        allowed = blue & ~forbidden
        while allowed:
            bit = allowed & -allowed
            allowed -= bit
            u = bit.bit_length()-1
            white_neighbors = adj[u] & ~blue
            if white_neighbors and not white_neighbors & (white_neighbors-1):
                next_blue |= white_neighbors
        if next_blue == blue:
            return blue
        blue = next_blue


def response(adj, initial):
    """Root is vertex 0 and is absent from the initial set."""
    assert initial & 1 == 0
    all_blue = (1 << len(adj))-1
    plus = closure(adj, initial | 1, 1)
    if plus == all_blue:
        category = 'P'
    elif closure(adj, initial | 1) == all_blue:
        category = 'A'
    else:
        return 'F'
    return category + str(int(bool(closure(adj, initial, 1) & 1)))


def profile(graph):
    adj = adjacency(graph)
    out = {s: [0]*len(graph) for s in ('P0','P1','A0','A1')}
    for compact in range(1 << (len(graph)-1)):
        initial = compact << 1
        state = response(adj, initial)
        if state != 'F':
            out[state][compact.bit_count()] += 1
    return {s: trim(p) for s,p in out.items()}


def full_polynomial(graph):
    adj = adjacency(graph)
    out = [0]*(len(graph)+1)
    for initial in range(1 << len(graph)):
        if closure(adj, initial) == (1 << len(graph))-1:
            out[initial.bit_count()] += 1
    return trim(out)


def root_at(graph, root):
    others = [v for v in graph if v != root]
    return nx.relabel_nodes(graph, {v:i for i,v in enumerate([root]+others)})


def marked(base, letters):
    graph = base.copy()
    initial = 0
    for i,(blue,demand) in enumerate(letters, 1):
        initial |= blue << i
        if demand:
            graph.add_edge(i,len(graph))
    return graph, initial


def clique_category(letters):
    white = sum(not b for b,d in letters)
    able_blue = any(b and not d for b,d in letters)
    if white >= 2:
        return 'F'
    if white == 0:
        return 'P1' if able_blue else 'P0'
    return 'P0' if able_blue else 'A0'


def letter_weights(p):
    return {
        (0,0):p['P0'],
        (1,0):add(shift(p['P0']),p['P1'],shift(p['P1'])),
        (0,1):p['A0'],
        (1,1):add(shift(p['A0']),p['A1'],shift(p['A1'])),
    }


def clique_transfer(profiles):
    # B: all effectively blue; D: all effectively blue with demands;
    # W: exactly one white; E: exactly one white, all blue vertices demanded.
    B,D,W,E = [1],[1],[0],[0]
    for p in profiles:
        weights = letter_weights(p)
        blue = add(weights[1,0], weights[1,1])
        demand_blue = weights[1,1]
        white = add(weights[0,0], weights[0,1])
        B,D,W,E = (
            mul(B,blue), mul(D,demand_blue),
            add(mul(W,blue),mul(B,white)),
            add(mul(E,demand_blue),mul(D,white)),
        )
    return {'P0':add(D,W,neg(E)), 'P1':add(B,neg(D)), 'A0':E, 'A1':[0]}


def effective_block_transfer(base, profiles):
    out = {s:[0] for s in ('P0','P1','A0','A1')}
    weights = [letter_weights(p) for p in profiles]
    letters = [(0,0),(1,0),(0,1),(1,1)]
    for word in it.product(letters,repeat=len(base)-1):
        term = [1]
        for w,t in zip(weights,word):
            term = mul(term,w[t])
        if term == [0]:
            continue
        graph,initial = marked(base,word)
        state = response(adjacency(graph),initial)
        if state != 'F':
            out[state] = add(out[state],term)
    return out


def check_replacement():
    atlas = nx.graph_atlas_g()
    branches = [root_at(g,r) for g in atlas if 1 <= len(g) <= 4 and nx.is_connected(g) for r in g]
    contexts = [root_at(g,r) for g in atlas if 1 <= len(g) <= 3 for r in g]
    count = 0
    for h in branches:
        ha = adjacency(h)
        for mask in range(1 << (len(h)-1)):
            state = response(ha,mask << 1)
            if state == 'F':
                continue
            for j in contexts:
                n = len(j)
                remap = {0:0, **{v:n+v-1 for v in h if v}}
                full = nx.compose(j,nx.relabel_nodes(h,remap))
                fa = adjacency(full)
                marker = j.copy()
                if state.startswith('A'):
                    marker.add_edge(0,n)
                ma = adjacency(marker)
                internal = mask << n
                external_mask = (1 << n)-1
                for initial in range(1 << n):
                    supplied = initial | int(state.endswith('1'))
                    for qmask in range(1 << (n-1)):
                        forbidden = qmask << 1
                        a = closure(fa,initial | internal,forbidden)
                        b = closure(ma,supplied,forbidden)
                        assert a & external_mask == b & external_mask
                        assert (a == (1 << len(fa))-1) == (b == (1 << len(ma))-1)
                        count += 1
    return count


def check_clique_words():
    total = 0
    letters = [(0,0),(1,0),(0,1),(1,1)]
    for n in range(2,10):
        base = nx.complete_graph(n)
        for word in it.product(letters,repeat=n-1):
            g,initial = marked(base,word)
            assert response(adjacency(g),initial) == clique_category(word), (n,word)
            total += 1
    return total


def check_decorated_blocks(package):
    sys.path.insert(0,str(package.resolve()))
    import zero_forcing_block_dp as submitted
    rng = random.Random(20260906)
    small = [root_at(g,r) for g in nx.graph_atlas_g()
             if 1 <= len(g) <= 4 and nx.is_connected(g) for r in g]
    arbitrary = [g for g in nx.graph_atlas_g() if 2 <= len(g) <= 5 and nx.is_biconnected(g)]
    for k in range(160):
        base = nx.complete_graph(rng.randint(2,5)) if k < 80 else rng.choice(arbitrary).copy()
        branches = [rng.choice(small) for _ in range(len(base)-1)]
        while len(base)+sum(len(h)-1 for h in branches) > 13:
            branches[rng.randrange(len(branches))] = nx.empty_graph(1)
        actual = base.copy()
        for v,h in enumerate(branches,1):
            start = len(actual)
            remap = {0:v, **{u:start+u-1 for u in h if u}}
            actual = nx.compose(actual,nx.relabel_nodes(h,remap))
        ps = [profile(h) for h in branches]
        expected = profile(actual)
        assert effective_block_transfer(base,ps) == expected
        old = submitted.block_transfer(base,0,{v:submitted.Profile.make(**p) for v,p in enumerate(ps,1)})
        assert {s:list(getattr(old,s)) for s in expected} == expected
        if k < 80:
            assert clique_transfer(ps) == expected
    return {'decorated_cliques':80,'decorated_arbitrary_blocks':80,'maximum_total_order':13}


def check_rank():
    graphs = [nx.empty_graph(1),nx.path_graph(2),nx.path_graph(3),root_at(nx.path_graph(3),1)]
    entries = []
    for g in graphs:
        row = []
        for h in graphs:
            remap = {0:0, **{v:len(g)+v-1 for v in h if v}}
            row.append(full_polynomial(nx.compose(g,nx.relabel_nodes(h,remap))))
        entries.append(row)
    observed = determinant(entries)
    expected = [0]*6+[4,8,4]
    assert observed == expected, observed
    return {'connection_minor':entries,'determinant_ascending_coefficients':observed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,default=Path('.'))
    args = parser.parse_args()
    start = time.monotonic()
    results = {'python':sys.version.split()[0], 'networkx':nx.__version__}
    results['restricted_replacement_configurations'] = check_replacement()
    print('Restricted replacement:',results['restricted_replacement_configurations'],flush=True)
    results['marked_clique_words_orders_2_through_9'] = check_clique_words()
    print('Marked clique words:',results['marked_clique_words_orders_2_through_9'],flush=True)
    results['decorated_blocks'] = check_decorated_blocks(args.package)
    print('Decorated blocks:',results['decorated_blocks'],flush=True)
    results['rank_four_check'] = check_rank()
    results['elapsed_seconds'] = round(time.monotonic()-start,3)
    results['status'] = 'ALL INDEPENDENT CHECKS PASSED'
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    main()
