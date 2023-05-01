from collections import defaultdict

def remove_irrelevant_edges(adj, s, t):
    if t in adj[s]:
        adj_trim = defaultdict(list)
        adj_trim[s].append(t)
        adj_trim[t].append(s)
        return adj_trim
    
    def dfs(v, seen):
        if v in seen:
            return
        seen.add(v)
        for w in adj[v]:
            dfs(w, seen)
    
    seen_s = {s, t}
    for w in adj[s]:
        dfs(w, seen_s)
    seen_t = {s, t}
    for w in adj[t]:
        dfs(w, seen_t)

    relevant = seen_s.intersection(seen_t)

    adj_trim = defaultdict(list)

    for v in relevant:
        for w in adj[v]:
            if w in relevant:
                adj_trim[v].append(w)
    
    return adj_trim


'''

test: x1 -- s -- x2 -- x3 -- x4 -- t -- x5 -- x6
       \                            \           \
        x7 --x8 -- x9 -- x10         x11 - x12 - x13
'''
adj = defaultdict(list)
adj['s'] = ['x1', 'x2']
adj['x1'] = ['s', 'x7']
adj['x2'] = ['s', 'x3']
adj['x3'] = ['x2', 'x4']
adj['x4'] = ['x3', 't']
adj['t'] = ['x4', 'x5', 'x11']
adj['x5'] = ['t', 'x6']
adj['x6'] = ['x5', 'x13']
adj['x7'] = ['x1', 'x8']
adj['x8'] = ['x7', 'x9']
adj['x9'] = ['x8', 'x10']   
adj['x10'] = ['x9']
adj['x11'] = ['t', 'x12']
adj['x12'] = ['x11', 'x13']
adj['x13'] = ['x12', 'x6']

adj_trim = remove_irrelevant_edges(adj, 's', 't')
for v in adj_trim:
    print(v, adj_trim[v])

