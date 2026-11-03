import heapq
import time
from collections import deque
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Search & CSP Visualizer", page_icon="🧭", layout="wide")

st.title("Search & Constraint Satisfaction Visualizer")
st.caption("An independent, step-by-step educational implementation inspired by the topics covered in Berkeley CS188.")

# ---------- Search problems ----------
GRAPH = {
    "A": {"B": 2, "C": 4},
    "B": {"A": 2, "D": 3, "E": 5},
    "C": {"A": 4, "E": 1, "F": 6},
    "D": {"B": 3, "G": 4},
    "E": {"B": 5, "C": 1, "G": 2, "H": 5},
    "F": {"C": 6, "H": 2},
    "G": {"D": 4, "E": 2, "I": 3},
    "H": {"E": 5, "F": 2, "I": 2},
    "I": {"G": 3, "H": 2},
}
POS = {
    "A": (0.0, 2.0), "B": (1.5, 3.2), "C": (1.5, 0.8),
    "D": (3.2, 4.0), "E": (3.2, 2.0), "F": (3.2, 0.2),
    "G": (5.1, 3.2), "H": (5.1, 1.0), "I": (6.8, 2.2),
}
HEURISTIC = {"A": 10, "B": 8, "C": 6, "D": 7, "E": 5, "F": 4, "G": 3, "H": 2, "I": 0}

def heuristic(goal):
    """Hand-made table for goal I; otherwise 80% of the true distance (admissible and consistent)."""
    if goal == "I":
        return HEURISTIC
    dist, pq = {goal: 0}, [(0, goal)]
    while pq:
        d, n = heapq.heappop(pq)
        if d > dist[n]:
            continue
        for m, c in GRAPH[n].items():
            if d + c < dist.get(m, float("inf")):
                dist[m] = d + c
                heapq.heappush(pq, (d + c, m))
    return {n: round(0.8 * dist[n], 1) for n in GRAPH}

def reconstruct(parent, node):
    out = [node]
    while node in parent:
        node = parent[node]
        out.append(node)
    return list(reversed(out))

def search_trace(start, goal, algorithm):
    """Return immutable-ish snapshots after each expansion."""
    h = heuristic(goal)
    frontier = []
    parent, g = {}, {start: 0}
    explored = set()
    counter = 0
    if algorithm == "BFS":
        frontier = deque([start])
    elif algorithm == "DFS":
        frontier = [start]
    else:
        priority = 0 if algorithm == "UCS" else h[start]
        heapq.heappush(frontier, (priority, counter, start))
    def frontier_view():
        if algorithm in ("BFS", "DFS"):
            return list(frontier)
        # Heap keeps stale entries (lazy deletion); show each live node once, best first.
        out = []
        for _, _, n in sorted(frontier):
            if n not in explored and n not in out:
                out.append(n)
        return out
    snapshots = [{
        "current": None, "frontier": [start], "explored": set(),
        "parent": {}, "g": {start: 0}, "status": "Initialized",
        "expanded": 0
    }]
    while frontier:
        if algorithm == "BFS":
            node = frontier.popleft()
        elif algorithm == "DFS":
            node = frontier.pop()
        else:
            _, _, node = heapq.heappop(frontier)
        if node in explored:
            continue
        explored.add(node)
        status = f"Expanded {node}"
        if node == goal:
            snapshots.append({
                "current": node, "frontier": frontier_view(),
                "explored": set(explored), "parent": dict(parent), "g": dict(g),
                "status": f"Goal found: {goal}", "expanded": len(explored),
                "solution": reconstruct(parent, goal)
            })
            break
        for nxt, cost in GRAPH[node].items():
            if nxt in explored:
                continue
            new_cost = g[node] + cost
            if algorithm in ("BFS", "DFS"):
                if nxt not in parent and nxt != start:
                    parent[nxt] = node
                    g[nxt] = new_cost
                    frontier.append(nxt)
            else:
                if new_cost < g.get(nxt, float("inf")):
                    g[nxt] = new_cost
                    parent[nxt] = node
                    counter += 1
                    heapq.heappush(frontier, (new_cost + (0 if algorithm == "UCS" else h[nxt]), counter, nxt))
        snapshots.append({
            "current": node, "frontier": frontier_view(), "explored": set(explored),
            "parent": dict(parent), "g": dict(g), "status": status,
            "expanded": len(explored)
        })
    if not any("solution" in s for s in snapshots):
        snapshots[-1]["status"] = "Frontier exhausted — no path found"
    return snapshots

def draw_graph(snap, start, goal):
    h = heuristic(goal)
    fig = go.Figure()
    # Edges
    for a, neighbors in GRAPH.items():
        for b, w in neighbors.items():
            if a < b:
                x0, y0 = POS[a]; x1, y1 = POS[b]
                fig.add_trace(go.Scatter(x=[x0,x1], y=[y0,y1], mode="lines",
                    line=dict(color="#9aa4b2", width=2), hoverinfo="text",
                    text=[f"{a} — {b}: cost {w}"]*2, showlegend=False))
                fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=str(w),
                    showarrow=False, font=dict(size=11, color="#64748b"),
                    bgcolor="rgba(255,255,255,0.85)")
    frontier = set(snap["frontier"])
    explored = set(snap["explored"])
    solution = set(snap.get("solution", []))
    xs, ys, labels, colors, sizes, hover = [], [], [], [], [], []
    for n, (x,y) in POS.items():
        xs.append(x); ys.append(y); labels.append(n)
        if n in solution: c = "#16a34a"
        elif n == snap.get("current"): c = "#f59e0b"
        elif n in explored: c = "#cbd5e1"
        elif n in frontier: c = "#60a5fa"
        else: c = "#f8fafc"
        colors.append(c); sizes.append(28 if n in (start,goal) else 23)
        hover.append(f"Node {n}<br>g(n): {snap['g'].get(n, '∞')}<br>h(n): {h[n]}")
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="markers+text",text=labels,
        textposition="middle center",hovertext=hover,hoverinfo="text",
        marker=dict(size=sizes,color=colors,line=dict(color="#334155",width=1.5)),
        showlegend=False))
    fig.add_annotation(x=POS[start][0],y=POS[start][1]+0.42,text="START",showarrow=False,
                       font=dict(size=10,color="#166534"))
    fig.add_annotation(x=POS[goal][0],y=POS[goal][1]+0.42,text="GOAL",showarrow=False,
                       font=dict(size=10,color="#166534"))
    fig.update_layout(height=470, margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(visible=False),yaxis=dict(visible=False,scaleanchor="x",scaleratio=1),
        plot_bgcolor="white", paper_bgcolor="white")
    return fig

# ---------- CSP ----------
AUSTRALIA = {
    "WA": {"NT","SA"}, "NT": {"WA","SA","Q"}, "SA": {"WA","NT","Q","NSW","V"},
    "Q": {"NT","SA","NSW"}, "NSW": {"Q","SA","V"}, "V": {"SA","NSW"}, "T": set()
}
# 3x3 grid with one diagonal per square (top-right to bottom-left); the first fixed order
# colors it greedily into a dead end, so plain backtracking really has to undo assignments.
GRID = {v: set() for v in "ABCDEFGHI"}
for r in range(3):
    for c in range(3):
        for rr, cc in ((r, c+1), (r+1, c), (r+1, c-1)):
            if rr < 3 and 0 <= cc < 3:
                a, b = "ABCDEFGHI"[3*r+c], "ABCDEFGHI"[3*rr+cc]
                GRID[a].add(b); GRID[b].add(a)
CSP_PROBLEMS = {
    "Australia map": {
        "neighbors": AUSTRALIA,
        "pos": {"WA":(0,1),"NT":(1,2),"SA":(1,1),"Q":(2,2),"NSW":(2,1),"V":(2,0),"T":(3,0.4)},
        "orders": {"Fixed: WA, NT, SA, Q, NSW, V, T": list(AUSTRALIA),
                   "Fixed (poor): WA, NT, NSW, V, SA, Q, T": ["WA", "NT", "NSW", "V", "SA", "Q", "T"]},
    },
    "Triangulated 3×3 grid": {
        "neighbors": GRID,
        "pos": {v: (i % 3, 2 - i // 3) for i, v in enumerate("ABCDEFGHI")},
        "orders": {"Fixed: row by row (A … I)": list("ABCDEFGHI"),
                   "Fixed: column by column (A, D, G, B …)": list("ADGBEHCFI")},
    },
}
COLORS = ["Red", "Green", "Blue"]
FILTERS = ["None", "Forward checking", "Arc consistency (AC-3)"]
MAX_CSP_STEPS = 15000  # ponytail: naive BFS on the grid needs ~14k steps / 29k tree nodes; raise for bigger graphs

@st.cache_resource(max_entries=16)  # traces are read-only; avoids re-solving (~1s for grid BFS) on every step
def csp_trace(problem="Australia map", algorithm="Backtracking", ordering="MRV", use_lcv=True,
              filtering="Forward checking"):
    """Return (snapshots, tree). Each tree node is {parent, var, val, depth, history}, where
    history is a list of (step, status) so the tree can be drawn as it looked at any step.
    Each snapshot's "queue" is (title, first items, total length) of the frontier / stack / arc queue."""
    trace, tree = [], []
    NB = CSP_PROBLEMS[problem]["neighbors"]
    order = None if ordering == "MRV" else CSP_PROBLEMS[problem]["orders"][ordering]
    def add_node(parent, var, val, status):
        depth = 0 if parent is None else tree[parent]["depth"] + 1
        tree.append({"parent": parent, "var": var, "val": val, "depth": depth, "history": [(len(trace), status)]})
        return len(tree) - 1
    def mark(node, status):
        tree[node]["history"].append((len(trace), status))
    def record(message, node, assignment, domains, failed=False, queue=None):
        if queue is None:  # backtracking: the recursion stack is the current partial assignment
            queue = ("Recursion stack (current path)", [f"{k} = {v}" for k, v in assignment.items()], len(assignment))
        trace.append({"assignment": dict(assignment), "domains": {k: list(v) for k, v in domains.items()},
                      "message": message, "failed": failed, "node": node, "n_nodes": len(tree), "queue": queue})

    if algorithm in ("BFS", "DFS"):
        # Naive search: states are partial assignments; constraints are only checked by the goal test.
        full = {v: list(COLORS) for v in NB}
        root = add_node(None, None, None, "frontier")
        frontier = deque([(root, {})])
        def fq():
            # Next item out first: the left end of a FIFO queue, the right end of a LIFO stack.
            head = [frontier[i] for i in (range(min(8, len(frontier))) if algorithm == "BFS"
                                          else range(len(frontier)-1, max(-1, len(frontier)-9), -1))]
            title = "Frontier: FIFO queue (next out first)" if algorithm == "BFS" else "Frontier: LIFO stack (top first)"
            return (title, [", ".join(f"{k}={v[0]}" for k, v in a.items()) or "{}" for _, a in head], len(frontier))
        record("Initial state: the empty assignment is on the frontier", root, {}, full, queue=fq())
        while frontier and len(trace) < MAX_CSP_STEPS:
            node, a = frontier.popleft() if algorithm == "BFS" else frontier.pop()
            if len(a) == len(NB):
                bad = [(x, y) for x in a for y in NB[x] if x < y and a[x] == a[y]]
                if not bad:
                    mark(node, "solution")
                    record("Goal test passed: complete and consistent assignment", node, a, full, queue=fq())
                    return trace, tree
                mark(node, "conflict")
                record(f"Goal test failed: {bad[0][0]} and {bad[0][1]} are both {a[bad[0][0]]}", node, a, full, True,
                       queue=fq())
                continue
            mark(node, "expanded")
            var = next(v for v in order if v not in a)
            children = [(add_node(node, var, val, "frontier"), {**a, var: val}) for val in COLORS]
            frontier.extend(children if algorithm == "BFS" else reversed(children))
            record(f"Expand: branch on {var} (no constraint checking yet)", node, a, full, queue=fq())
        record(f"Step limit ({MAX_CSP_STEPS}) reached: naive search grows exponentially" if frontier
               else "Frontier exhausted: no solution", node, a, full, True, queue=fq())
        return trace, tree

    assignment, domains = {}, {v: list(COLORS) for v in NB}
    def choose_var():
        unassigned = [v for v in (order or NB) if v not in assignment]
        if order is None:  # MRV, ties broken by degree
            return min(unassigned, key=lambda v: (len(domains[v]), -sum(n not in assignment for n in NB[v])))
        return unassigned[0]
    def prune(x, val, why, node, queue=None):
        """Remove val from x's domain; return False on wipeout."""
        domains[x].remove(val)
        record(f"{why}: remove {val} from {x}", node, assignment, domains, queue=queue)
        return bool(domains[x])
    def forward_check(var, val, node):
        for n in NB[var]:
            if n not in assignment and val in domains[n] and not prune(n, val, f"Forward check {n}→{var}", node):
                return n
    def ac3(node):
        queue = deque((x, y) for x in NB if x not in assignment for y in NB[x])
        while queue:
            x, y = queue.popleft()
            for v in list(domains[x]):
                if all(w == v for w in domains[y]):  # no value in y is compatible with x=v
                    queue.extend((z, x) for z in NB[x] if z != y and z not in assignment)
                    arcs = ("AC-3 arc queue (front first)", [f"{p}→{q}" for p, q in list(queue)[:8]], len(queue))
                    if not prune(x, v, f"Arc {x}→{y}", node, arcs):
                        return x
    def recurse(parent):
        if len(assignment) == len(NB):
            mark(parent, "solution")
            record("All variables assigned: solution found", parent, assignment, domains)
            return True
        if len(trace) >= MAX_CSP_STEPS:
            return False
        var = choose_var()
        values = list(domains[var])
        if use_lcv:
            values.sort(key=lambda val: sum(val in domains[n] for n in NB[var] if n not in assignment))
        record(f"Select variable {var}" + (f" (MRV: {len(values)} values left)" if order is None else ""),
               parent, assignment, domains)
        for val in values:
            if any(assignment.get(n) == val for n in NB[var]):
                child = add_node(parent, var, val, "conflict")
                record(f"Reject {var} = {val}: conflicts with an assigned neighbor", child, assignment, domains, True)
                continue
            saved = {k: list(v) for k, v in domains.items()}
            assignment[var], domains[var] = val, [val]
            child = add_node(parent, var, val, "expanded")
            record(f"Assign {var} = {val}", child, assignment, domains)
            wiped = {"Forward checking": lambda: forward_check(var, val, child),
                     "Arc consistency (AC-3)": lambda: ac3(child)}.get(filtering, lambda: None)()
            if wiped:
                mark(child, "wipeout")
                record(f"Domain of {wiped} is empty: dead end", child, assignment, domains, True)
            elif recurse(child):
                return True
            else:
                mark(child, "backtracked")
            assignment.pop(var)
            domains.clear(); domains.update(saved)
            record(f"Backtrack: undo {var} = {val}", child, assignment, domains, True)
        return False
    root = add_node(None, None, None, "expanded")
    record("Initial state: empty assignment", root, assignment, domains)
    recurse(root)
    return trace, tree

TREE_COLORS = {"frontier": "#bfdbfe", "expanded": "#f1f5f9", "conflict": "#ef4444",
               "wipeout": "#a855f7", "backtracked": "#94a3b8", "solution": "#16a34a"}

def tree_x(tree):
    """Leaves get consecutive x positions (from the final tree, so nodes don't move while stepping)."""
    children = {i: [] for i in range(len(tree))}
    for i, n in enumerate(tree[1:], 1):
        children[n["parent"]].append(i)
    x, leaf = {}, 0
    def place(i):
        nonlocal leaf
        for c in children[i]:
            place(c)
        if children[i]:
            x[i] = (x[children[i][0]] + x[children[i][-1]]) / 2
        else:
            x[i], leaf = leaf, leaf + 1
    place(0)
    return x

def draw_tree(tree, snap, step):
    xs = tree_x(tree)
    visible = range(snap["n_nodes"])
    status = {i: [s for t, s in tree[i]["history"] if t <= step][-1] for i in visible}
    path, n = set(), snap["node"]
    while n is not None:
        path.add(n); n = tree[n]["parent"]
    fig = go.Figure()
    for on_path, color, width in ((False, "#cbd5e1", 1), (True, "#f59e0b", 3)):
        ex, ey = [], []
        for i in visible:
            p = tree[i]["parent"]
            if p is not None and (i in path) == on_path:
                ex += [xs[p], xs[i], None]; ey += [-tree[p]["depth"], -tree[i]["depth"], None]
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=color, width=width),
                                 hoverinfo="skip", showlegend=False))
    small = len(visible) > 80
    label = lambda i: "{}" if tree[i]["var"] is None else f"{tree[i]['var']}={tree[i]['val'][0]}"
    fig.add_trace(go.Scatter(
        x=[xs[i] for i in visible], y=[-tree[i]["depth"] for i in visible],
        mode="markers" if small else "markers+text", text=[label(i) for i in visible],
        textposition="bottom center", textfont=dict(size=10),
        hovertext=[f"{label(i)}<br>{status[i]}" for i in visible], hoverinfo="text",
        marker=dict(size=7 if small else 16, color=[TREE_COLORS[status[i]] for i in visible],
                    line=dict(color=["#f59e0b" if i == snap["node"] else "#334155" for i in visible],
                              width=[4 if i == snap["node"] else 1 for i in visible])),
        cliponaxis=False, showlegend=False))
    # Fixed to the final tree's extent so the tree grows in place instead of rescaling;
    # padding scales with the width so edge nodes aren't cut off on wide (DFS/BFS) trees.
    lo, hi = min(xs.values()), max(xs.values())
    pad = max(1, 0.04 * (hi - lo))
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=10, b=25), plot_bgcolor="white",
                      xaxis=dict(visible=False, range=[lo - pad, hi + pad]),
                      yaxis=dict(visible=False, range=[-max(n["depth"] for n in tree) - 0.6, 0.4]))
    return fig

# ---------- UI ----------
def shift_step(key, delta):
    # Runs as a callback, i.e. before the slider with this key is re-created.
    st.session_state[key] += delta

tab_search, tab_csp, tab_about = st.tabs(["Graph search", "Constraint satisfaction", "About"])

with tab_search:
    st.subheader("Graph search")
    c1,c2,c3 = st.columns(3)
    with c1:
        algorithm = st.selectbox("Algorithm", ["BFS","DFS","UCS","A*"])
    with c2:
        start = st.selectbox("Start node", list(GRAPH), index=0)
    with c3:
        goal = st.selectbox("Goal node", list(GRAPH), index=8)
    if start == goal:
        st.info("Start and goal are identical. Select different nodes to see a search.")
    else:
        key = f"search_{algorithm}_{start}_{goal}"
        if st.button("Run / reset search", key="run_search"):
            st.session_state[key] = search_trace(start, goal, algorithm)
            st.session_state[f"{key}_step"] = 0
        if key not in st.session_state:
            st.session_state[key] = search_trace(start, goal, algorithm)
            st.session_state[f"{key}_step"] = 0
        trace = st.session_state[key]
        step_key = f"{key}_step"
        step = st.slider("Execution step", 0, len(trace)-1, key=step_key)
        snap = trace[step]
        left,right = st.columns([1.8,1])
        with left:
            st.plotly_chart(draw_graph(snap,start,goal), width="stretch")
            st.caption("Node colors: green = solution path · orange = currently expanded · blue = frontier · grey = explored")
        with right:
            st.markdown(f"### Step {step} / {len(trace)-1}")
            st.write(snap["status"])
            st.metric("Nodes expanded", snap["expanded"])
            st.write("**Frontier (next candidates)**")
            st.code(" → ".join(snap["frontier"]) if snap["frontier"] else "∅")
            st.write("**Explored**")
            st.code(", ".join(sorted(snap["explored"])) if snap["explored"] else "∅")
            st.write("**Path cost g(n)**")
            st.json({k: snap["g"][k] for k in sorted(snap["g"])})
            if "solution" in snap:
                path = snap["solution"]
                cost = sum(GRAPH[a][b] for a,b in zip(path,path[1:]))
                st.success(f"Path: {' → '.join(path)} | total cost: {cost}")
        b1,b2,b3 = st.columns([1,1,3])
        with b1:
            st.button("◀ Previous", key="search_prev", disabled=step == 0, on_click=shift_step, args=(step_key, -1))
        with b2:
            st.button("Next ▶", key="search_next", disabled=step >= len(trace)-1, on_click=shift_step, args=(step_key, 1))
        with b3:
            st.caption({
                "BFS":"FIFO queue; expands by depth. Finds a shallowest path when step costs are equal.",
                "DFS":"LIFO stack; follows one branch deeply before backtracking. Not generally optimal.",
                "UCS":"Priority queue ordered by accumulated path cost g(n). Optimal for nonnegative costs.",
                "A*":"Priority queue ordered by f(n)=g(n)+h(n). Optimal with an admissible heuristic."
            }[algorithm])

with tab_csp:
    st.subheader("Graph-coloring CSP")
    p_,a,b,c,d = st.columns(5)
    with p_:
        problem = st.selectbox("Graph", list(CSP_PROBLEMS))
    prob = CSP_PROBLEMS[problem]
    with a:
        csp_algo = st.selectbox("Search algorithm", ["Backtracking","DFS","BFS"])
    naive = csp_algo != "Backtracking"
    with b:
        ordering = st.selectbox("Variable ordering", ([] if naive else ["MRV"]) + list(prob["orders"]))
    with c:
        filtering = st.selectbox("Filtering", FILTERS, index=1, disabled=naive)
    with d:
        lcv = st.checkbox("Least-constraining value (LCV)", value=True, disabled=naive)
    if naive:
        st.caption("Naive search over partial assignments: constraints are checked only by the goal test on complete "
                   "assignments, so filtering and LCV don't apply.")
        filtering, lcv = "None", False
    trace, tree = csp_trace(problem, csp_algo, ordering, lcv, filtering)
    csp_key = f"csp_{problem}_{csp_algo}_{ordering}_{lcv}_{filtering}"
    # Autoplay: a tick (rerun requested at the end of the script) advances one step
    # before the slider is created, which is the only point Streamlit allows it.
    if st.session_state.pop("csp_tick", False) and st.session_state.get(csp_key, 0) < len(trace)-1:
        st.session_state[csp_key] = st.session_state.get(csp_key, 0) + 1
    step = st.slider("CSP execution step", 0, len(trace)-1, key=csp_key)
    snap = trace[step]
    if step >= len(trace)-1:
        st.session_state.csp_playing = False
    playing = st.session_state.get("csp_playing", False)
    c1,c2,c3,c4,c5=st.columns([1,1,1,2,4])
    with c1:
        st.button("◀ Previous",key="csp_prev",disabled=step==0,on_click=shift_step,args=(csp_key,-1))
    with c2:
        st.button("Next ▶",key="csp_next",disabled=step>=len(trace)-1,on_click=shift_step,args=(csp_key,1))
    with c3:
        st.button("⏸ Pause" if playing else "⏵ Play", key="csp_play", disabled=step>=len(trace)-1,
                  on_click=lambda: st.session_state.update(csp_playing=not playing))
    with c4:
        speed = st.slider("Speed (steps / second)", 1, 30, 4, key="csp_speed", label_visibility="collapsed",
                          help="Autoplay speed in steps per second (large trees render more slowly)")
    with c5:
        # Conflicts are flagged inline (not with st.warning) so the page doesn't jump when stepping.
        msg = f"Step {step} / {len(trace)-1}: {snap['message']}"
        st.markdown(f":red[**⚠ {msg}**]" if snap["failed"] else f"**{msg}**")
    # Render map as an adjacency-based node layout.
    map_pos = prob["pos"]
    fig = go.Figure()
    for region, neighbors in prob["neighbors"].items():
        for neighbor in neighbors:
            if region < neighbor:
                x0,y0=map_pos[region]; x1,y1=map_pos[neighbor]
                fig.add_trace(go.Scatter(x=[x0,x1],y=[y0,y1],mode="lines",
                    line=dict(color="#94a3b8",width=3),hoverinfo="skip",showlegend=False))
    palette={"Red":"#ef4444","Green":"#22c55e","Blue":"#3b82f6"}
    for region,(x,y) in map_pos.items():
        val=snap["assignment"].get(region)
        fill=palette[val] if val else "#e2e8f0"
        fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",text=[region],
            textposition="middle center",marker=dict(size=55,color=fill,line=dict(color="#334155",width=1.5)),
            textfont=dict(color="white" if val else "#0f172a",size=12),showlegend=False,
            hovertext=f"{region}: {val or 'unassigned'}",hoverinfo="text"))
    fig.update_layout(height=300,margin=dict(l=5,r=5,t=5,b=5),xaxis=dict(visible=False),
        yaxis=dict(visible=False,scaleanchor="x",scaleratio=1),plot_bgcolor="white")
    # Map + domains on the left, search tree and frontier/queue on the right, all visible while stepping.
    l,r,q=st.columns([1,1.5,0.6])
    with l:
        st.plotly_chart(fig,width="stretch")
        rows = "\n".join(f"| **{region}** | {snap['assignment'].get(region) or ', '.join(snap['domains'][region]) or '∅'} |"
                         for region in prob["neighbors"])
        st.markdown("| Region | Value / domain |\n|---|---|\n" + rows)
        m1,m2 = st.columns(2)
        m1.metric("Assigned variables",len(snap["assignment"]))
        m2.metric("Search-tree nodes", snap["n_nodes"])
    with r:
        st.markdown("**Search tree**")
        st.plotly_chart(draw_tree(tree, snap, step), width="stretch")
        st.caption("Orange path = current partial assignment · light blue = on frontier · white = expanded · "
                   "red = constraint violated · purple = filtering wiped out a domain · grey = backtracked · "
                   "green = solution. Labels (WA=R …) show the assignment made on that branch; zoom in for large trees.")
    with q:
        title, items, total = snap["queue"]
        st.markdown(f"**{title}**")
        st.code("\n".join(items) + (f"\n… +{total - len(items)} more" if total > len(items) else "") if items else "∅")
        st.caption(f"{total} item{'s' if total != 1 else ''}")

with tab_about:
    st.subheader("What this prototype demonstrates")
    st.markdown("""
- **BFS / DFS:** frontier discipline and expansion order.
- **UCS:** path-cost-based priority and optimal path selection.
- **A\*:** combined path cost and heuristic estimate.
- **CSP search:** naive BFS / DFS over partial assignments vs. backtracking, with variable ordering (MRV),
  value ordering (LCV), and filtering (forward checking or arc consistency / AC-3), drawn as a search tree.

This is a clean-room educational implementation, not a copy of Berkeley's source code or exact UI.
    """)
    st.info("Current scope: a weighted graph search sandbox and graph-coloring CSPs (Australia map, triangulated 3×3 grid). Minimax, alpha-beta pruning, maze editing, and reinforcement learning are not included in this first version.")

# Autoplay tick for the CSP tab. Done last so every tab is fully rendered before the rerun.
if st.session_state.get("csp_playing"):
    time.sleep(1 / speed)
    st.session_state.csp_tick = True
    st.rerun()
