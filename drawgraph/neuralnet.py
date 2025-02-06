from graphviz import Digraph


def trace(root):
    nodes = set()
    edges = set()

    def dfs(u):
        if u not in nodes:
            nodes.add(u)
            for v in u._prev:
                edges.add((v, u))  # child -> parent directed edges
                if v not in nodes:
                    dfs(v)

    dfs(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="dot", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root)

    for node in nodes:
        uid = str(id(node))
        label_text = f"{node.label} | data: {node.data:.4f} | grad: {node.grad:.4f}"
        dot.node(
            uid,
            label=label_text,
            shape="record",
            style="filled",
            fillcolor="lightyellow",
        )

        if node._op:
            dot.node(
                name=uid + node._op,
                label=f"{node._op}",
                style="filled",
                fillcolor="lightblue",
            )
            dot.edge(uid + node._op, uid)

    for u, v in edges:
        dot.edge(str(id(u)), str(id(v)) + v._op)

    return dot
