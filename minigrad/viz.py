from graphviz import Digraph
from minigrad.tensor import Tensor

def trace(root):
    """
    Recursively builds a set of all nodes and edges in the computation graph.
    """
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_graph(root, format='svg', filename='graph'):
    """
    Visualizes the computation graph starting from a root Tensor.
    Requires the 'graphviz' Python package AND the Graphviz system software.
    """
    dot = Digraph(format=format, graph_attr={'rankdir': 'LR'}) # Left to Right layout
    
    nodes, edges = trace(root)
    
    for n in nodes:
        uid = str(id(n))
        
        # Build the label for the Tensor
        label = n.label if n.label else ("Tensor" if n._op else "Input/Param")
        shape_str = f"shape={n.shape}"
        
        node_label = f"{{ {label} | {shape_str} }}"
        dot.node(name=uid, label=node_label, shape='record', style='filled', fillcolor='lightblue' if n.requires_grad else 'lightgrey')
        
        # If this Tensor was created by an operation, draw a separate Operation node
        if n._op:
            op_uid = uid + "_" + n._op
            dot.node(name=op_uid, label=n._op, shape='oval', style='filled', fillcolor='orange')
            # Connect the Operation node to the resulting Tensor
            dot.edge(op_uid, uid)
            
    # Connect parent Tensors to the Operation nodes that consumed them
    for n1, n2 in edges:
        n1_uid = str(id(n1))
        op_uid = str(id(n2)) + "_" + n2._op
        dot.edge(n1_uid, op_uid)
        
    try:
        dot.render(filename, cleanup=True)
        print(f"Graph successfully rendered to {filename}.{format}")
    except Exception as e:
        print(f"Notice: Local Graphviz software not found. Falling back to web API for rendering...")
        dot.save(filename + '.dot')
        
        # Fallback to QuickChart API to render the DOT source into an image
        import urllib.parse
        import urllib.request
        import ssl
        try:
            dot_source = dot.source
            # Request PNG format from QuickChart explicitly
            url = 'https://quickchart.io/graphviz?format=png&graph=' + urllib.parse.quote(dot_source)
            
            # Bypass SSL verification which fails on some MacOS Python installs
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            with urllib.request.urlopen(url, context=ctx) as response, open(filename + '.png', 'wb') as out_file:
                out_file.write(response.read())
                
            print(f"Graph successfully rendered and downloaded to {filename}.png (via QuickChart API)")
        except Exception as api_e:
            print(f"Web API fallback also failed: {api_e}")
            print(f"Saved raw graph source to {filename}.dot instead.")
        
    return dot
