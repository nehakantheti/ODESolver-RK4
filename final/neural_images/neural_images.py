import graphviz

def draw_network(name, layer_sizes, layer_names):
    dot = graphviz.Digraph(comment=name, format='png')
    dot.attr(rankdir='LR', splines='line', nodesep='0.1', ranksep='1.5')
    dot.attr('node', shape='circle', style='solid', fixedsize='true', width='0.3', label='')
    
    # Create Nodes
    for i, (size, name) in enumerate(zip(layer_sizes, layer_names)):
        with dot.subgraph(name=f'cluster_{i}') as c:
            c.attr(color='white', label=name, fontname='Arial') # Invisible cluster for labels
            for j in range(size):
                c.node(f'L{i}_N{j}')

    # Create fully connected Edges
    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i+1]):
                dot.edge(f'L{i}_N{j}', f'L{i+1}_N{k}', color='gray70', penwidth='0.5')
                
    dot.render(f'{name}_architecture', view=False)
    print(f"Generated {name}_architecture.png")

# Generate Coarse Propagator
draw_network(
    "Coarse_Propagator", 
    [5, 8, 8, 8, 3], 
    ["Input\n(D+1+P)", "Hidden 1\n(128 Units)", "Hidden 2\n(128 Units)", "Hidden 3\n(128 Units)", "Output\nf-hat"]
)

# Generate K-Factor (Simplified without skip connections for standard graphviz)
draw_network(
    "K_Factor_ResNet", 
    [6, 7, 7, 7, 9], 
    ["Input\nk1, y, t, p", "Projection\n(96 Units)", "ResBlock 1\n(96 Units)", "ResBlock 2\n(96 Units)", "Output\n(3D Deltas)"]
)