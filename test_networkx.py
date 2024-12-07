import inspect
import networkx as nx
import matplotlib.pyplot as plt
import sklearn  # Change this to the library you want to explore

# Step 1: Inspect the library
def explore_library(lib):
    # Create a graph
    G = nx.Graph()

    # Add the main library as a node
    G.add_node(lib.__name__)

    '''
    # Step 2: Iterate through the attributes of the library
    for name, obj in inspect.getmembers(lib):
        if inspect.ismodule(obj):
            G.add_node(name)
            G.add_edge(lib.__name__, name)  # Link the library to the module
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.isclass(obj):
            G.add_node(name)
            G.add_edge(lib.__name__, name)  # Link the library to its functions
        elif inspect.isclass(obj):
            G.add_node(name)
            G.add_edge(lib.__name__, name)  # Link the library to its classes
            # Explore the methods inside the class
            for method_name, method_obj in inspect.getmembers(obj):
                if inspect.isfunction(method_obj):
                    G.add_node(method_name)
                    G.add_edge(name, method_name)  # Link the class to its methods

    return G
    '''
    def populate_graph(graph=G, box=inspect.getmembers(lib)):
        for name, obj in box:
            node_name = "." + name
            if 




# Step 3: Visualize the graph
def plot_graph(G):
    pos = nx.spring_layout(G)  # Position nodes using spring layout
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
    plt.show()

# Explore the 'math' library (can be replaced with any library you need)
library = sklearn  # Replace this with the library you want to explore (e.g., numpy, pandas, etc.)
graph = explore_library(library)

# Visualize the content graph
plot_graph(graph)
