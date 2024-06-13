import matplotlib.pyplot as plt
import networkx as nx

# Define the directory structure for Windows system
windows_structure = {
    "C:": ["Windows", "Program Files", "Users", "Temp"],
    "Users": ["User1", "User2"],
    "D:": ["Documents", "Presentations", "Templates"],
    "Documents": ["Reports", "Spreadsheets"],
    "E:": ["Projects", "Libraries", "Scripts"],
    "Projects": ["Project1", "Project2"],
    "Libraries": ["Lib1", "Lib2"],
    "Scripts": ["Script1", "Script2"]
}

# Create a directed graph
G_win = nx.DiGraph()

# Add nodes and edges
def add_edges_windows(parent, children):
    for child in children:
        G_win.add_edge(parent, child)
        if child in windows_structure:
            add_edges_windows(child, windows_structure[child])

# Initialize the graph with root directories
add_edges_windows("C:", windows_structure["C:"])
add_edges_windows("D:", windows_structure["D:"])
add_edges_windows("E:", windows_structure["E:"])

# Using pygraphviz_layout for hierarchical layout
pos_win = nx.nx_agraph.graphviz_layout(G_win, prog="dot")

# Draw the graph
plt.figure(figsize=(15, 11))  # Increase the figure size to accommodate more nodes
nx.draw(G_win, pos_win, 
        with_labels=True, 
        node_size=1200,     # Reduce node size
        node_color="lightgreen", 
        font_size=8,        # Reduce font size
        font_weight="bold", 
        arrows=False,
        width=0.5,          # Thinner edges for better clarity
        edge_color="gray")  # Softer edge color

# Title and show the graph
plt.title("Windows Filesystem Hierarchy", fontsize=16)
plt.show()