import matplotlib.pyplot as plt
import networkx as nx

# Define the directory structure as a dictionary
directory_structure = {
    "/": ["bin", "boot", "dev", "etc", "home", "lib", "lib64", "media", "mnt", "opt", "proc", "root", "run", "sbin", "srv", "sys", "tmp", "usr", "var"],
    "boot": ["grub", "vmlinuz"],
    "dev": ["sda", "tty"],
    "etc": ["passwd", "fstab"],
    "home": ["user1", "user2"],
    "lib": ["libc.so.6", "libm.so.6"],
    "media": ["cdrom", "usb"],
    "mnt": ["external", "temp"],
    "opt": ["myapp"],
    "proc": ["cpuinfo", "meminfo"],
    "run": ["systemd"],
    "sbin": ["fsck", "reboot"],
    "sys": ["devices", "firmware"],
    "tmp": ["temp_file1", "temp_file2"],
    "usr": ["bin", "lib", "share"],
    "var": ["cache", "log", "mail"]
}

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges
def add_edges(parent, children):
    for child in children:
        G.add_edge(parent, child)
        if child in directory_structure:
            add_edges(child, directory_structure[child])

# Initialize the graph with root directory
add_edges("/", directory_structure["/"])

# Using pygraphviz_layout for hierarchical layout
pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

# Draw the graph
plt.figure(figsize=(20, 16))  # Increase the figure size to accommodate more nodes
nx.draw(G, pos, 
        with_labels=True, 
        node_size=1200,     # Reduce node size
        node_color="lightblue", 
        font_size=8,        # Reduce font size
        font_weight="bold", 
        arrows=False,
        width=0.5,          # Thinner edges for better clarity
        edge_color="gray")  # Softer edge color

# Title and show the graph
plt.title("Linux Filesystem Hierarchy", fontsize=16)
plt.show()