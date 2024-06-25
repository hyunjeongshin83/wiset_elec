import random
from sklearn.cluster import KMeans
import pandas as pd

class TapestryNode:
    def __init__(self, node_id, base=16):
        self.node_id = node_id
        self.base = base
        self.routing_table = {}

    def add_node(self, new_node):
        prefix_len = self.common_prefix_length(self.node_id, new_node.node_id)
        if prefix_len not in self.routing_table:
            self.routing_table[prefix_len] = []
        self.routing_table[prefix_len].append(new_node)

    def common_prefix_length(self, id1, id2):
        return sum(1 for x, y in zip(id1, id2) if x == y)

    def find_node(self, key):
        prefix_len = self.common_prefix_length(self.node_id, key)
        if prefix_len in self.routing_table:
            for node in self.routing_table[prefix_len]:
                if node.node_id == key:
                    return node
        return None

    def __repr__(self):
        return f"TapestryNode({self.node_id})"

def generate_random_id(base=16, length=8):
    return ''.join(random.choices('0123456789abcdef', k=length))

if __name__ == "__main__":
    device_data = pd.read_csv("device_data.csv")

    kmeans = KMeans(n_clusters=5, random_state=42)
    device_data['cluster'] = kmeans.fit_predict(device_data[['usage_time', 'voltage']])

    clusters = device_data['cluster'].unique()
    tapestry_nodes = {cluster: TapestryNode(generate_random_id()) for cluster in clusters}

    for cluster, node in tapestry_nodes.items():
        cluster_data = device_data[device_data['cluster'] == cluster]
        for _, row in cluster_data.iterrows():
            node.add_node(TapestryNode(row['model_number']))

    for cluster, node in tapestry_nodes.items():
        print(f"Cluster {cluster}: {node}")

    print('Tapestry DHT setup complete.')
