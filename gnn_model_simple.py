"""
gnn_model_simple.py

SimpleSCMGNN: A lightweight Graph Neural Network for Structured Causal Model
(SCM) decomposition step prediction.

This model takes graph-structured SCM decomposition states as input and predicts
the next operation type (`op`) among four possible classes. The architecture is
based on Graph Attention Networks (GAT/GATv2), with GraphNorm and residual
connections for more stable message passing.

Main components:
    1. Input projection from handcrafted node features to hidden embeddings.
    2. Multi-layer graph attention message passing.
    3. Graph-level pooling using global mean pooling.
    4. A simple MLP classifier for operation prediction.

Expected input:
    A PyTorch Geometric `Data` or `Batch` object with:
        - x:          node feature matrix, shape [num_nodes, 199]
        - edge_index: graph connectivity, shape [2, num_edges]
        - edge_attr:  edge feature matrix, shape [num_edges, 12]
        - batch:      batch assignment vector, shape [num_nodes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    global_mean_pool,
    GraphNorm
)


class SimpleOpPredictor(nn.Module):
    """
    A lightweight MLP classifier for operation prediction.

    The classifier receives a graph-level embedding and outputs logits over
    four operation classes. It is intentionally simple because the main
    representation learning is handled by the GNN layers.
    """

    def __init__(self, hidden_dim, num_classes=4, dropout=0.1):
        """
        Initialize the operation predictor.

        Args:
            hidden_dim (int):
                Dimension of the graph-level embedding.
            num_classes (int):
                Number of operation classes to predict.
                Default is 4.
            dropout (float):
                Dropout probability used for regularization.
        """
        super().__init__()

        # Two-layer MLP classifier:
        # hidden_dim -> hidden_dim / 2 -> num_classes
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Apply Xavier initialization to linear layers for stable training.
        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, graph_emb):
        """
        Predict operation logits from graph-level embeddings.

        Args:
            graph_emb (torch.Tensor):
                Graph embeddings with shape [batch_size, hidden_dim].

        Returns:
            torch.Tensor:
                Operation logits with shape [batch_size, num_classes].
        """
        return self.predictor(graph_emb)


class SimpleSCMGNN(nn.Module):
    """
    Graph Neural Network for SCM operation prediction.

    This model encodes a partial SCM decomposition graph and predicts the next
    operation type. It supports both GAT and GATv2 convolution layers. GATv2 is
    used by default because it provides a more expressive attention mechanism
    than the original GAT formulation.

    The model currently focuses only on operation prediction (`op`) and does not
    predict shift values or operand node selections.
    """

    def __init__(
        self,
        node_in_dim=199,
        edge_in_dim=12,
        hidden_dim=256,
        num_heads=8,
        conv_type='gatv2',
        dropout=0.1,
        num_gnn_layers=4
    ):
        """
        Initialize the SimpleSCMGNN model.

        Args:
            node_in_dim (int):
                Input node feature dimension.
                In this project, each SCM node has 199 engineered features.
            edge_in_dim (int):
                Input edge feature dimension.
                In this project, each edge has 12 engineered features.
            hidden_dim (int):
                Hidden embedding dimension used throughout the GNN.
            num_heads (int):
                Number of attention heads in each GAT/GATv2 layer.
            conv_type (str):
                Type of graph attention convolution.
                Supported values: 'gat' and 'gatv2'.
            dropout (float):
                Dropout probability for both attention and MLP layers.
            num_gnn_layers (int):
                Number of graph message-passing layers.
        """
        super().__init__()

        self.dropout = dropout
        self.conv_type = conv_type.lower()
        self.num_gnn_layers = num_gnn_layers
        self.hidden_dim = hidden_dim

        # Print model configuration for experiment tracking.
        print(f"Initializing SimpleSCMGNN:")
        print(f"   - Node features:       {node_in_dim} dims")
        print(f"   - Hidden dim:          {hidden_dim}")
        print(f"   - GNN layers:          {num_gnn_layers}")
        print(f"   - Conv type:           {conv_type}")
        print(f"   - Graph Normalization: yes")
        print(f"   - Residual Connections: yes")
        print(f"   - Task:                OP prediction only (4 classes)")

        # Project handcrafted node features into the GNN hidden space.
        self.input_proj = nn.Linear(node_in_dim, hidden_dim)

        # Store message-passing layers, normalization layers, and residual paths.
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        for i in range(num_gnn_layers):
            # Each attention head outputs hidden_dim // num_heads features.
            # With concat=True, the final output dimension becomes hidden_dim.
            if self.conv_type == 'gatv2':
                conv = GATv2Conv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    edge_dim=edge_in_dim,
                    dropout=dropout
                )

            elif self.conv_type == 'gat':
                conv = GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    edge_dim=edge_in_dim,
                    dropout=dropout
                )

            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

            self.convs.append(conv)

            # GraphNorm normalizes node embeddings within each graph,
            # improving training stability for batches of variable-size graphs.
            self.norms.append(GraphNorm(hidden_dim))

            # Identity residual path preserves previous-layer information.
            self.residuals.append(nn.Identity())

        # Final graph-level classifier for predicting operation type.
        self.op_predictor = SimpleOpPredictor(
            hidden_dim=hidden_dim,
            num_classes=4,
            dropout=dropout
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize all linear layers in the model.

        Xavier uniform initialization is used to keep activation variance stable
        across layers, while biases are initialized to zero.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data):
        """
        Run the forward pass of the model.

        Args:
            data:
                A PyTorch Geometric Data or Batch object with the following
                attributes:
                    - x:
                        Node features, shape [num_nodes, 199].
                    - edge_index:
                        Directed graph edges, shape [2, num_edges].
                    - edge_attr:
                        Edge features, shape [num_edges, edge_in_dim].
                    - batch:
                        Batch assignment vector, shape [num_nodes].
                        This is automatically provided by PyG Batch objects.

        Returns:
            dict:
                A dictionary containing:
                    - "op":
                        Operation prediction logits, shape [batch_size, 4].
        """
        x = data.x
        edge_index = data.edge_index

        # Edge features are optional, but expected in this SCM graph setting.
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        # If a single graph is used, `batch` may be absent.
        # PyG pooling functions can still handle this case in many settings.
        batch = data.batch if hasattr(data, 'batch') else None

        # Initial node embedding projection:
        # [num_nodes, node_in_dim] -> [num_nodes, hidden_dim]
        h = self.input_proj(x)

        # Multi-layer graph attention message passing.
        for i in range(self.num_gnn_layers):
            # Apply graph attention convolution using node and edge features.
            h_new = self.convs[i](h, edge_index, edge_attr)

            # Normalize node representations graph-wise.
            h_new = self.norms[i](h_new, batch)

            # Nonlinear activation.
            h_new = F.relu(h_new)

            # Dropout regularization during training.
            h_new = F.dropout(
                h_new,
                p=self.dropout,
                training=self.training
            )

            # Residual connection with a small scaling factor.
            # This helps preserve lower-layer information while avoiding
            # excessive residual dominance.
            h = h_new + 0.1 * self.residuals[i](h)

        # Pool node embeddings into graph-level embeddings.
        # Shape: [num_nodes, hidden_dim] -> [batch_size, hidden_dim]
        g = global_mean_pool(h, batch)

        # Predict operation logits from graph-level representation.
        op_logits = self.op_predictor(g)

        return {
            "op": op_logits
        }


if __name__ == "__main__":
    """
    Minimal sanity check for SimpleSCMGNN.

    This block creates a small batch of random graphs and verifies that the
    model can run a forward pass successfully. It is useful for debugging model
    shape compatibility before connecting the model to the real SCM dataset.
    """

    from torch_geometric.data import Data, Batch

    print("\n" + "=" * 60)
    print("Testing SimpleSCMGNN with GraphNorm + Residual")
    print("=" * 60 + "\n")

    # Create a small test model.
    model = SimpleSCMGNN(
        node_in_dim=199,
        edge_in_dim=12,
        hidden_dim=128,
        num_heads=4,
        conv_type='gatv2',
        dropout=0.1,
        num_gnn_layers=3
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create synthetic graph data for a shape-level sanity test.
    num_graphs = 4
    graphs = []

    for i in range(num_graphs):
        # Randomly choose the number of nodes for each graph.
        num_nodes = torch.randint(5, 12, (1,)).item()

        # Random node features matching the expected 199-dimensional input.
        x = torch.randn(num_nodes, 199)

        # Random directed edges.
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))

        # Random edge features matching the expected 12-dimensional input.
        edge_attr = torch.randn(edge_index.size(1), 12)

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        graphs.append(graph)

    # Combine multiple graphs into one PyG batch.
    batch = Batch.from_data_list(graphs)

    # Run inference without gradient tracking.
    model.eval()

    with torch.no_grad():
        output = model(batch)

    print(f"\nOutput shapes:")
    print(f"  op logits: {output['op'].shape}")

    # Convert logits to predicted class indices.
    op_pred = output['op'].argmax(dim=1)

    print(f"\nPredicted ops: {op_pred.tolist()}")
    print(f"\nGraphNorm layers:      {len(model.norms)}")
    print(f"Residual connections:  {len(model.residuals)}")

    print("\n" + "=" * 60)
    print("SimpleSCMGNN test passed!")
    print("=" * 60)