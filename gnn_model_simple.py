"""gnn_model_simple.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, GraphNorm


class SimpleOpPredictor(nn.Module):
    def __init__(self, hidden_dim, num_classes=4, dropout=0.1):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, graph_emb):
        return self.predictor(graph_emb)


class SimpleSCMGNN(nn.Module):
    def __init__(self,
                 node_in_dim=199,
                 edge_in_dim=12,
                 hidden_dim=256,
                 num_heads=8,
                 conv_type='gatv2',
                 dropout=0.1,
                 num_gnn_layers=4):
        super().__init__()

        self.dropout = dropout
        self.conv_type = conv_type.lower()
        self.num_gnn_layers = num_gnn_layers
        self.hidden_dim = hidden_dim

        print(f"Initializing SimpleSCMGNN:")
        print(f"   - Node features:       {node_in_dim} dims")
        print(f"   - Hidden dim:          {hidden_dim}")
        print(f"   - GNN layers:          {num_gnn_layers}")
        print(f"   - Conv type:           {conv_type}")
        print(f"   - Graph Normalization: yes")
        print(f"   - Residual Connections: yes")
        print(f"   - Task:                OP prediction only (4 classes)")

        self.input_proj = nn.Linear(node_in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        for i in range(num_gnn_layers):
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
            self.norms.append(GraphNorm(hidden_dim))
            self.residuals.append(nn.Identity())

        self.op_predictor = SimpleOpPredictor(
            hidden_dim=hidden_dim,
            num_classes=4,
            dropout=dropout
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Data object with attributes:
                - x:          [num_nodes, 199]        - node features
                - edge_index: [2, num_edges]           - edge indices
                - edge_attr:  [num_edges, edge_in_dim] - edge features (optional)
                - batch:      [num_nodes]              - batch indices (optional)

        Returns:
            dict with key:
                - op: [B, 4] - op prediction logits
        """
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch      = data.batch     if hasattr(data, 'batch')     else None

        h = self.input_proj(x)  

        for i in range(self.num_gnn_layers):
            h_new = self.convs[i](h, edge_index, edge_attr)
            h_new = self.norms[i](h_new, batch)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new + 0.1 * self.residuals[i](h)


        g = global_mean_pool(h, batch)  
        op_logits = self.op_predictor(g)  

        return {
            "op": op_logits
        }

if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    print("\n" + "="*60)
    print("Testing SimpleSCMGNN with GraphNorm + Residual")
    print("="*60 + "\n")

    # Create model
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

    # Create sample data
    num_graphs = 4
    graphs = []

    for i in range(num_graphs):
        num_nodes  = torch.randint(5, 12, (1,)).item()
        x          = torch.randn(num_nodes, 199)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_attr  = torch.randn(edge_index.size(1), 12)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(graph)

    batch = Batch.from_data_list(graphs)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch)

    print(f"\nOutput shapes:")
    print(f"  op logits: {output['op'].shape}")
    op_pred = output['op'].argmax(dim=1)
    
    print(f"\nPredicted ops: {op_pred.tolist()}")
    print(f"\nGraphNorm layers:      {len(model.norms)}")
    print(f"Residual connections:  {len(model.residuals)}")
    
    print("\n" + "="*60)
    print("SimpleSCMGNN test passed!")
    print("="*60)