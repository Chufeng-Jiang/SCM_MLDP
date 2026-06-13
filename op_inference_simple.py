"""
op_inference_simple.py

GNN-based operation inference script for Structured Causal Model (SCM)
decomposition.

This script loads a trained SimpleSCMGNN checkpoint and applies it to one or
more inference requests. Each request contains a target constant, the current
multiplier/constant to be decomposed, and an optional history of previous
decomposition steps. The script reconstructs a PyTorch Geometric graph from the
history, computes the same 199-dimensional node feature representation expected
by the model, and predicts the next operation type.

Predicted operation classes:
    0: SPLUS   -> shifted-plus decomposition
    1: SMINUS  -> shifted-minus decomposition
    2: MINUSS  -> minus-shifted decomposition
    3: BASE    -> base-case / terminal operation

Main workflow:
    1. Load checkpoint and recover model hyperparameters.
    2. Rebuild graph features for each inference request.
    3. Run the trained GNN in evaluation mode.
    4. Export operation probabilities to CSV for downstream analysis.

This file is intended for research-oriented inference and confidence analysis,
not for model training.
"""

import torch
import json
import csv
import argparse
import math
import time
from datetime import timedelta
from typing import List, Dict, Any, Optional
from torch_geometric.data import Data

from gnn_model_simple import SimpleSCMGNN
from graph_dataset import SCMGraphDataset


class GNNSimpleInference:
    """
    Lightweight inference wrapper for SimpleSCMGNN.

    The class encapsulates checkpoint loading, graph reconstruction, feature
    computation, single-step inference, batch inference, and CSV export. This
    keeps the command-line interface simple while preserving the same feature
    construction protocol used during training.
    """
    
    def __init__(self, model_path, device='cuda'):
        """
        Load a trained SimpleSCMGNN checkpoint and prepare inference utilities.

        Args:
            model_path (str):
                Path to the saved model checkpoint. The checkpoint is expected
                to contain `model_state_dict` and optionally a `config` dict.
            device (str):
                Preferred inference device. If CUDA is requested but unavailable,
                the script automatically falls back to CPU.
        """

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Device: {self.device}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})
        
        print(f"📋 Detected config parameters:")
        for key in ['hidden_dim', 'num_gnn_layers', 'conv_type', 'num_heads']:
            if key in config:
                print(f"   - {key}: {config[key]}")
        
        # Recover the minimal model configuration required for inference.
        model_config = {
            'node_in_dim': config.get('node_in_dim', 199),
            'edge_in_dim': config.get('edge_in_dim', 12),
            'hidden_dim': config.get('hidden_dim', 256),
            'num_heads': config.get('num_heads', 8),
            'conv_type': config.get('conv_type', 'gatv2'),
            'num_gnn_layers': config.get('num_gnn_layers', 4),
            'dropout': 0.0,  # Disable dropout during deterministic inference.
        }
        
        self.model = SimpleSCMGNN(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
        print(f"   - Hidden dim: {model_config['hidden_dim']}")
        print(f"   - GNN layers: {model_config['num_gnn_layers']}")
        print(f"   - Conv type: {model_config['conv_type']}")
        
        # Create a lightweight dataset helper for reusing feature functions.
        # __new__ avoids loading the full dataset while still allowing access
        # to feature-computation methods defined in SCMGraphDataset.
        self.dataset_helper = SCMGraphDataset.__new__(SCMGraphDataset)
        self.dataset_helper.MAX_SHIFT = 32  # Keep feature computation compatible with training.
        self.dataset_helper.MAX_NODES = 11
        self.dataset_helper.MAX_OP = 4
        
        self.op_names = {0: 'SPLUS', 1: 'SMINUS', 2: 'MINUSS', 3: 'BASE'}
    
    def _safe_compute_shifted_operand_features(self, eq, history_equations, nodes_set):
        """
        Compute shifted-operand features with defensive fallbacks.

        During inference, partial histories may not contain all operands used by
        the original decomposition. Invalid or missing operand indices are mapped
        to a zero vector so that feature dimensionality remains stable.
        """
        left_idx = eq.get("left", -1)
        right_idx = eq.get("right", -1)
        
        # Missing operands are represented by a zero feature vector.
        if left_idx == -1 or right_idx == -1:
            return [0.0] * 44  # shifted operand features is 44-dim
        
        # Compute features only when operand indices are within the expected range.
        if 0 <= left_idx <= 9 and 0 <= right_idx <= 9:
            try:
                return self.dataset_helper.compute_shifted_operand_features(
                    eq, history_equations, nodes_set
                )
            except:
                return [0.0] * 44
        else:
            return [0.0] * 44
    
    def _safe_compute_operand_relationship_features(self, eq, history_equations, nodes_set):
        """
        Compute operand-relationship features with robust missing-index handling.

        The function protects inference from incomplete graph histories by
        returning a zero vector whenever operands are unavailable or invalid.
        """
        left_idx = eq.get("left", -1)
        right_idx = eq.get("right", -1)

        # Missing operands are represented by a zero feature vector.
        if left_idx == -1 or right_idx == -1:
            return [0.0] * 10
        
        if 0 <= left_idx <= 9 and 0 <= right_idx <= 9:
            try:
                return self.dataset_helper.compute_operand_relationship_features(
                    eq, history_equations, nodes_set
                )
            except:
                return [0.0] * 10
        else:
            return [0.0] * 10
    
    def _compute_node_features_simple(self, mult, target, eq, eq_idx, 
                                      history_equations, graph_idx, 
                                      num_graph_nodes, sorted_nodes, 
                                      eq_idx_to_graph_idx):
        """
        Construct the 199-dimensional node feature vector for one graph node.

        The feature vector mirrors the representation used during model training:
        compact numerical features, operand/shift features, dependency features,
        bit-level features, target-distance features, and special pattern flags.
        Any non-finite value is replaced with zero before returning.
        """
        
        op = eq.get('op', 3)
        nodes_set = set(sorted_nodes)
        
        # 1. Compact number features (22-dim)
        mult_feat = self.dataset_helper.compute_compact_number_features(mult, target)
        
        # 2. Shifted operand features (44-dim)
        shifted_feat = self._safe_compute_shifted_operand_features(
            eq, history_equations, nodes_set
        )
        
        # 3. Operation one-hot (4-dim)
        op_feat = self.dataset_helper.one_hot(op, self.dataset_helper.MAX_OP)
        
        # 4. Dependency features (6-dim)
        left_idx = eq.get("left", -1)
        right_idx = eq.get("right", -1)
        left_graph_idx = eq_idx_to_graph_idx.get(left_idx, -1)
        right_graph_idx = eq_idx_to_graph_idx.get(right_idx, -1)
        dep_feat = self.dataset_helper.compute_dependency_features(
            graph_idx, left_graph_idx, right_graph_idx, num_graph_nodes
        )
        
        # 5. Tree level features (3-dim)
        if len(history_equations) > 0:
            tree_feat = self.dataset_helper.compute_tree_level_features(
                eq_idx, eq, history_equations, sorted_nodes, eq_idx_to_graph_idx
            )
        else:
            tree_feat = [0.0] * 3
        
        # 6. Operand relationship features (10-dim)
        operand_feat = self._safe_compute_operand_relationship_features(
            eq, history_equations, nodes_set
        )
        
        # 7. Bit distance features (7-dim)
        bit_feat = self.dataset_helper.compute_bit_distance_features(mult, target)
        
        # 8. Gap features (14-dim)
        gap_feat = self.dataset_helper.compute_gap_features(mult, target)
        
        # 9. Theoretical shifts (10-dim)
        shift_feat = self.dataset_helper.compute_theoretical_shifts(mult, target)
        
        # 10. Pairwise potential stats (30-dim)
        if len(history_equations) > 0:
            try:
                pair_feat = self.dataset_helper.compute_pairwise_potential_stats(
                    graph_idx, sorted_nodes, history_equations, target
                )
            except:
                pair_feat = [0.0] * 30
        else:
            pair_feat = [0.0] * 30
        
        # 11. Positional features (2-dim)
        pos_feat = self.dataset_helper.compute_positional_features(
            graph_idx, num_graph_nodes
        )
        
        # 12. Top-down specific features (12-dim)
        if len(history_equations) > 0:
            try:
                topdown_feat = self.dataset_helper.compute_topdown_specific_features(
                    mult, history_equations, eq_idx, sorted_nodes
                )
            except:
                topdown_feat = [0.0] * 12
        else:
            topdown_feat = [0.0] * 12
        
        # 13. Mult bit features (32-dim)
        mult_bit_feat = self.dataset_helper.compute_bit_features(mult)
        
        # 14. Special pattern features (6-dim)
        special_pattern_feat = self.dataset_helper.compute_special_pattern_features(mult)
        
        # Concatenate all feature groups into the final node representation.
        all_feat = (
            mult_feat + shifted_feat + op_feat + dep_feat + 
            tree_feat + operand_feat + bit_feat + gap_feat +
            shift_feat + pair_feat + pos_feat + topdown_feat + 
            mult_bit_feat + special_pattern_feat
        )
        
        # Replace NaN/Inf values to avoid invalid tensor values at inference time.
        all_feat = [x if math.isfinite(x) else 0.0 for x in all_feat]
        
        # Enforce the expected model input dimensionality.
        if len(all_feat) != 199:
            if len(all_feat) < 199:
                all_feat.extend([0.0] * (199 - len(all_feat)))
            else:
                all_feat = all_feat[:199]
        
        return all_feat
    
    def build_graph_from_history(self, current_mult, history_equations, target):
        """
        Reconstruct a PyTorch Geometric graph from an inference history.

        Args:
            current_mult (int):
                Current multiplier/constant that the model should decompose.
            history_equations (list[dict]):
                Previous decomposition steps. Each step may include `mult`, `op`,
                `shift`, `left`, and `right`.
            target (int):
                Original target constant for contextual feature computation.

        Returns:
            torch_geometric.data.Data:
                A graph containing node features, directed edges, and edge
                attributes suitable for SimpleSCMGNN inference.
        """
        k = len(history_equations)
        
        # Use multiplier values as inference-time graph nodes.
        if k > 0:
            sorted_nodes = sorted(
                set(eq['mult'] for eq in history_equations),
                key=lambda x: x
            )
        else:
            sorted_nodes = []
        
        # Mapping from equation index to graph node index
        eq_idx_to_graph_idx = {}
        for eq_idx, eq in enumerate(history_equations):
            mult_val = eq['mult']
            if mult_val in sorted_nodes:
                graph_idx = sorted_nodes.index(mult_val)
                eq_idx_to_graph_idx[eq_idx] = graph_idx
        
        num_graph_nodes = len(sorted_nodes)
        
        # Compute node features for each node
        x_list = []
        for graph_idx, mult in enumerate(sorted_nodes):
            # Find the equation that created this mult
            eq = None
            eq_idx = None
            for i, e in enumerate(history_equations):
                if e['mult'] == mult:
                    eq = e
                    eq_idx = i
                    break
            
            if eq is None:
                # Should not happen
                feat = [0.0] * 199
            else:
                feat = self._compute_node_features_simple(
                    mult=mult,
                    target=target,
                    eq=eq,
                    eq_idx=eq_idx,
                    history_equations=history_equations[:eq_idx],
                    graph_idx=graph_idx,
                    num_graph_nodes=num_graph_nodes,
                    sorted_nodes=sorted_nodes,
                    eq_idx_to_graph_idx=eq_idx_to_graph_idx
                )
            
            x_list.append(feat)
        
        # Add a prediction node representing the current decomposition state.
        pred_node_feat = self._compute_node_features_simple(
            mult=current_mult,
            target=target,
            eq={'op': 3, 'left': -1, 'right': -1, 'shift': 0, 'mult': current_mult},
            eq_idx=k,
            history_equations=history_equations,
            graph_idx=num_graph_nodes,
            num_graph_nodes=num_graph_nodes + 1,
            sorted_nodes=sorted_nodes + [current_mult],
            eq_idx_to_graph_idx=eq_idx_to_graph_idx
        )
        x_list.append(pred_node_feat)
        
        x = torch.tensor(x_list, dtype=torch.float32)
        
        # Build PyG edge_index and edge attributes.
        edge_src = []
        edge_dst = []
        edge_attr_list = []
        
        pred_node_idx = num_graph_nodes
        
        # 1. Historical dependency edges reconstructed from previous steps.
        for eq_idx, eq in enumerate(history_equations):
            left_idx = eq.get('left', -1)
            right_idx = eq.get('right', -1)
            
            src_graph_idx = eq_idx_to_graph_idx.get(eq_idx, -1)
            if src_graph_idx == -1:
                continue
            
            # Left-operand dependency edge.
            if left_idx != -1 and left_idx in eq_idx_to_graph_idx:
                dst_graph_idx = eq_idx_to_graph_idx[left_idx]
                edge_src.append(src_graph_idx)
                edge_dst.append(dst_graph_idx)
                op_code = eq.get('op', 3)
                shift = eq.get('shift', 0)
                edge_feat = self.dataset_helper.compute_dependency_edge_features(
                    op_code, shift, is_left=True
                )
                edge_attr_list.append(edge_feat)
            
            # Right-operand dependency edge.
            if right_idx != -1 and right_idx in eq_idx_to_graph_idx:
                dst_graph_idx = eq_idx_to_graph_idx[right_idx]
                edge_src.append(src_graph_idx)
                edge_dst.append(dst_graph_idx)
                op_code = eq.get('op', 3)
                shift = eq.get('shift', 0)
                edge_feat = self.dataset_helper.compute_dependency_edge_features(
                    op_code, shift, is_left=False
                )
                edge_attr_list.append(edge_feat)
        
        # 2. Self-loops preserve each node's own representation.
        for i in range(num_graph_nodes):
            edge_src.append(i)
            edge_dst.append(i)
            edge_attr_list.append([1.0] + [0.0] * 11)
        
        # 3. Connect the prediction node to historical nodes for context.
        for i in range(num_graph_nodes):
            edge_src.append(pred_node_idx)
            edge_dst.append(i)
            edge_attr_list.append([0.0] * 12)
        
        # 4. Add a self-loop for the prediction node.
        if num_graph_nodes == 0 or True:  # Always add self-loop for prediction node
            edge_src.append(pred_node_idx)
            edge_dst.append(pred_node_idx)
            edge_attr_list.append([1.0] + [0.0] * 11)
        
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        
        # Construct the PyTorch Geometric Data object consumed by the model.
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        
        return data
    
    def predict_next_action(self, current_mult, history_equations, target):
        """
        Predict the next operation type for a single decomposition state.

        Returns a dictionary containing the predicted operation ID, its symbolic
        name, and the full probability distribution over all operation classes.
        """
        # Build graph data
        data = self.build_graph_from_history(current_mult, history_equations, target)
        data = data.to(self.device)
        
        # Model inference
        with torch.no_grad():
            pred = self.model(data)
            
            # Extract op predictions
            op_logits = pred['op']  # shape: [1, 4]
            op_probs = torch.softmax(op_logits, dim=-1).squeeze(0)  # shape: [4]
            op_predicted = op_probs.argmax().item()
            
            # Convert probabilities to a plain Python dictionary for logging/CSV export.
            op_probabilities = {
                i: float(op_probs[i].item()) 
                for i in range(len(op_probs))
            }
            
            return {
                'op_predicted': op_predicted,
                'op_name': self.op_names[op_predicted],
                'op_probabilities': op_probabilities,
            }
    
    def load_input_file(self, json_path: str, start_c: Optional[int] = None, 
                       end_c: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load inference requests from a JSON file with optional range filtering.

        The expected input format is a list of request dictionaries containing
        `target`, `current_mult`, and optionally `history`. The optional range
        filter is applied to `current_mult`.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Apply optional filtering to restrict inference to a constant range.
        if start_c is not None or end_c is not None:
            original_count = len(data)
            filtered_data = []
            for item in data:
                c = item.get('current_mult')
                if c is not None:
                    if start_c is not None and c < start_c:
                        continue
                    if end_c is not None and c > end_c:
                        continue
                    filtered_data.append(item)
            data = filtered_data
            print(f"📂 Loaded {len(data)} requests from {json_path} (filtered from {original_count})")
            if start_c is not None:
                print(f"   Range filter: c >= {start_c}")
            if end_c is not None:
                print(f"   Range filter: c <= {end_c}")
        else:
            print(f"📂 Loaded {len(data)} requests from {json_path}")
        
        return data
    
    def predict_batch_from_file(self, json_path: str, output_path: Optional[str] = None,
                               start_c: Optional[int] = None, end_c: Optional[int] = None):
        """
        Run batch inference over all requests in an input JSON file.

        If `output_path` is provided, the script writes a CSV file containing
        the predicted probability for each operation class and basic timing
        statistics for reproducibility and profiling.
        """
        # Record start time
        start_time = time.time()
        
        requests = self.load_input_file(json_path, start_c, end_c)
        results = []
        
        print(f"\n{'='*70}")
        print(f"🚀 Beginning batch prediction...")
        if start_c is not None or end_c is not None:
            range_str = f"c ∈ [{start_c if start_c is not None else '∞'}, {end_c if end_c is not None else '∞'}]"
            print(f"📊 Processing range: {range_str}")
        print(f"{'='*70}")
        
        for idx, req in enumerate(requests):
            target = req['target']
            current_mult = req['current_mult']
            history = req.get('history', [])
            
            # Track progress and elapsed time for long-running inference jobs.
            progress = (idx + 1) / len(requests) * 100
            elapsed = time.time() - start_time
            
            print(f"\n📊 Request {idx + 1}/{len(requests)} ({progress:.1f}%) - Elapsed: {timedelta(seconds=int(elapsed))}")
            print(f"  Target: {target}")
            print(f"  Current mult: {current_mult}")
            print(f"  History steps: {len(history)}")
            
            try:
                prediction = self.predict_next_action(current_mult, history, target)
                
                print(f"  🎯 Prediction Result:")
                print(f"    Best Op: {prediction['op_name']} (op={prediction['op_predicted']})")
                
                print(f"    Op Probabilities:")
                probs = prediction['op_probabilities']
                for op_id in range(4):
                    op_name = self.op_names[op_id]
                    prob = probs[op_id]
                    bar = '█' * int(prob * 20)
                    print(f"      {op_name:8s} (op={op_id}): {prob:.4f} ({prob*100:5.2f}%) {bar}")
                
                results.append({
                    'current_mult': current_mult,
                    'prediction': prediction
                })
            except Exception as e:
                print(f"  ❌ Inference Failed: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append({
                    'current_mult': current_mult,
                    'error': str(e)
                })
        
        # Summarize throughput after all requests have been processed.
        total_time = time.time() - start_time
        avg_time_per_sample = total_time / len(requests) if len(requests) > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"✅ Inference Done!")
        print(f"{'='*70}")
        print(f"⏱️  Total time: {timedelta(seconds=int(total_time))} ({total_time:.2f}s)")
        print(f"⏱️  Average time per sample: {avg_time_per_sample:.3f}s")
        print(f"📊 Processed samples: {len(requests)}")
        print(f"{'='*70}")
        
        # Save to CSV
        if output_path:
            self._save_to_csv(results, output_path, total_time, avg_time_per_sample)
        
        return results
    
    def _save_to_csv(self, results: List[Dict], output_path: str, 
                    total_time: float, avg_time: float):
        """
        Save operation probabilities and timing metadata to a CSV file.

        The first rows store timing information as comment-like rows, followed
        by one row per input constant. Failed samples are recorded with zero
        probabilities to preserve row alignment.
        """
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write timing information as comments
            writer.writerow([f'# Total processing time: {total_time:.2f}s ({timedelta(seconds=int(total_time))})'])
            writer.writerow([f'# Average time per sample: {avg_time:.3f}s'])
            writer.writerow([f'# Total samples processed: {len(results)}'])
            writer.writerow([])  # Empty line
            
            # Write header
            writer.writerow(['Constant', 'prob_SPLUS', 'prob_SMINUS', 'prob_MINUSS', 'prob_BASE'])
            
            # Write data rows
            for result in results:
                if 'error' in result:
                    # If there's an error, write current_mult and zeros
                    writer.writerow([result['current_mult'], 0.0, 0.0, 0.0, 0.0])
                else:
                    current_mult = result['current_mult']
                    probs = result['prediction']['op_probabilities']
                    
                    # Round probabilities to 4 decimal places
                    row = [
                        current_mult,
                        round(probs[0], 4),
                        round(probs[1], 4),
                        round(probs[2], 4),
                        round(probs[3], 4)
                    ]
                    writer.writerow(row)
        
        print(f"💾 Results saved to CSV: {output_path}")
    
    def predict_single(self, target: int, current_mult: int, 
                      history: List[Dict] = None):
        """
        Run and print inference for one manually specified decomposition state.

        This method is useful for debugging individual examples before running
        the full batch prediction pipeline.
        """
        if history is None:
            history = []
        
        print(f"\n{'='*70}")
        print(f"🔍 Single Step Inference")
        print(f"{'='*70}")
        print(f"Target constant: {target}")
        print(f"Current mult: {current_mult}")
        print(f"History steps: {len(history)}")
        
        prediction = self.predict_next_action(current_mult, history, target)
        
        print(f"\n🎯 Prediction Result:")
        print(f"  Best Op: {prediction['op_name']} (op={prediction['op_predicted']})")
        
        print(f"\n  Op Probabilities:")
        probs = prediction['op_probabilities']
        for op_id in range(4):
            op_name = self.op_names[op_id]
            prob = probs[op_id]
            bar = '█' * int(prob * 30)
            print(f"    {op_name:8s} (op={op_id}): {prob:.4f} ({prob*100:5.2f}%) {bar}")
        
        return prediction


def main():
    """Parse command-line arguments and run batch inference."""
    parser = argparse.ArgumentParser(description='GNN Simple Inference Script - CSV Output (Op Only)')
    parser.add_argument('--model', type=str, required=True,
                        help='model path (e.g., best_model_simple.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='input JSON file path')
    parser.add_argument('--output', type=str, default=None,
                        help='output CSV file path (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device (cuda/cpu)')
    parser.add_argument('--start-c', type=int, default=None,
                        help='start constant (inclusive), e.g., --start-c 1000')
    parser.add_argument('--end-c', type=int, default=None,
                        help='end constant (inclusive), e.g., --end-c 2000')
    
    args = parser.parse_args()
    
    # Validate range parameters
    if args.start_c is not None and args.end_c is not None:
        if args.start_c > args.end_c:
            print(f"❌ Error: start-c ({args.start_c}) must be <= end-c ({args.end_c})")
            return
    
    # Initialize the inference wrapper and load the trained checkpoint.
    inferencer = GNNSimpleInference(
        model_path=args.model,
        device=args.device
    )
    
    # Run batch inference and optionally export results to CSV.
    inferencer.predict_batch_from_file(
        json_path=args.input,
        output_path=args.output,
        start_c=args.start_c,
        end_c=args.end_c
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        print("="*70)
        print("GNN Simple Inference Script - CSV Output (Op Only)")
        print("="*70)
        print("\n✨ This script performs inference using SimpleSCMGNN model")
        print("   (predicts only the operation type: SPLUS/SMINUS/MINUSS/BASE)")
        print("\nUsage:")
        print("  Basic usage:")
        print("    python op_inference_simple.py --model ./model_results/best_model_simple.pth --input ./test_numbers/inference_input/17input.json --output ./test_numbers/inference_input/17confidence.csv")
        print("\n  With constant range filtering:")
        print("    python op_inference_simple.py --model ./model_results/best_model_simple.pth --input ./test_numbers/inference_input/17input.json --output ./test_numbers/inference_input/17confidence.csv --start-c 1000 --end-c 2000")