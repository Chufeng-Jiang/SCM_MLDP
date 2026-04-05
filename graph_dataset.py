"""graph_dataset_integrated.py: PyG Dataset with ALL features integrated (199-dim nodes)

FEATURE DIMENSIONS:
- Node features: 199-dim (170 original + 8 factor_quality + 21 decomposition_pattern)
- Edge features: 12-dim
- Target features: 22-dim

NEW FEATURES ADDED:
1. Factor Quality Features (8-dim) - Based on Picat decomposition patterns
2. Decomposition Pattern Features (21-dim) - Split potential analysis
3. Reuse Pattern Detection - Enhanced reuse pattern classification
"""

import json
import torch
from torch_geometric.data import Data, Dataset
import os
import math
import numpy as np
from functools import lru_cache
import pickle
from pathlib import Path


class SCMGraphDataset(Dataset):
    """
    PyG Dataset for SCM (Single Constant Multiplication) with Top-Down decomposition.
    
    LAZY LOADING WITH CACHE VERSION with ALL FEATURES INTEGRATED
    
    Node features: 199-dim breakdown:
      - mult features: 22-dim
      - shifted_operand features: 44-dim  
      - op features: 4-dim
      - dependency features: 6-dim
      - tree_level features: 8-dim
      - operand_relationship features: 12-dim
      - bit_distance features: 5-dim
      - gap features: 9-dim
      - theoretical_shifts features: 3-dim
      - pairwise_potential features: 8-dim
      - positional features: 6-dim
      - topdown_specific features: 5-dim
      - factor_quality features: 8-dim (NEW)
      - decomposition_pattern features: 21-dim (NEW)
      - bit features: 32-dim
      - special_pattern features: 6-dim
    
    Edge features: 12-dim
    """

    MAX_OP = 4
    MAX_NODES = 11

    def __init__(self, json_path, transform=None, pre_transform=None, max_prefix_len=8,
                 split_type='all', train_targets=None, test_targets=None, max_shift=32,
                 cache_size=5000, disk_cache_dir=None, use_disk_cache=False):
        """
        Args:
            json_path: Path to JSON file
            max_prefix_len: Maximum prefix length
            split_type: 'all', 'train', or 'test'
            train_targets: List of training targets
            test_targets: List of test targets
            max_shift: Maximum shift value
            cache_size: Size of in-memory LRU cache (default: 5000)
            disk_cache_dir: Directory for disk cache (default: None)
            use_disk_cache: Whether to use persistent disk cache (default: False)
        """

        self.json_path = json_path
        self.max_prefix_len = max_prefix_len
        self.split_type = split_type
        self.train_targets = set(train_targets) if train_targets else set()
        self.test_targets = set(test_targets) if test_targets else set()
        self.MAX_SHIFT = max_shift
        self.cache_size = cache_size
        self.use_disk_cache = use_disk_cache

        # Setup disk cache
        if use_disk_cache:
            if disk_cache_dir is None:
                disk_cache_dir = os.path.join(
                    os.path.dirname(json_path), 
                    f"cache_integrated_{os.path.basename(json_path).replace('.json', '')}"
                )
            self.disk_cache_dir = Path(disk_cache_dir)
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.disk_cache_dir = None

        # Load and filter samples
        self._load_samples()
        
        # Initialize in-memory cache
        self._init_cache()
        
        super().__init__(os.path.dirname(json_path), transform, pre_transform)

    def _load_samples(self):
        """Load JSON and create index mapping"""
        with open(self.json_path, "r") as f:
            raw = json.load(f)

        self.samples = []
        self.dfs_orders = []
        self.index_map = []
        
        for sample_idx, sample in enumerate(raw):
            target = int(sample["c"])
            
            # Filter by split
            if self.split_type == 'train' and target not in self.train_targets:
                continue
            elif self.split_type == 'test' and target not in self.test_targets:
                continue
            
            self.samples.append(sample)
            
            # Precompute DFS order
            dfs_order = self._compute_dfs_order(sample["equations"])
            self.dfs_orders.append(dfs_order)
            
            N = len(sample["equations"])
            
            for k in range(0, min(N, self.max_prefix_len)):
                self.index_map.append((sample_idx, k))

    def _compute_dfs_order(self, equations):
        """
        Reconstruct DFS traversal order following Picat's strategy:
        1. Start from target node (equations[0])
        2. Recursively decompose left first
        3. Then decompose right
        """
        if not equations:
            return []
        
        visited = set()
        order = []
        
        def dfs(idx):
            if idx < 0 or idx >= len(equations):
                return
            
            if idx in visited:
                return
            
            visited.add(idx)
            order.append(idx)
            
            eq = equations[idx]
            left_idx = eq.get("left", -1)
            right_idx = eq.get("right", -1)
            
            if left_idx >= 0:
                dfs(left_idx)
            
            if right_idx >= 0:
                dfs(right_idx)
        
        dfs(0)
        
        return order

    def _init_cache(self):
        """Initialize in-memory cache"""
        self._cache = {}
        self._cache_order = []
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_cache_key(self, sample_idx, k):
        """Generate cache key for a graph"""
        return (sample_idx, k)

    def _get_disk_cache_path(self, sample_idx, k):
        """Get path for disk cache file"""
        if self.disk_cache_dir is None:
            return None
        return self.disk_cache_dir / f"graph_{sample_idx}_{k}.pkl"

    def _load_from_disk_cache(self, sample_idx, k):
        """Load graph from disk cache"""
        if not self.use_disk_cache:
            return None
        
        cache_path = self._get_disk_cache_path(sample_idx, k)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                cache_path.unlink()
                return None
        return None

    def _save_to_disk_cache(self, sample_idx, k, data):
        """Save graph to disk cache"""
        if not self.use_disk_cache:
            return
        
        cache_path = self._get_disk_cache_path(sample_idx, k)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass

    def _update_cache(self, key, data):
        """Update in-memory cache with LRU policy"""
        if key in self._cache:
            self._cache_order.remove(key)
        
        self._cache[key] = data
        self._cache_order.append(key)
        
        while len(self._cache) > self.cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]

    def len(self):
        """Return total number of graphs"""
        return len(self.index_map)

    def get(self, idx):
        """Generate graph on-the-fly for given index with caching"""
        sample_idx, k = self.index_map[idx]
        cache_key = self._get_cache_key(sample_idx, k)
        
        # Check in-memory cache
        if cache_key in self._cache:
            self.cache_hits += 1
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._cache[cache_key]
        
        self.cache_misses += 1
        
        # Check disk cache
        if self.use_disk_cache:
            data = self._load_from_disk_cache(sample_idx, k)
            if data is not None:
                self._update_cache(cache_key, data)
                return data
        
        # Build graph
        sample = self.samples[sample_idx]
        equations = sample["equations"]
        target = int(sample["c"])
        dfs_order = self.dfs_orders[sample_idx]
        
        data = self._build_graph(equations, target, k, dfs_order)
        
        # Cache the built graph
        self._update_cache(cache_key, data)
        
        # Save to disk cache
        if self.use_disk_cache:
            self._save_to_disk_cache(sample_idx, k, data)
        
        return data

    def get_cache_stats(self):
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

    def clear_cache(self):
        """Clear in-memory cache"""
        self._cache.clear()
        self._cache_order.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_disk_cache(self):
        """Clear disk cache"""
        if self.disk_cache_dir and self.disk_cache_dir.exists():
            for cache_file in self.disk_cache_dir.glob("graph_*.pkl"):
                cache_file.unlink()

    def precompute_all(self, verbose=True):
        """Precompute all graphs and cache them"""
        if verbose:
            print(f"Precomputing {len(self)} graphs...")
        
        for idx in range(len(self)):
            _ = self.get(idx)
            if verbose and (idx + 1) % 1000 == 0:
                stats = self.get_cache_stats()
                print(f"  Progress: {idx + 1}/{len(self)} "
                      f"(Cache: {stats['cache_size']}/{stats['max_cache_size']}, "
                      f"Hit rate: {stats['hit_rate']:.2%})")
        
        if verbose:
            stats = self.get_cache_stats()
            print(f"✓ Precomputation complete!")
            print(f"  Final cache stats: {stats}")

    # ========================================
    # Feature Computation Methods
    # ========================================
    
    def one_hot(self, idx, dim):
        v = [0.0] * dim
        if 0 <= idx < dim:
            v[idx] = 1.0
        return v

    def compute_bit_features(self, value):
        """32-bit representation of value (32-dim)"""
        if value < 0 or value >= (1 << 32):
            return [0.0] * 32
        
        bit_features = []
        for i in range(32):
            bit = float((value >> i) & 1)
            bit_features.append(bit)
        
        return bit_features

    def compute_log_positional_features(self, value, reference=None):
        """Log-space continuous representation (10-dim)"""
        if value == 0:
            return [0.0] * 10
        
        log_val = math.log2(value + 1)
        log_norm = log_val / 32.0
        log_val_sin = math.sin(log_val * math.pi / 16)
        log_val_cos = math.cos(log_val * math.pi / 16)
        
        bit_len = value.bit_length()
        bit_len_norm = bit_len / 32.0
        
        leading_zeros = 32 - bit_len if bit_len > 0 else 32
        leading_zeros_norm = leading_zeros / 32.0
        
        trailing_zeros = (value & -value).bit_length() - 1 if value > 0 else 0
        trailing_zeros_norm = trailing_zeros / 32.0
        
        ones_count = bin(value).count('1')
        bit_density = ones_count / bit_len if bit_len > 0 else 0.0
        
        if reference is not None and reference > 0:
            log_ratio = math.log2((value + 1) / (reference + 1))
            log_ratio_norm = max(-1.0, min(1.0, log_ratio / 10.0))
            
            scale_ratio = value / reference
            scale_log = math.log10(scale_ratio + 1e-10)
            scale_norm = max(-1.0, min(1.0, scale_log))
            
            same_magnitude = float(abs(log_ratio) < 1.0)
        else:
            log_ratio_norm = 0.0
            scale_norm = 0.0
            same_magnitude = 0.0
        
        return [
            log_norm, log_val_sin, log_val_cos,
            bit_len_norm, leading_zeros_norm, trailing_zeros_norm,
            bit_density, log_ratio_norm, scale_norm, same_magnitude
        ]
    
    def compute_shift_centric_features(self, value, reference=None):
        """Shift-centric features for SCM (12-dim)"""
        if value == 0:
            return [0.0] * 12
        
        trailing_zeros = (value & -value).bit_length() - 1 if value > 0 else 0
        is_power_of_2 = float((value & (value - 1)) == 0)
        
        bit_len = value.bit_length()
        nearest_pow2_lower = 1 << (bit_len - 1)
        nearest_pow2_upper = 1 << bit_len
        
        dist_to_lower = (value - nearest_pow2_lower) / nearest_pow2_lower
        dist_to_upper = (nearest_pow2_upper - value) / nearest_pow2_upper
        
        parity = float(value & 1)
        
        decomp_count = 0
        for a in range(min(bit_len + 3, 32)):
            for b in range(a):
                if value == (1 << a) + (1 << b) or value == (1 << a) - (1 << b):
                    decomp_count += 1
        decomposability = min(1.0, decomp_count / 10.0)
        
        ones_count = bin(value).count('1')
        bit_complexity = ones_count / bit_len if bit_len > 0 else 0.0
        is_sparse = float(ones_count <= 3)
        
        if reference is not None and reference > 0:
            if value < reference:
                approx_shift = math.log2(reference / value) if value > 0 else 0
            else:
                approx_shift = math.log2(value / reference) if reference > 0 else 0
            shift_hint = min(1.0, approx_shift / 32.0)
            
            best_alignment = 0.0
            for s in range(1, min(17, self.MAX_SHIFT + 1)):
                shifted = value << s
                if shifted > reference:
                    alignment = 1.0 - abs(shifted - reference) / shifted
                    best_alignment = max(best_alignment, alignment)
            
            prefer_left_shift = float(value < reference)
            prefer_right_shift = float(value > reference)
        else:
            shift_hint = 0.0
            best_alignment = 0.0
            prefer_left_shift = 0.0
            prefer_right_shift = 0.0
        
        return [
            trailing_zeros / 32.0, is_power_of_2, dist_to_lower, dist_to_upper,
            parity, decomposability, bit_complexity, is_sparse,
            shift_hint, best_alignment, prefer_left_shift, prefer_right_shift
        ]
    
    def compute_compact_number_features(self, value, reference=None):
        """Combine log-positional and shift-centric features (22-dim)"""
        log_pos = self.compute_log_positional_features(value, reference)
        shift_cent = self.compute_shift_centric_features(value, reference)
        return log_pos + shift_cent

    def compute_dependency_features(self, node_idx, left_idx, right_idx, prefix_len):
        """Dependency relationship features (6-dim)"""
        if left_idx >= 0:
            left_distance = abs(left_idx - node_idx) / max(prefix_len - 1, 1)
        else:
            left_distance = 0.0
        
        if right_idx >= 0:
            right_distance = abs(right_idx - node_idx) / max(prefix_len - 1, 1)
        else:
            right_distance = 0.0
        
        left_depth = left_idx / max(prefix_len - 1, 1) if left_idx >= 0 else 0.0
        right_depth = right_idx / max(prefix_len - 1, 1) if right_idx >= 0 else 0.0
        
        same_dependency = float(left_idx >= 0 and left_idx == right_idx)
        
        if left_idx >= 0 and right_idx >= 0:
            dependency_gap = abs(left_idx - right_idx) / max(prefix_len - 1, 1)
        else:
            dependency_gap = 0.0
        
        return [
            left_distance, right_distance,
            left_depth, right_depth,
            same_dependency, dependency_gap
        ]
    
    def compute_tree_level_features(self, eq_idx, eq, equations, sorted_nodes, eq_idx_to_graph_idx):
        """Tree hierarchy features (8-dim)"""
        mult = eq["mult"]
        left_idx = eq.get("left", -1)
        right_idx = eq.get("right", -1)
        
        graph_idx = eq_idx_to_graph_idx.get(eq_idx, 0)
        absolute_level = graph_idx / max(len(sorted_nodes) - 1, 1)
        
        has_valid_left = left_idx >= 0 and left_idx < len(equations)
        has_valid_right = right_idx >= 0 and right_idx < len(equations)
        is_leaf = float(not (has_valid_left or has_valid_right))
        is_root = float(eq_idx == 0)
        
        if has_valid_left and has_valid_right:
            left_graph_idx = eq_idx_to_graph_idx.get(left_idx, -1)
            right_graph_idx = eq_idx_to_graph_idx.get(right_idx, -1)
            if left_graph_idx >= 0 and right_graph_idx >= 0:
                decomp_width = abs(left_graph_idx - right_graph_idx) / max(len(sorted_nodes) - 1, 1)
            else:
                decomp_width = 0.0
        else:
            decomp_width = 0.0
        
        if has_valid_left:
            left_mult = equations[left_idx]["mult"]
            left_ratio = left_mult / max(mult, 1)
        else:
            left_ratio = 0.0
        
        if has_valid_right:
            right_mult = equations[right_idx]["mult"]
            right_ratio = right_mult / max(mult, 1)
        else:
            right_ratio = 0.0
        
        if has_valid_left and has_valid_right:
            left_mult = equations[left_idx]["mult"]
            right_mult = equations[right_idx]["mult"]
            balance = min(left_mult, right_mult) / max(left_mult, right_mult, 1)
        else:
            balance = 0.0
        
        return [
            absolute_level, is_leaf, is_root, decomp_width,
            left_ratio, right_ratio, balance,
            float(has_valid_left and has_valid_right)
        ]

    def compute_operand_relationship_features(self, eq, equations, nodes_in_graph=None):
        """Relationship between node and its operands (12-dim)"""
        mult = eq["mult"]
        op = eq.get("op", 3)
        shift = eq.get("shift", 0)
        left_idx = eq.get("left", -1)
        right_idx = eq.get("right", -1)
        
        # Only access nodes in graph
        if nodes_in_graph is not None:
            left_mult = equations[left_idx]["mult"] if (left_idx >= 0 and left_idx in nodes_in_graph) else 0
            right_mult = equations[right_idx]["mult"] if (right_idx >= 0 and right_idx in nodes_in_graph) else 0
        else:
            left_mult = equations[left_idx]["mult"] if 0 <= left_idx < len(equations) else 0
            right_mult = equations[right_idx]["mult"] if 0 <= right_idx < len(equations) else 0
        
        if left_mult > 0 and mult > 0:
            left_to_mult_ratio = left_mult / mult
            left_log_diff = math.log2(mult + 1) - math.log2(left_mult + 1)
            left_log_diff_norm = left_log_diff / 32.0
        else:
            left_to_mult_ratio = 0.0
            left_log_diff_norm = 0.0
        
        if right_mult > 0 and mult > 0:
            right_to_mult_ratio = right_mult / mult
            right_log_diff = math.log2(mult + 1) - math.log2(right_mult + 1)
            right_log_diff_norm = right_log_diff / 32.0
        else:
            right_to_mult_ratio = 0.0
            right_log_diff_norm = 0.0
        
        if left_mult > 0 and op in [0, 1]:
            shift_amp = (1 << shift)
            shift_amp_norm = min(1.0, math.log2(shift_amp + 1) / 5.0)
        else:
            shift_amp_norm = 0.0
        
        if right_mult > 0 and op == 2:
            shift_amp = (1 << shift)
            shift_right_norm = min(1.0, math.log2(shift_amp + 1) / 5.0)
        else:
            shift_right_norm = 0.0
        
        if left_mult > 0 and right_mult > 0:
            if op == 0:
                reconstructed = (left_mult << shift) + right_mult
            elif op == 1:
                reconstructed = (left_mult << shift) - right_mult
            elif op == 2:
                reconstructed = left_mult - (right_mult << shift)
            else:
                reconstructed = mult
            
            reconstruction_error = abs(reconstructed - mult) / max(mult, 1)
            reconstruction_error = min(1.0, reconstruction_error)
            reconstruction_correct = float(reconstructed == mult)
        else:
            reconstruction_error = 0.0
            reconstruction_correct = 0.0
        
        if mult > 0:
            if left_mult > 0:
                if op in [0, 1]:
                    left_contrib = (left_mult << shift) / mult
                else:
                    left_contrib = left_mult / mult
                left_contrib = min(2.0, left_contrib)
            else:
                left_contrib = 0.0
            
            if right_mult > 0:
                if op == 2:
                    right_contrib = (right_mult << shift) / mult
                else:
                    right_contrib = right_mult / mult
                right_contrib = min(2.0, right_contrib)
            else:
                right_contrib = 0.0
        else:
            left_contrib = 0.0
            right_contrib = 0.0
        
        operand_relation = float(left_mult < right_mult) if (left_mult > 0 and right_mult > 0) else 0.0
        has_two_operands = float(left_mult > 0 and right_mult > 0)
        
        return [
            left_to_mult_ratio, right_to_mult_ratio,
            left_log_diff_norm, right_log_diff_norm,
            shift_amp_norm, shift_right_norm,
            reconstruction_error, reconstruction_correct,
            left_contrib, right_contrib,
            operand_relation, has_two_operands
        ]

    def compute_shifted_operand_features(self, eq, equations, nodes_in_graph=None):
        """Compact features for shifted operands (44-dim = 22+22)"""
        op = eq.get("op", 3)
        assert 0 <= op <= 3, f"Invalid op={op}"
        
        shift = eq.get("shift", 0)
        assert 0 <= shift <= 31, f"Invalid shift={shift}"
        
        left_idx = eq.get("left", -1)
        assert 0 <= left_idx <= 9, f"Invalid left index={left_idx}"
        
        right_idx = eq.get("right", -1)
        assert 0 <= right_idx <= 9, f"Invalid right index={right_idx}"
        
        mult = eq["mult"]
        
        # Only access nodes in graph
        if nodes_in_graph is not None:
            left_mult = equations[left_idx]["mult"] if (left_idx >= 0 and left_idx in nodes_in_graph) else 0
            right_mult = equations[right_idx]["mult"] if (right_idx >= 0 and right_idx in nodes_in_graph) else 0
        else:
            left_mult = equations[left_idx]["mult"] if 0 <= left_idx < len(equations) else 0
            right_mult = equations[right_idx]["mult"] if 0 <= right_idx < len(equations) else 0
        
        if op in [0, 1]:
            if left_mult > 0:
                shifted_left = left_mult << shift
                left_features = self.compute_compact_number_features(shifted_left, mult)
            else:
                left_features = [0.0] * 22
            
            if right_mult > 0:
                right_features = self.compute_compact_number_features(right_mult, mult)
            else:
                right_features = [0.0] * 22
        elif op == 2:
            if left_mult > 0:
                left_features = self.compute_compact_number_features(left_mult, mult)
            else:
                left_features = [0.0] * 22
            
            if right_mult > 0:
                shifted_right = right_mult << shift
                right_features = self.compute_compact_number_features(shifted_right, mult)
            else:
                right_features = [0.0] * 22
        else:
            left_features = [0.0] * 22
            right_features = [0.0] * 22
        
        return left_features + right_features

    def compute_bit_distance_features(self, mult, target):
        """Bit-level distance to target (5-dim)"""
        xor_result = mult ^ target
        bit_diff_ratio = bin(xor_result).count('1') / 32.0
        
        and_result = mult & target
        common_ones_ratio = bin(and_result).count('1') / 32.0
        
        or_result = mult | target
        coverage_ratio = bin(or_result).count('1') / 32.0
        
        need_to_set = target & (~mult)
        set_ratio = bin(need_to_set).count('1') / 32.0
        
        need_to_clear = mult & (~target)
        clear_ratio = bin(need_to_clear).count('1') / 32.0
        
        return [bit_diff_ratio, common_ones_ratio, coverage_ratio, set_ratio, clear_ratio]
    
    def compute_gap_features(self, mult, target):
        """Gap analysis features (9-dim)"""
        gap = abs(target - mult)
        if target == 0:
            return [0.0] * 9
        
        gap_linear = min(10.0, gap / target)
        gap_log = math.log2(gap + 1) / math.log2(target + 1) if target > 0 else 0.0
        gap_sqrt = min(10.0, math.sqrt(gap) / math.sqrt(target)) if target > 0 else 0.0
        
        is_power_of_2 = float((gap & (gap - 1)) == 0 and gap > 0)
        gap_shift_hint = math.log2(gap) / math.log2(self.MAX_SHIFT) if is_power_of_2 and gap > 0 else 0.0
        
        gap_leading_zeros = (32 - gap.bit_length()) if gap > 0 else 32
        gap_leading_zeros_norm = gap_leading_zeros / 32.0
        
        gap_complexity = bin(gap).count('1') / 32.0
        is_below = float(mult < target)
        is_above = float(mult > target)
        
        result = [gap_linear, gap_log, gap_sqrt, is_power_of_2, gap_shift_hint,
                  gap_leading_zeros_norm, gap_complexity, is_below, is_above]
        return [x if math.isfinite(x) else 0.0 for x in result]
    
    def compute_theoretical_shifts(self, mult, target):
        """Theoretical shift hints (3-dim)"""
        if mult > 1:
            ratio = target / mult if target > mult else mult / max(target, 1)
            shift_hint_decompose = math.log2(max(ratio, 1))
            shift_hint_decompose = min(1.0, shift_hint_decompose / math.log2(self.MAX_SHIFT))
        else:
            shift_hint_decompose = 0.0
        
        shift_hint_bitlen = min(1.0, mult.bit_length() / 32.0) if mult > 0 else 0.0
        shift_hint_avg = (shift_hint_decompose + shift_hint_bitlen) / 2.0
        
        return [shift_hint_decompose, shift_hint_bitlen, shift_hint_avg]
    
    def compute_pairwise_potential_stats(self, graph_idx, sorted_nodes, equations, target):
        """Pairwise combination potential (8-dim)"""
        eq_idx = sorted_nodes[graph_idx]
        mult_i = equations[eq_idx]["mult"]
        
        if target == 0 or mult_i == 0:
            return [0.0] * 8
        
        all_errors = []
        exact_count = 0
        close_count = 0
        
        for j in range(graph_idx):
            other_eq_idx = sorted_nodes[j]
            mult_j = equations[other_eq_idx]["mult"]
            
            for op in [0, 1, 2]:
                for shift in range(1, min(17, self.MAX_SHIFT + 1)):
                    try:
                        if op == 0:
                            result = (mult_j << shift) + mult_i
                        elif op == 1:
                            result = (mult_j << shift) - mult_i
                        elif op == 2:
                            result = mult_j - (mult_i << shift)
                        else:
                            continue
                        
                        if 0 < result < 1000000:
                            error = abs(result - target)
                            all_errors.append(min(error, 100000))
                            if error == 0:
                                exact_count += 1
                            if error / target < 0.01:
                                close_count += 1
                    except:
                        continue
        
        if all_errors:
            min_error = min(10.0, min(all_errors) / target)
            avg_error = min(10.0, sum(all_errors) / len(all_errors) / target)
        else:
            min_error = 10.0
            avg_error = 10.0
        
        total_comb = max(1, graph_idx * 3 * 16)
        exact_ratio = min(1.0, exact_count / total_comb)
        close_ratio = min(1.0, close_count / total_comb)
        
        result = [
            min_error, avg_error, exact_ratio, close_ratio,
            float(exact_count > 0), float(close_count > 0),
            math.log2(exact_count + 1) / 5.0,
            math.log2(close_count + 1) / 5.0
        ]
        return [x if math.isfinite(x) else 0.0 for x in result]
    
    def compute_positional_features(self, node_idx, prefix_len):
        """Positional encoding (6-dim)"""
        abs_pos = node_idx / max(prefix_len - 1, 1)
        dist_from_start = abs_pos
        dist_from_end = 1.0 - abs_pos
        
        pe_sin = math.sin(node_idx * math.pi / 8)
        pe_cos = math.cos(node_idx * math.pi / 8)
        is_first = float(node_idx == 0)
        
        return [abs_pos, dist_from_start, dist_from_end, pe_sin, pe_cos, is_first]
    
    def compute_topdown_specific_features(self, mult, equations, eq_idx, sorted_nodes):
        """Top-Down decomposition specific features (5-dim)"""
        if eq_idx == 0:
            decomp_progress = 0.0
        else:
            first_mult = equations[0]["mult"]
            if first_mult > 0:
                progress = math.log2(first_mult + 1) - math.log2(mult + 1)
                decomp_progress = min(1.0, max(0.0, progress / math.log2(first_mult + 1)))
            else:
                decomp_progress = 0.0
        
        if eq_idx > 0:
            prev_mult = equations[eq_idx - 1]["mult"]
            decay_ratio = mult / max(prev_mult, 1)
        else:
            decay_ratio = 1.0
        
        steps_norm = min(1.0, math.log2(mult + 1) / 10.0) if mult > 1 else 0.0
        decomposability = bin(mult).count('1') / 32.0
        near_base = float(mult <= 3)
        
        return [decomp_progress, decay_ratio, steps_norm, decomposability, near_base]

    # ========================================
    # NEW FEATURES FROM SECOND FILE
    # ========================================
    
    def compute_factor_quality_features(self, value, target):
        """
        Factor quality features based on Picat decomposition patterns (8-dim)
        
        Features:
        1. Bit length normalized
        2. Low bits pattern (presence of 1s in low bits)
        3. High bits pattern (presence of 1s in high bits)
        4. All-ones pattern (C = 2^k - 1)
        5. Low zeros pattern (presence of 0s in low bits)
        6. Near power-of-2 score
        7. Symmetric split potential
        8. Complement pattern quality
        """
        if value <= 0:
            return [0.0] * 8
        
        # 1. Bit length feature
        bit_length = value.bit_length()
        bit_length_norm = bit_length / 32.0
        
        # 2. Low bits pattern (split_pp finds low 1s)
        low_bits_pattern = 0.0
        for i in range(min(4, bit_length)):
            if (value >> i) & 1:
                low_bits_pattern = 1.0
                break
        
        # 3. High bits pattern
        high_bits = value >> max(0, bit_length - 4)
        has_high_ones = float(high_bits > 0)
        
        # 4. All-ones pattern (C = 2^k - 1)
        is_all_ones = 0.0
        if value > 0 and (value & (value + 1)) == 0:
            is_all_ones = 1.0
        
        # 5. Low zeros pattern (split_pn finds low 0s)
        low_zeros_pattern = 0.0
        for i in range(min(4, bit_length)):
            if (value >> i) & 1 == 0:
                low_zeros_pattern = 1.0
                break
        
        # 6. Near power-of-2 (split_np uses Left0)
        nearest_pow2 = 1 << (bit_length - 1)
        pow2_distance = abs(value - nearest_pow2) / max(nearest_pow2, 1)
        near_pow2 = math.exp(-pow2_distance * 5)
        
        # 7. Symmetric pattern (split_pp symmetric splits)
        symmetric_score = 0.0
        for s in range(1, min(6, bit_length//2 + 1)):
            left = value >> s
            right = value & ((1 << s) - 1)
            if left > 0 and right > 0:
                similarity = 1.0 - abs(left - right) / max(left, right)
                symmetric_score = max(symmetric_score, similarity)
        
        # 8. Complement pattern (split_pn and split_np complement operations)
        complement_score = 0.0
        N = bit_length
        full_mask = (1 << N) - 1
        complement = full_mask ^ value
        if complement > 0:
            complement_bits = complement.bit_length()
            complement_score = 1.0 - (complement_bits / N) if N > 0 else 0.0
        
        return [
            bit_length_norm,
            low_bits_pattern,
            has_high_ones,
            is_all_ones,
            low_zeros_pattern,
            near_pow2,
            symmetric_score,
            complement_score
        ]
    
    def compute_decomposition_pattern_features(self, curr_mult, target, available_mults, equations, k):
        """
        Decomposition pattern features based on Picat algorithm (21-dim)
        
        Analyzes potential for three split strategies:
        - split_pp: C = Left*(1<<S) + Right
        - split_pn: C = Left*(1<<S) - Right  
        - split_np: C = Left - Right*(1<<S)
        
        Features breakdown:
        [0-2]: Split potential scores (pp, pn, np)
        [3-5]: Best shift values for each strategy
        [6-8]: Special pattern detection
        [9-11]: Bit pattern features
        [12-14]: Target relationship
        [15]: Decomposition progress
        [16]: Reuse potential
        [17-20]: Miscellaneous (parity, density, size, combined)
        """
        features = [0.0] * 21
        C = curr_mult
        
        if C <= 0:
            return features
        
        N = C.bit_length()
        
        # ===== 1. split_pp potential (C = Left*(1<<S) + Right) =====
        pp_potential = 0.0
        best_pp_shift = 0.0
        
        for pos in range(N):
            if (C >> pos) & 1:  # Find 1
                right = C & ((1 << pos) - 1)
                left = C >> pos
                
                # Remove trailing zeros
                while left > 0 and (left & 1) == 0:
                    left >>= 1
                
                if left > 0 and left != C and right > 0:
                    left_quality = 1.0 - (left.bit_length() / max(N, 1))
                    right_quality = 1.0 if right > 1 else 0.5
                    score = (left_quality + right_quality) / 2.0
                    
                    if score > pp_potential:
                        pp_potential = score
                        best_pp_shift = pos / 32.0
        
        # ===== 2. split_pn potential (C = Left*(1<<S) - Right) =====
        pn_potential = 0.0
        best_pn_shift = 0.0
        
        for pos in range(N + 1):
            if pos == N or (C >> pos) & 1 == 0:  # Find 0 or exceed
                left = (C >> pos) + 1 if pos < N else 1
                
                # Calculate complement part
                right = 0
                for i in range(pos):
                    bit_i = (C >> i) & 1
                    if bit_i == 0:
                        right += (1 << i)
                
                if left > 0 and left != C and right >= 0:
                    score = 1.0 - (left.bit_length() / max(N + 1, 1))
                    if score > pn_potential:
                        pn_potential = score
                        best_pn_shift = pos / 32.0
        
        # ===== 3. split_np potential (C = Left - Right*(1<<S)) =====
        np_potential = 0.0
        best_np_shift = 0.0
        
        Left0 = 1 << N
        
        for pos in range(N):
            if (C >> pos) & 1 == 0:  # Find 0
                left = Left0 - (1 << pos)
                
                # Add low bits that are 1
                for i in range(pos):
                    if (C >> i) & 1:
                        left += (1 << i)
                
                right = (1 << N) - 1 - C
                
                if left > 0 and left != C and right > 0:
                    score = 1.0 - ((left - C) / max(left, 1))
                    if score > np_potential:
                        np_potential = score
                        best_np_shift = pos / 32.0
        
        features[0] = pp_potential
        features[1] = pn_potential
        features[2] = np_potential
        features[3] = best_pp_shift
        features[4] = best_pn_shift
        features[5] = best_np_shift
        
        # ===== 4. Special pattern detection =====
        is_all_ones = float(C > 0 and (C & (C + 1)) == 0)
        is_pow2_plus_one = float(C > 2 and (C - 1) & (C - 2) == 0)
        
        nearest_pow2 = 1 << (N - 1)
        near_pow2_score = 1.0 - abs(C - nearest_pow2) / max(nearest_pow2, 1)
        
        features[6] = is_all_ones
        features[7] = is_pow2_plus_one
        features[8] = near_pow2_score
        
        # ===== 5. Bit pattern features =====
        high_bits = C >> max(0, N - 4) if N >= 4 else C
        high_pattern = high_bits / 15.0
        
        low_bits = C & 0xF
        low_pattern = low_bits / 15.0
        
        middle_sparsity = 0.0
        if N > 6:
            middle_mask = ((1 << (N - 4)) - 1) << 2
            middle_bits = (C & middle_mask) >> 2
            middle_sparsity = 1.0 - bin(middle_bits).count('1') / max((N - 6), 1)
        
        features[9] = high_pattern
        features[10] = low_pattern
        features[11] = middle_sparsity
        
        # ===== 6. Target relationship =====
        if target > 0:
            xor_diff = bin(C ^ target).count('1') / 32.0
            and_similarity = bin(C & target).count('1') / 32.0
            
            shift_potential = 0.0
            for s in range(1, 17):
                shifted = C << s
                if shifted > 0:
                    diff = abs(shifted - target) / max(target, 1)
                    if diff < 0.2:
                        shift_potential = max(shift_potential, 1.0 - diff * 5)
                        break
            
            features[12] = xor_diff
            features[13] = and_similarity
            features[14] = shift_potential
        else:
            features[12] = 0.0
            features[13] = 0.0
            features[14] = 0.0
        
        # ===== 7. Decomposition progress =====
        if k > 0 and target > 0 and len(equations) > 0:
            first_mult = equations[0]["mult"]
            if first_mult > 0:
                start_log = math.log2(first_mult + 1)
                curr_log = math.log2(C + 1)
                target_log = math.log2(target + 1)
                
                if start_log > target_log:
                    progress = (start_log - curr_log) / (start_log - target_log)
                    features[15] = min(1.0, max(0.0, progress))
                else:
                    features[15] = 0.0
            else:
                features[15] = 0.0
        else:
            features[15] = 0.0
        
        # ===== 8. Historical reuse potential =====
        reuse_potential = 0.0
        if available_mults:
            for mult in available_mults:
                if mult != C:
                    for s in range(1, min(9, self.MAX_SHIFT)):
                        left_candidate = mult
                        remaining_plus = C - (left_candidate << s)
                        remaining_minus = (left_candidate << s) - C
                        
                        if remaining_plus > 0 and remaining_plus in available_mults:
                            reuse_potential = max(reuse_potential, 0.8)
                        if remaining_minus > 0 and remaining_minus in available_mults:
                            reuse_potential = max(reuse_potential, 0.7)
        
        features[16] = reuse_potential
        
        # ===== 9. Other features =====
        features[17] = float(C % 2 == 1)
        features[18] = float(bin(C).count('1') / max(N, 1))
        features[19] = float(N / 32.0)
        features[20] = (pp_potential + pn_potential + np_potential) / 3.0
        
        return features
    
    def _detect_reuse_pattern(self, equations, step_idx, eq_idx_to_graph_idx):
        """
        Detect reuse pattern (returns integer label)
        
        Returns:
        0: No reuse
        1: Reuse left operand only
        2: Reuse right operand only
        3: Symmetric reuse (same value)
        4: Chain reuse (previous step)
        5: Reuse two different values
        """
        eq = equations[step_idx]
        left_idx = eq.get("left", -1)
        right_idx = eq.get("right", -1)
        
        left_is_reuse = left_idx >= 0 and left_idx in eq_idx_to_graph_idx
        right_is_reuse = right_idx >= 0 and right_idx in eq_idx_to_graph_idx
        
        if not left_is_reuse and not right_is_reuse:
            return 0
        
        elif left_is_reuse and not right_is_reuse:
            return 1
        
        elif not left_is_reuse and right_is_reuse:
            return 2
        
        elif left_is_reuse and right_is_reuse:
            if equations[left_idx]["mult"] == equations[right_idx]["mult"]:
                return 3
            else:
                if left_idx == step_idx - 1 or right_idx == step_idx - 1:
                    return 4
                else:
                    return 5
        
        return 0

    def compute_edge_features(self, src_eq_idx, dst_eq_idx, eq, equations, 
                             is_next_step=False, is_self_loop=False, is_both=False):
        """Edge features (12-dim)"""
        if is_next_step or is_self_loop:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    float(not is_self_loop), float(is_next_step), float(is_self_loop)]
        
        op = eq.get("op", 3)
        shift = eq.get("shift", 1)
        left_idx = eq.get("left", -1)
        right_idx = eq.get("right", -1)
        
        if is_both:
            is_left = 1.0
            is_right = 1.0
        else:
            is_left = float(dst_eq_idx == left_idx)
            is_right = float(dst_eq_idx == right_idx)
        
        abs_distance_norm = abs(dst_eq_idx - src_eq_idx) / max(len(equations) - 1, 1)
        rel_distance = (dst_eq_idx - src_eq_idx) / max(abs(dst_eq_idx) + 1, 1)
        
        shift_norm = shift / self.MAX_SHIFT
        is_shift_pow2 = float(shift > 0 and (shift & (shift - 1)) == 0)
        shift_effective = 1.0 if (((op in [0, 1]) and is_left > 0.5) or (op == 2 and is_right > 0.5)) else 0.0
        
        src_mult = equations[src_eq_idx]["mult"] if 0 <= src_eq_idx < len(equations) else 0
        dst_mult = equations[dst_eq_idx]["mult"] if 0 <= dst_eq_idx < len(equations) else 0
        
        if src_mult > 0 and dst_mult > 0:
            mult_diff_norm = (math.log2(src_mult + 1) - math.log2(dst_mult + 1)) / 32.0
            mult_ratio = min(1.0, dst_mult / max(src_mult, 1))
        else:
            mult_diff_norm = 0.0
            mult_ratio = 0.0
        
        return [
            is_left, is_right,
            abs_distance_norm, rel_distance,
            shift_norm, is_shift_pow2, shift_effective,
            mult_diff_norm, mult_ratio,
            1.0, 0.0, 0.0
        ]
    
    def compute_target_features(self, target):
        """Target compact representation (22-dim)"""
        return self.compute_compact_number_features(target, reference=None)
    
    def compute_special_pattern_features(self, value):
        """Special patterns recognized by Picat solver (6-dim)"""
        if value <= 0:
            return [0.0] * 6
        
        is_identity = float(value == 1)
        
        is_power_of_2 = float((value & (value - 1)) == 0)
        if is_power_of_2:
            power_of_2_exponent = (value.bit_length() - 1) / 32.0
        else:
            power_of_2_exponent = 0.0
        
        is_all_ones = float((value & (value + 1)) == 0)
        if is_all_ones:
            all_ones_length = value.bit_length() / 32.0
        else:
            all_ones_length = 0.0
        
        is_pow2_plus_1 = False
        if value > 1:
            candidate = value - 1
            is_pow2_plus_1 = (candidate & (candidate - 1)) == 0
        is_pow2_plus_1 = float(is_pow2_plus_1)
        
        is_pow2_minus_1 = False
        if value > 0:
            candidate = value + 1
            is_pow2_minus_1 = (candidate & (candidate - 1)) == 0
        is_pow2_minus_1 = float(is_pow2_minus_1)
        
        return [
            is_identity,
            is_power_of_2,
            power_of_2_exponent,
            is_all_ones,
            is_pow2_plus_1,
            is_pow2_minus_1
        ]

    # ========================================
    # Main Graph Building with ALL Features
    # ========================================
    
    def _build_graph(self, equations, target, k, dfs_order):
        """
        Build a single graph with ALL FEATURES INTEGRATED (199-dim nodes)
        
        Feature breakdown:
        1. mult: 22-dim
        2. shifted_operand: 44-dim
        3. op: 4-dim
        4. dependency: 6-dim
        5. tree_level: 8-dim
        6. operand_relationship: 12-dim
        7. bit_distance: 5-dim
        8. gap: 9-dim
        9. theoretical_shifts: 3-dim
        10. pairwise_potential: 8-dim
        11. positional: 6-dim
        12. topdown_specific: 5-dim
        13. factor_quality: 8-dim (NEW)
        14. decomposition_pattern: 21-dim (NEW)
        15. bit: 32-dim
        16. special_pattern: 6-dim
        
        Total: 199-dim
        """

        nodes_in_graph = set()
        decomposed_nodes = set()
        
        # k=0: only target node
        if k == 0:
            nodes_in_graph.add(0)
        
        # k>0: use DFS order
        if k > 0:
            for i in range(min(k, len(dfs_order))):
                idx = dfs_order[i]
                eq = equations[idx]
                
                nodes_in_graph.add(idx)
                decomposed_nodes.add(idx)
                
                left_idx = eq.get("left", -1)
                right_idx = eq.get("right", -1)
                
                if left_idx >= 0:
                    nodes_in_graph.add(left_idx)
                if right_idx >= 0:
                    nodes_in_graph.add(right_idx)
        
        undecomposed_nodes = nodes_in_graph - decomposed_nodes
        
        sorted_nodes = sorted(list(nodes_in_graph))
        eq_idx_to_graph_idx = {eq_idx: graph_idx for graph_idx, eq_idx in enumerate(sorted_nodes)}
        num_graph_nodes = len(sorted_nodes)
        
        # Collect available mults for decomposition pattern features
        available_mults = set()
        for eq_idx in nodes_in_graph:
            if eq_idx < len(equations):
                available_mults.add(equations[eq_idx]["mult"])
        
        # Build node features with ALL 199 dimensions
        x_list = []
        
        if k == 0:
            # Target node at k=0 (simplified features)
            eq = equations[0]
            mult = eq["mult"]
            
            # Basic features
            mult_feat = self.compute_compact_number_features(mult, target)  # 22
            bit_feat = self.compute_bit_features(mult)  # 32
            special_feat = self.compute_special_pattern_features(mult)  # 6
            
            # NEW: Add factor quality and decomposition pattern features
            factor_quality_feat = self.compute_factor_quality_features(mult, target)  # 8
            decomp_pattern_feat = self.compute_decomposition_pattern_features(
                mult, target, available_mults, equations, k
            )  # 21
            
            # Padding for other features
            zero_padding = [0.0] * (199 - 22 - 32 - 6 - 8 - 21)  # 110
            
            # Combine: 22 + 110 + 8 + 21 + 32 + 6 = 199
            all_feat = mult_feat + zero_padding + factor_quality_feat + decomp_pattern_feat + bit_feat + special_feat
            all_feat = [x if math.isfinite(x) else 0.0 for x in all_feat]
            x_list.append(torch.tensor(all_feat, dtype=torch.float32))
        else:
            nodes_set = set(sorted_nodes)
            
            for graph_idx, eq_idx in enumerate(sorted_nodes):
                eq = equations[eq_idx]
                mult = eq["mult"]
                op = eq.get("op", 3)
                
                # 1. mult features: 22-dim
                mult_feat = self.compute_compact_number_features(mult, target)
                
                # 2. shifted_operand features: 44-dim
                shifted_feat = self.compute_shifted_operand_features(eq, equations, nodes_set)
                
                # 3. op features: 4-dim
                op_feat = self.one_hot(op, self.MAX_OP)
                
                # 4. dependency features: 6-dim
                left_idx = eq.get("left", -1)
                right_idx = eq.get("right", -1)
                left_graph_idx = eq_idx_to_graph_idx.get(left_idx, -1)
                right_graph_idx = eq_idx_to_graph_idx.get(right_idx, -1)
                dep_feat = self.compute_dependency_features(graph_idx, left_graph_idx, right_graph_idx, num_graph_nodes)
                
                # 5. tree_level features: 8-dim
                tree_feat = self.compute_tree_level_features(eq_idx, eq, equations, sorted_nodes, eq_idx_to_graph_idx)
                
                # 6. operand_relationship features: 12-dim
                operand_feat = self.compute_operand_relationship_features(eq, equations, nodes_set)
                
                # 7. bit_distance features: 5-dim
                bit_dist_feat = self.compute_bit_distance_features(mult, target)
                
                # 8. gap features: 9-dim
                gap_feat = self.compute_gap_features(mult, target)
                
                # 9. theoretical_shifts features: 3-dim
                shift_feat = self.compute_theoretical_shifts(mult, target)
                
                # 10. pairwise_potential features: 8-dim
                pair_feat = self.compute_pairwise_potential_stats(graph_idx, sorted_nodes, equations, target)
                
                # 11. positional features: 6-dim
                pos_feat = self.compute_positional_features(graph_idx, num_graph_nodes)
                
                # 12. topdown_specific features: 5-dim
                topdown_feat = self.compute_topdown_specific_features(mult, equations, eq_idx, sorted_nodes)
                
                # 13. factor_quality features: 8-dim (NEW)
                factor_quality_feat = self.compute_factor_quality_features(mult, target)
                
                # 14. decomposition_pattern features: 21-dim (NEW)
                # Only compute for the current decomposition step
                if eq_idx == k:
                    decomp_pattern_feat = self.compute_decomposition_pattern_features(
                        mult, target, available_mults, equations, k
                    )
                else:
                    decomp_pattern_feat = [0.0] * 21
                
                # 15. bit features: 32-dim
                mult_bit_feat = self.compute_bit_features(mult)
                
                # 16. special_pattern features: 6-dim
                special_pattern_feat = self.compute_special_pattern_features(mult)
                
                # Combine all features: 22+44+4+6+8+12+5+9+3+8+6+5+8+21+32+6 = 199
                all_feat = (
                    mult_feat +              # 22
                    shifted_feat +           # 44
                    op_feat +                # 4
                    dep_feat +               # 6
                    tree_feat +              # 8
                    operand_feat +           # 12
                    bit_dist_feat +          # 5
                    gap_feat +               # 9
                    shift_feat +             # 3
                    pair_feat +              # 8
                    pos_feat +               # 6
                    topdown_feat +           # 5
                    factor_quality_feat +    # 8 (NEW)
                    decomp_pattern_feat +    # 21 (NEW)
                    mult_bit_feat +          # 32
                    special_pattern_feat     # 6
                )
                
                # Verify dimension
                assert len(all_feat) == 199, f"Feature dimension mismatch: expected 199, got {len(all_feat)}"
                
                all_feat = [x if math.isfinite(x) else 0.0 for x in all_feat]
                x_list.append(torch.tensor(all_feat, dtype=torch.float32))

            # Prediction node (all zeros)
            x_list.append(torch.zeros(199, dtype=torch.float32))
        
        x = torch.stack(x_list, dim=0)

        # Build edges (same as before)
        edge_src, edge_dst, edge_attr_list = [], [], []
        existing_edges = {}

        if k > 0:
            for i in range(k):
                if i not in eq_idx_to_graph_idx:
                    continue
                    
                eq = equations[i]
                left_idx = eq.get("left", -1)
                right_idx = eq.get("right", -1)
                
                src_graph_idx = eq_idx_to_graph_idx[i]
                
                dst_left_graph_idx = eq_idx_to_graph_idx.get(left_idx, -1) if left_idx >= 0 else -1
                dst_right_graph_idx = eq_idx_to_graph_idx.get(right_idx, -1) if right_idx >= 0 else -1
                
                if dst_left_graph_idx >= 0 and dst_left_graph_idx == dst_right_graph_idx:
                    edge_key = (src_graph_idx, dst_left_graph_idx)
                    if edge_key not in existing_edges:
                        edge_src.append(src_graph_idx)
                        edge_dst.append(dst_left_graph_idx)
                        edge_feat = self.compute_edge_features(
                            i, left_idx, eq, equations, 
                            is_next_step=False, is_self_loop=False, is_both=True
                        )
                        edge_attr_list.append(edge_feat)
                        existing_edges[edge_key] = len(edge_attr_list) - 1
                else:
                    if dst_left_graph_idx >= 0:
                        edge_key = (src_graph_idx, dst_left_graph_idx)
                        if edge_key not in existing_edges:
                            edge_src.append(src_graph_idx)
                            edge_dst.append(dst_left_graph_idx)
                            edge_feat = self.compute_edge_features(
                                i, left_idx, eq, equations,
                                is_next_step=False, is_self_loop=False, is_both=False
                            )
                            edge_attr_list.append(edge_feat)
                            existing_edges[edge_key] = len(edge_attr_list) - 1
                    
                    if dst_right_graph_idx >= 0:
                        edge_key = (src_graph_idx, dst_right_graph_idx)
                        if edge_key not in existing_edges:
                            edge_src.append(src_graph_idx)
                            edge_dst.append(dst_right_graph_idx)
                            edge_feat = self.compute_edge_features(
                                i, right_idx, eq, equations,
                                is_next_step=False, is_self_loop=False, is_both=False
                            )
                            edge_attr_list.append(edge_feat)
                            existing_edges[edge_key] = len(edge_attr_list) - 1

        # Prediction edges
        pred_node_idx = num_graph_nodes
        if k == 0:
            edge_src.append(0)
            edge_dst.append(0)
            edge_feat = self.compute_edge_features(
                -1, -1, {}, equations,
                is_next_step=False, is_self_loop=True, is_both=False
            )
            edge_attr_list.append(edge_feat)
        else:
            for eq_idx in undecomposed_nodes:
                if eq_idx in eq_idx_to_graph_idx:
                    graph_idx = eq_idx_to_graph_idx[eq_idx]
                    edge_src.append(graph_idx)
                    edge_dst.append(pred_node_idx)
                    edge_feat = self.compute_edge_features(
                        -1, -1, {}, equations,
                        is_next_step=True, is_self_loop=False, is_both=False
                    )
                    edge_attr_list.append(edge_feat)

            num_total_nodes = num_graph_nodes + 1
            in_deg = [0] * num_total_nodes
            for d in edge_dst:
                if d < num_total_nodes:
                    in_deg[d] += 1
            
            for nid in range(num_total_nodes):
                if in_deg[nid] == 0:
                    edge_src.append(nid)
                    edge_dst.append(nid)
                    edge_feat = self.compute_edge_features(
                        -1, -1, {}, equations,
                        is_next_step=False, is_self_loop=True, is_both=False
                    )
                    edge_attr_list.append(edge_feat)

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

        # Labels
        next_step = equations[k]
        y_shift = max(1, min(self.MAX_SHIFT, next_step.get("shift", 0))) - 1
        y_op = next_step.get("op", 3)

        assert 0 <= y_op <= 3, f"Invalid y_op={y_op}"
        
        y_left_eq_idx = next_step.get("left", -1)
        y_right_eq_idx = next_step.get("right", -1)
        
        # Get mult values
        if y_left_eq_idx >= 0 and y_left_eq_idx < len(equations):
            y_left_mult = equations[y_left_eq_idx]["mult"]
        else:
            y_left_mult = 1
        
        if y_right_eq_idx >= 0 and y_right_eq_idx < len(equations):
            y_right_mult = equations[y_right_eq_idx]["mult"]
        else:
            y_right_mult = 1
        
        # Node selection
        y_left = eq_idx_to_graph_idx.get(y_left_eq_idx, -1)
        y_right = eq_idx_to_graph_idx.get(y_right_eq_idx, -1)
        
        # Reuse indicators
        left_is_reuse = float(y_left >= 0)
        right_is_reuse = float(y_right >= 0)
        
        # NEW: Reuse pattern detection
        reuse_pattern = self._detect_reuse_pattern(equations, k, eq_idx_to_graph_idx)
        
        curr_mult_value = next_step.get("mult", target if k == 0 else 1)
        
        # NEW: Factor quality features for current mult
        curr_mult_quality_features = self.compute_factor_quality_features(curr_mult_value, target)

        target_feat = self.compute_target_features(target)
        target_log_norm = math.log2(target + 1) / 32.0

        # Available mask
        FIXED_MAX_NODES = 11
        num_available = num_graph_nodes
        
        available_mask = torch.zeros(FIXED_MAX_NODES, dtype=torch.bool)
        if num_available > 0:
            available_mask[:min(num_available, FIXED_MAX_NODES)] = True

        data = Data(
            x=x,  # 199-dim node features
            edge_index=edge_index,
            edge_attr=edge_attr,
            
            # Core prediction targets
            y_shift=torch.tensor([y_shift], dtype=torch.long),
            y_op=torch.tensor([y_op], dtype=torch.long),
            y_left=torch.tensor([y_left], dtype=torch.long),
            y_right=torch.tensor([y_right], dtype=torch.long),
            
            # Mult values
            y_left_mult=torch.tensor([y_left_mult], dtype=torch.long),
            y_right_mult=torch.tensor([y_right_mult], dtype=torch.long),
            
            # Reuse indicators
            left_is_reuse=torch.tensor([left_is_reuse], dtype=torch.float32),
            right_is_reuse=torch.tensor([right_is_reuse], dtype=torch.float32),
            reuse_pattern=torch.tensor([reuse_pattern], dtype=torch.long),  # NEW
            
            # Current mult features
            curr_mult_value=torch.tensor([curr_mult_value], dtype=torch.long),
            curr_mult_quality_features=torch.tensor([curr_mult_quality_features], dtype=torch.float32),  # NEW (8-dim)
            
            # Target features
            target_features=torch.tensor([target_feat], dtype=torch.float32),
            target_normalized=torch.tensor([target_log_norm], dtype=torch.float32),
            
            # Other
            available_indices=available_mask,
            prefix_len=torch.tensor([k], dtype=torch.long),
            raw_target=torch.tensor([target], dtype=torch.long),
        )

        return data

    def get_node_feature_dim(self):
        """Node feature dimension: 199"""
        return 199

    def get_edge_feature_dim(self):
        """Edge feature dimension: 12"""
        return 12

    def get_target_feature_dim(self):
        """Target feature dimension: 22"""
        return 22