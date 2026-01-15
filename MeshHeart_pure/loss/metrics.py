import torch
import numpy as np
from pytorch3d.ops import knn_points
from typing import Tuple, List
import time

def compute_hausdorff_distance(pred_verts: torch.Tensor, gt_verts: torch.Tensor) -> torch.Tensor:
    """
    Compute Hausdorff distance between two point sets.
    
    Args:
        pred_verts: Predicted vertices [N1, 3]
        gt_verts: Ground truth vertices [N2, 3]
    
    Returns:
        Hausdorff distance (scalar tensor)
    """
    # Ensure tensors are on the same device and have batch dimension
    if pred_verts.dim() == 2:
        pred_verts = pred_verts.unsqueeze(0)  # [1, N1, 3]
    if gt_verts.dim() == 2:
        gt_verts = gt_verts.unsqueeze(0)  # [1, N2, 3]
    
    # Compute distances from pred to gt
    knn_pred_to_gt = knn_points(pred_verts, gt_verts, K=1)
    dist_pred_to_gt = knn_pred_to_gt.dists.sqrt()  # [1, N1, 1]
    max_dist_pred_to_gt = dist_pred_to_gt.max()
    
    # Compute distances from gt to pred
    knn_gt_to_pred = knn_points(gt_verts, pred_verts, K=1)
    dist_gt_to_pred = knn_gt_to_pred.dists.sqrt()  # [1, N2, 1]
    max_dist_gt_to_pred = dist_gt_to_pred.max()
    
    # Hausdorff distance is the maximum of both directions
    hausdorff_dist = torch.max(max_dist_pred_to_gt, max_dist_gt_to_pred)
    
    return hausdorff_dist

def compute_assd(pred_verts: torch.Tensor, gt_verts: torch.Tensor) -> torch.Tensor:
    """
    Compute Average Symmetric Surface Distance (ASSD) between two point sets.
    
    Args:
        pred_verts: Predicted vertices [N1, 3]
        gt_verts: Ground truth vertices [N2, 3]
    
    Returns:
        ASSD (scalar tensor)
    """
    # Ensure tensors are on the same device and have batch dimension
    if pred_verts.dim() == 2:
        pred_verts = pred_verts.unsqueeze(0)  # [1, N1, 3]
    if gt_verts.dim() == 2:
        gt_verts = gt_verts.unsqueeze(0)  # [1, N2, 3]
    
    # Compute distances from pred to gt
    knn_pred_to_gt = knn_points(pred_verts, gt_verts, K=1)
    dist_pred_to_gt = knn_pred_to_gt.dists.sqrt()  # [1, N1, 1]
    mean_dist_pred_to_gt = dist_pred_to_gt.mean()
    
    # Compute distances from gt to pred
    knn_gt_to_pred = knn_points(gt_verts, pred_verts, K=1)
    dist_gt_to_pred = knn_gt_to_pred.dists.sqrt()  # [1, N2, 1]
    mean_dist_gt_to_pred = dist_gt_to_pred.mean()
    
    # ASSD is the average of both directions
    assd = (mean_dist_pred_to_gt + mean_dist_gt_to_pred) / 2.0
    
    return assd

def compute_mesh_metrics_for_sequence(pred_sequence: torch.Tensor, 
                                    gt_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute HD and ASSD metrics for a sequence of meshes (across time frames).
    
    Args:
        pred_sequence: Predicted vertex sequence [seq_len, num_vertices, 3]
        gt_sequence: Ground truth vertex sequence [seq_len, num_vertices, 3]
    
    Returns:
        avg_hd: Average Hausdorff distance across time frames
        avg_assd: Average ASSD across time frames
    """
    seq_len = pred_sequence.shape[0]
    hd_values = []
    assd_values = []
    
    for t in range(seq_len):
        pred_verts = pred_sequence[t]  # [num_vertices, 3]
        gt_verts = gt_sequence[t]     # [num_vertices, 3]
        
        # Compute metrics for this time frame
        hd = compute_hausdorff_distance(pred_verts, gt_verts)
        assd = compute_assd(pred_verts, gt_verts)
        
        hd_values.append(hd)
        assd_values.append(assd)
    
    # Average across time frames
    avg_hd = torch.stack(hd_values).mean()
    avg_assd = torch.stack(assd_values).mean()
    
    return avg_hd, avg_assd

def compute_batch_metrics(pred_batch: torch.Tensor, 
                         gt_batch: torch.Tensor,
                         patient_ids: List[str] = None) -> dict:
    """
    Compute HD and ASSD metrics for a batch of mesh sequences.
    
    Args:
        pred_batch: Predicted batch [batch_size, seq_len, num_vertices, 3]
        gt_batch: Ground truth batch [batch_size, seq_len, num_vertices, 3]
        patient_ids: List of patient IDs for debugging
    
    Returns:
        Dictionary with metrics and statistics
    """
    batch_size = pred_batch.shape[0]
    hd_values = []
    assd_values = []
    
    for b in range(batch_size):
        try:
            pred_seq = pred_batch[b]  # [seq_len, num_vertices, 3]
            gt_seq = gt_batch[b]      # [seq_len, num_vertices, 3]
            
            # Compute metrics for this patient
            avg_hd, avg_assd = compute_mesh_metrics_for_sequence(pred_seq, gt_seq)
            
            hd_values.append(avg_hd.item())
            assd_values.append(avg_assd.item())
            
        except Exception as e:
            print(f"Error computing metrics for batch {b}: {e}")
            if patient_ids:
                print(f"Patient ID: {patient_ids[b]}")
            # Skip this sample
            continue
    
    if not hd_values:
        # Return default values if no valid computations
        return {
            'hd_mean': 0.0, 'hd_median': 0.0, 'hd_q1': 0.0, 'hd_q3': 0.0,
            'assd_mean': 0.0, 'assd_median': 0.0, 'assd_q1': 0.0, 'assd_q3': 0.0,
            'num_valid_samples': 0
        }
    
    # Convert to numpy for statistics
    hd_array = np.array(hd_values)
    assd_array = np.array(assd_values)
    
    # Compute statistics
    hd_stats = {
        'hd_mean': float(np.mean(hd_array)),
        'hd_median': float(np.median(hd_array)),
        'hd_q1': float(np.percentile(hd_array, 25)),
        'hd_q3': float(np.percentile(hd_array, 75)),
    }
    
    assd_stats = {
        'assd_mean': float(np.mean(assd_array)),
        'assd_median': float(np.median(assd_array)),
        'assd_q1': float(np.percentile(assd_array, 25)),
        'assd_q3': float(np.percentile(assd_array, 75)),
    }
    
    # Combine results
    metrics = {**hd_stats, **assd_stats, 'num_valid_samples': len(hd_values)}
    
    return metrics

class EpochMetricsAccumulator:
    """
    Accumulates metrics across batches within an epoch.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.hd_values = []
        self.assd_values = []
    
    def update(self, batch_metrics: dict):
        """Add metrics from a batch."""
        # We need to extract individual patient values, not the aggregated statistics
        # This assumes batch_metrics contains the raw values
        pass
    
    def update_raw_values(self, hd_values: List[float], assd_values: List[float]):
        """Add raw metric values from a batch."""
        self.hd_values.extend(hd_values)
        self.assd_values.extend(assd_values)
    
    def compute_epoch_stats(self) -> dict:
        """Compute statistics for the entire epoch."""
        if not self.hd_values:
            return {
                'hd_mean': 0.0, 'hd_median': 0.0, 'hd_q1': 0.0, 'hd_q3': 0.0,
                'assd_mean': 0.0, 'assd_median': 0.0, 'assd_q1': 0.0, 'assd_q3': 0.0,
                'num_samples': 0
            }
        
        hd_array = np.array(self.hd_values)
        assd_array = np.array(self.assd_values)
        
        return {
            'hd_mean': float(np.mean(hd_array)),
            'hd_median': float(np.median(hd_array)),
            'hd_q1': float(np.percentile(hd_array, 25)),
            'hd_q3': float(np.percentile(hd_array, 75)),
            'assd_mean': float(np.mean(assd_array)),
            'assd_median': float(np.median(assd_array)),
            'assd_q1': float(np.percentile(assd_array, 25)),
            'assd_q3': float(np.percentile(assd_array, 75)),
            'num_samples': len(self.hd_values)
        }

def print_metrics_summary(metrics: dict, prefix: str = ""):
    """Print a formatted summary of metrics."""
    print(f"{prefix}Mesh Reconstruction Metrics:")
    print(f"  Hausdorff Distance  - Mean: {metrics['hd_mean']:.4f}, Median: {metrics['hd_median']:.4f}")
    print(f"                      - Q1: {metrics['hd_q1']:.4f}, Q3: {metrics['hd_q3']:.4f}")
    print(f"  ASSD               - Mean: {metrics['assd_mean']:.4f}, Median: {metrics['assd_median']:.4f}")
    print(f"                      - Q1: {metrics['assd_q1']:.4f}, Q3: {metrics['assd_q3']:.4f}")
    print(f"  Valid Samples: {metrics.get('num_samples', metrics.get('num_valid_samples', 0))}")