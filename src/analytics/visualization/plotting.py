"""
Visualization tools for research-grade analytics.

Provides publication-quality plots for:
- CKA similarity heatmaps
- Attention distance profiles
- Hessian trace comparisons
- Locality spectrum (forensics)
- Experiment result summaries
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class AnalyticsVisualizer:
    """
    Visualization tools for research-grade analytics.
    """

    @staticmethod
    def plot_cka_heatmap(
        cka_matrix,
        layer_indices_x=None,
        layer_indices_y=None,
        title="CKA Similarity",
        xlabel="Teacher Layer",
        ylabel="Student Layer",
        save_path=None,
        figsize=(10, 8),
        cmap='viridis'
    ):
        """Plot CKA similarity heatmap."""
        cka_matrix = np.array(cka_matrix)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(cka_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('CKA Similarity', fontsize=12)

        # Set axis labels
        if layer_indices_x is not None:
            ax.set_xticks(range(len(layer_indices_x)))
            ax.set_xticklabels([f'L{i}' for i in layer_indices_x], fontsize=9)
        if layer_indices_y is not None:
            ax.set_yticks(range(len(layer_indices_y)))
            ax.set_yticklabels([f'L{i}' for i in layer_indices_y], fontsize=9)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)

        # Add value annotations for small matrices
        if cka_matrix.shape[0] <= 12 and cka_matrix.shape[1] <= 12:
            for i in range(cka_matrix.shape[0]):
                for j in range(cka_matrix.shape[1]):
                    value = cka_matrix[i, j]
                    color = 'white' if value < 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color=color, fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"CKA heatmap saved to {save_path}")

        plt.show()
        return fig

    @staticmethod
    def plot_attention_distances(
        layer_distances,
        title="Mean Attention Distance per Layer",
        save_path=None,
        figsize=(12, 6),
        color='steelblue',
        comparison_data=None
    ):
        """Plot mean attention distance per layer."""
        fig, ax = plt.subplots(figsize=figsize)

        layers = list(layer_distances.keys())
        distances = list(layer_distances.values())

        x = np.arange(len(layers))
        width = 0.35 if comparison_data else 0.7

        bars1 = ax.bar(x - width/2 if comparison_data else x, distances, width,
                      label='Model', color=color, alpha=0.8)

        if comparison_data:
            comp_distances = [comparison_data.get(l, 0) for l in layers]
            bars2 = ax.bar(x + width/2, comp_distances, width,
                          label='Comparison', color='coral', alpha=0.8)

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Mean Attention Distance (patches)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)

        if comparison_data:
            ax.legend()

        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention distance plot saved to {save_path}")

        plt.show()
        return fig

    @staticmethod
    def plot_hessian_comparison(
        results,
        metric='trace',
        title="Hessian Trace Comparison",
        save_path=None,
        figsize=(10, 6)
    ):
        """Plot Hessian metric comparison across experiments."""
        fig, ax = plt.subplots(figsize=figsize)

        experiments = list(results.keys())
        values = [results[exp].get(metric, 0) for exp in experiments]
        errors = [results[exp].get(f'{metric}_std', 0) for exp in experiments]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(experiments)))

        bars = ax.bar(experiments, values, yerr=errors, capsize=5,
                     color=colors, alpha=0.8, edgecolor='black')

        ax.set_ylabel(f'Hessian {metric.replace("_", " ").title()}', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        if len(experiments) > 4:
            plt.xticks(rotation=45, ha='right')

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Hessian comparison plot saved to {save_path}")

        plt.show()
        return fig

    @staticmethod
    def plot_experiment_summary(
        results,
        metrics=['accuracy', 'hessian_trace', 'mean_attention_distance'],
        title="Experiment Comparison Summary",
        save_path=None,
        figsize=(14, 5)
    ):
        """Create multi-panel summary of experiment results."""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

        if n_metrics == 1:
            axes = [axes]

        experiments = list(results.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))

        metric_labels = {
            'accuracy': 'Test Accuracy (%)',
            'hessian_trace': 'Hessian Trace',
            'mean_attention_distance': 'Mean Attn Distance',
            'cka_diagonal_mean': 'CKA Diagonal Mean'
        }

        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = []

            for exp in experiments:
                if metric in results[exp]:
                    values.append(results[exp][metric])
                elif metric == 'hessian_trace' and 'hessian' in results[exp]:
                    values.append(results[exp]['hessian'].get('trace', 0))
                elif metric == 'mean_attention_distance' and 'attention' in results[exp]:
                    values.append(results[exp]['attention'].get('mean_distance', 0))
                else:
                    values.append(0)

            bars = ax.bar(experiments, values, color=colors, alpha=0.8, edgecolor='black')
            ax.set_ylabel(metric_labels.get(metric, metric.replace('_', ' ').title()))
            ax.set_title(metric_labels.get(metric, metric.replace('_', ' ').title()))
            ax.grid(True, alpha=0.3, axis='y')

            if len(experiments) > 3:
                ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Experiment summary saved to {save_path}")

        plt.show()
        return fig

    @staticmethod
    def plot_distillation_loss_curves(
        metrics_history,
        title="Distillation Training Curves",
        save_path=None,
        figsize=(14, 10)
    ):
        """Plot detailed distillation training curves."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Total loss
        if 'train_loss' in metrics_history:
            axes[0, 0].plot(metrics_history['train_loss'], label='Train', color='blue')
            if 'val_loss' in metrics_history:
                axes[0, 0].plot(metrics_history['val_loss'], label='Val', color='orange')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        if 'train_acc' in metrics_history:
            axes[0, 1].plot(metrics_history['train_acc'], label='Train', color='blue')
            if 'val_acc' in metrics_history:
                axes[0, 1].plot(metrics_history['val_acc'], label='Val', color='orange')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # CE Loss
        if 'train_ce_loss' in metrics_history:
            axes[0, 2].plot(metrics_history['train_ce_loss'], label='CE Loss', color='green')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].set_title('Classification Loss')
            axes[0, 2].grid(True, alpha=0.3)

        # Token Loss
        if 'train_tok_loss' in metrics_history:
            axes[1, 0].plot(metrics_history['train_tok_loss'], label='Token Loss', color='purple')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Token Representation Loss')
            axes[1, 0].grid(True, alpha=0.3)

        # CKA/Correlation Loss
        if 'train_cka_loss' in metrics_history:
            axes[1, 1].plot(metrics_history['train_cka_loss'], label='CKA Loss', color='red')
        if 'train_rel_loss' in metrics_history:
            axes[1, 1].plot(metrics_history['train_rel_loss'], label='Correlation Loss',
                          color='coral', linestyle='--')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Structural Losses')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Lambda schedule
        if 'effective_lambda_rel' in metrics_history or 'effective_lambda_cka' in metrics_history:
            if 'effective_lambda_rel' in metrics_history:
                axes[1, 2].plot(metrics_history['effective_lambda_rel'],
                               label='Lambda Rel', color='coral')
            if 'effective_lambda_cka' in metrics_history:
                axes[1, 2].plot(metrics_history['effective_lambda_cka'],
                               label='Lambda CKA', color='red')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Lambda')
            axes[1, 2].set_title('Loss Weight Schedule')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distillation curves saved to {save_path}")

        plt.show()
        return fig

    @staticmethod
    def plot_locality_spectrum(
        results_dict: Dict[str, Dict],
        output_dir: str,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot the distribution of attention distances across heads (sorted).

        This visualization avoids the head-matching problem by sorting distances.

        Expected patterns:
            - CNN-Student curve: "convex" (most heads local)
            - Baseline curve: Medium
            - DINO-Student curve: "concave" or linear (many global heads)
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

        for idx, (model_name, stats) in enumerate(results_dict.items()):
            all_distances = []
            if 'per_layer_stats' in stats:
                for layer_stats in stats['per_layer_stats'].values():
                    if isinstance(layer_stats, dict) and 'mean_distance' in layer_stats:
                        all_distances.extend(layer_stats['mean_distance'].tolist()
                                            if hasattr(layer_stats['mean_distance'], 'tolist')
                                            else list(layer_stats['mean_distance']))

            if all_distances:
                sorted_dist = np.sort(all_distances)
                ax.plot(sorted_dist, label=model_name, linewidth=2.5,
                       color=colors[idx % len(colors)])

        ax.set_title("Locality Spectrum: Distribution of Attention Capacity", fontsize=14)
        ax.set_ylabel("Mean Attention Distance (Patch Units)", fontsize=12)
        ax.set_xlabel("Head Rank (Sorted Local -> Global)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/locality_spectrum.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/locality_spectrum.pdf", bbox_inches='tight')
        logger.info(f"Locality spectrum saved to {output_dir}/locality_spectrum.{{png,pdf}}")

        plt.close()
        return fig

    @staticmethod
    def plot_layer_progression(
        results_dict: Dict[str, Dict],
        output_dir: str,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """Plot average attention distance per layer."""
        fig, ax = plt.subplots(figsize=figsize)

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
        markers = ['o', 's', '^', 'D', 'v']

        for idx, (model_name, stats) in enumerate(results_dict.items()):
            if 'per_layer_stats' not in stats:
                continue

            layer_keys = sorted(stats['per_layer_stats'].keys(),
                               key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else 0)
            avg_distances = []

            for layer_key in layer_keys:
                layer_stats = stats['per_layer_stats'][layer_key]
                if isinstance(layer_stats, dict) and 'mean_distance' in layer_stats:
                    mean_dist = layer_stats['mean_distance']
                    avg_distances.append(np.mean(mean_dist))

            if avg_distances:
                ax.plot(range(len(avg_distances)), avg_distances, 'o-',
                       label=model_name, linewidth=2, markersize=8,
                       color=colors[idx % len(colors)],
                       marker=markers[idx % len(markers)])

        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Mean Attention Distance", fontsize=12)
        ax.set_title("Attention Locality by Layer Depth", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/layer_progression.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/layer_progression.pdf", bbox_inches='tight')
        logger.info(f"Layer progression saved to {output_dir}/layer_progression.{{png,pdf}}")

        plt.close()
        return fig

    @staticmethod
    def plot_forensics_summary(
        all_results: Dict[str, Dict],
        output_dir: str,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """Create 4-panel publication-ready summary figure."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        model_names = list(all_results.keys())
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12'][:len(model_names)]

        # Panel 1: Hessian Trace
        ax = axes[0, 0]
        traces = [all_results[m].get('hessian_trace', 0) for m in model_names]
        bars = ax.bar(model_names, traces, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel("Hessian Trace", fontsize=11)
        ax.set_title("Loss Landscape Sharpness", fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, traces):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

        # Panel 2: Locality Spectrum (inline version)
        ax = axes[0, 1]
        for idx, (model_name, stats) in enumerate(all_results.items()):
            all_distances = []
            if 'per_layer_stats' in stats:
                for layer_stats in stats['per_layer_stats'].values():
                    if isinstance(layer_stats, dict) and 'mean_distance' in layer_stats:
                        md = layer_stats['mean_distance']
                        all_distances.extend(md.tolist() if hasattr(md, 'tolist') else list(md))
            if all_distances:
                sorted_dist = np.sort(all_distances)
                ax.plot(sorted_dist, label=model_name, linewidth=2, color=colors[idx])
        ax.set_xlabel("Head Rank", fontsize=11)
        ax.set_ylabel("Attention Distance", fontsize=11)
        ax.set_title("Locality Spectrum", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 3: CLS Dispersion
        ax = axes[1, 0]
        dispersions = [all_results[m].get('avg_cls_dispersion', 0) for m in model_names]
        bars = ax.bar(model_names, dispersions, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel("CLS Dispersion", fontsize=11)
        ax.set_title("CLS Token Spatial Coverage", fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, dispersions):
            ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

        # Panel 4: Collapsed Heads Ratio
        ax = axes[1, 1]
        collapsed = [all_results[m].get('collapsed_heads_ratio', 0) * 100 for m in model_names]
        bars = ax.bar(model_names, collapsed, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel("Collapsed Heads (%)", fontsize=11)
        ax.set_title("Heads with MAD < 1.5 patches", fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, collapsed):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

        plt.suptitle("Locality Curse Forensics Summary", fontsize=14, y=1.02)
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/forensics_summary.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/forensics_summary.pdf", bbox_inches='tight')
        logger.info(f"Forensics summary saved to {output_dir}/forensics_summary.{{png,pdf}}")

        plt.close()
        return fig


__all__ = ['AnalyticsVisualizer']
