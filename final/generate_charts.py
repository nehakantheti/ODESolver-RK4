"""Generate publication-quality charts for the Final Year Project report.

Reads the benchmark CSV files and generates three key charts:
1. Speedup Scaling (Damped Oscillator) across execution modes.
2. Multi-System Performance Summary (Multiproc Mode).
3. Iteration Bound (K vs P) across systems.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure seaborn for academic/publication aesthetic
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# Directory paths
RESULTS_DIR = Path("benchmarks/results")
CHARTS_DIR = Path("final/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load and combine all benchmark data."""
    dataframes = []
    modes = ["cpu", "gpu", "multiproc"]
    
    for mode in modes:
        file_path = RESULTS_DIR / mode / "parareal_hard_benchmark.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            dataframes.append(df)
        else:
            print(f"Warning: {file_path} not found.")
            
    if not dataframes:
        raise FileNotFoundError("No benchmark CSVs found in benchmarks/results/")
        
    return pd.concat(dataframes, ignore_index=True)


def plot_speedup_scaling(df):
    """Chart 1: Speedup vs Slabs (P) for Damped Oscillator across all modes."""
    damped_df = df[df['system'] == 'damped_oscillator'].copy()
    
    if damped_df.empty:
        return
        
    plt.figure(figsize=(8, 5))
    
    # Plot modes
    palette = {"cpu": "#e74c3c", "gpu": "#f39c12", "multiproc": "#27ae60"}
    markers = {"cpu": "X", "gpu": "s", "multiproc": "o"}
    
    sns.lineplot(
        data=damped_df, 
        x='n_slabs', y='speedup', hue='mode', style='mode',
        palette=palette, markers=markers, dashes=False, linewidth=2, markersize=8
    )
    
    # Add breakeven line
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label="1.0x (Serial Baseline)")
    
    plt.title("Parareal Speedup Scaling (Damped Oscillator)", pad=15, fontweight='bold')
    plt.xlabel("Number of Parallel Slabs ($P$)")
    plt.ylabel("Wall-clock Speedup ($T_{serial} / T_{parareal}$)")
    
    # Set x-ticks explicitly to match slab counts
    plt.xticks([2, 4, 8, 12, 16])
    
    plt.legend(title="Execution Mode", frameon=True)
    plt.tight_layout()
    
    # Save chart
    plt.savefig(CHARTS_DIR / "1_speedup_scaling.png", bbox_inches='tight')
    plt.savefig(CHARTS_DIR / "1_speedup_scaling.pdf", bbox_inches='tight')
    plt.close()


def plot_multi_system_summary(df):
    """Chart 2: Max speedup per system in multiproc mode."""
    multiproc_df = df[df['mode'] == 'multiproc'].copy()
    
    if multiproc_df.empty:
        return
        
    # Get max speedup for each system
    max_speedups = multiproc_df.groupby('system')['speedup'].max().reset_index()
    
    # Prettify system names
    name_mapping = {
        'damped_oscillator': 'Damped\nOscillator',
        'van_der_pol': 'Van der Pol\nOscillator',
        'lotka_volterra': 'Lotka-Volterra\n(Predator-Prey)',
        'lorenz': 'Lorenz\nAttractor'
    }
    max_speedups['system_display'] = max_speedups['system'].map(name_mapping)
    
    plt.figure(figsize=(8, 5))
    
    # Use a clean blue palette
    ax = sns.barplot(
        data=max_speedups, 
        x='system_display', y='speedup', 
        palette="Blues_d",
        edgecolor=".2"
    )
    
    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(
            f"{p.get_height():.2f}x",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            xytext=(0, 5), textcoords='offset points',
            fontweight='bold'
        )
    
    # Add breakeven line
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label="1.0x (Serial Baseline)")
    
    plt.title("Peak Parallel Speedup by ODE System (Multiproc Mode)", pad=15, fontweight='bold')
    plt.xlabel("")
    plt.ylabel("Maximum Achieved Speedup")
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    
    # Save chart
    plt.savefig(CHARTS_DIR / "2_system_performance.png", bbox_inches='tight')
    plt.savefig(CHARTS_DIR / "2_system_performance.pdf", bbox_inches='tight')
    plt.close()


def plot_iteration_bound(df):
    """Chart 3: Iterations (K) vs Slabs (P) for multiproc mode."""
    multiproc_df = df[df['mode'] == 'multiproc'].copy()
    
    if multiproc_df.empty:
        return
        
    # Prettify system names
    name_mapping = {
        'damped_oscillator': 'Damped Oscillator',
        'van_der_pol': 'Van der Pol Oscillator',
        'lotka_volterra': 'Lotka-Volterra',
        'lorenz': 'Lorenz Attractor'
    }
    multiproc_df['system_display'] = multiproc_df['system'].map(name_mapping)
    
    plt.figure(figsize=(8, 5))
    
    # Plot K vs P
    sns.lineplot(
        data=multiproc_df, 
        x='n_slabs', y='iterations', hue='system_display', style='system_display',
        markers=True, dashes=False, linewidth=2, markersize=8
    )
    
    # Add the K = P theoretical bound line
    plt.plot([2, 16], [2, 16], color='red', linestyle='--', alpha=0.5, label="Worst-Case Bound ($K = P$)")
    
    plt.title("Neural Parareal Iteration Bound ($K$ vs $P$)", pad=15, fontweight='bold')
    plt.xlabel("Number of Parallel Slabs ($P$)")
    plt.ylabel("Iterations to Converge ($K$)")
    
    plt.xticks([2, 4, 8, 12, 16])
    plt.yticks([2, 4, 6, 8, 10, 12, 14, 16, 18])
    
    plt.legend(title="ODE System", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save chart
    plt.savefig(CHARTS_DIR / "3_iteration_bound.png", bbox_inches='tight')
    plt.savefig(CHARTS_DIR / "3_iteration_bound.pdf", bbox_inches='tight')
    plt.close()


def main():
    try:
        df = load_data()
        print(f"Loaded {len(df)} benchmark records.")
        
        plot_speedup_scaling(df)
        print("Generated 1_speedup_scaling")
        
        plot_multi_system_summary(df)
        print("Generated 2_system_performance")
        
        plot_iteration_bound(df)
        print("Generated 3_iteration_bound")
        
        print(f"\nSuccess! All charts saved to {CHARTS_DIR}")
        
    except Exception as e:
        print(f"Error generating charts: {e}")


if __name__ == "__main__":
    main()
