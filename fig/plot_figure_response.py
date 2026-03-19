"""
Three-panel coupling schematic figure for NMI paper.
Panel (a): Decoupled — Gen only scales, Sel fixed
Panel (b): Negative coupling — co-scaling reduces marginal return
Panel (c): Positive coupling — co-scaling amplifies marginal return

Each panel shows:
  - Top: Architecture diagram (B_ref input, resource distribution, info flow)
  - Bottom: J vs B_ref curves (fixed configs faded, nested envelope)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_rcparams():
    plt.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'legend.fontsize': 5.5,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
    })


def draw_architecture(ax, mode='decoupled'):
    """
    Draw architecture diagram showing B_ref input, resource distribution, info flow.

    mode: 'decoupled', 'negative', 'positive'
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # B_ref input from outside (left)
    ax.annotate('', xy=(1.0, 4.5), xytext=(-2.5, 4.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#2166AC'))
    ax.text(-1.5, 5, r'$\mathcal{B}_\mathrm{ref}$', ha='center', fontsize=12,color = '#2166AC')

    # Generator box
    gen_box = FancyBboxPatch((2, 6), 2.5, 2, boxstyle="round,pad=0.1",
                             edgecolor='#2166AC', facecolor='#D1E5F0', lw=1.2)
    ax.add_patch(gen_box)
    ax.text(3.25, 7, 'Gen', ha='center', va='center', fontsize=12, weight='bold')

    # Selector box
    sel_box = FancyBboxPatch((2, 0.5), 2.5, 2, boxstyle="round,pad=0.1",
                             edgecolor='#B2182B', facecolor='#F4A582', lw=1.2)
    ax.add_patch(sel_box)
    ax.text(3.25, 1.5, 'Sel', ha='center', va='center', fontsize=12, weight='bold')

    if mode == 'decoupled':
        # B_ref → Gen only
        ax.annotate('', xy=(2, 6), xytext=(1, 4.5),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color='#2166AC'))
        ax.text(-0.3, 5.5, r'$\mathcal{B}_\mathrm{gen}$', fontsize=12, color='#2166AC')
        # Sel fixed (no arrow from B_ref)
        ax.annotate('', xy=(2, 2.5), xytext=(1, 4.5),
                    arrowprops=dict(arrowstyle='-', lw=1.2, color='#666'))
        ax.text(1.2, 2.75, r'$\mathcal{B}_\mathrm{sel}$ fixed', fontsize=12,
                ha='right', style='italic', color='#666')
    else:
        # B_ref → both Gen and Sel
        ax.annotate('', xy=(2, 6), xytext=(1, 4.5),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color='#2166AC'))
        ax.text(-0.3, 5.5, r'$\mathcal{B}_\mathrm{gen}$', fontsize=12, color='#2166AC')
        if mode == 'negative':
            ax.annotate('', xy=(2, 2.5), xytext=(1, 4.5),
                        arrowprops=dict(arrowstyle='->', lw=1.2, color='#B2182B'))
            ax.text(-1.2, 2.75, r'$\mathcal{B}_\mathrm{sel} -$', fontsize=12, color='#B2182B')
        else:
            ax.annotate('', xy=(2, 2.5), xytext=(1, 4.5),
                        arrowprops=dict(arrowstyle='->', lw=1.2, color='#4393C3'))
            ax.text(-1.2, 2.75, r'$\mathcal{B}_\mathrm{sel} +$', fontsize=12, color='#4393C3')

        # Coupling indicator between Gen and Sel
        coupling_color = '#D6604D' if mode == 'negative' else '#4393C3'
        # coupling_label = '−' if mode == 'negative' else '+'
        # ax.annotate('', xy=(4.5, 6), xytext=(4.5, 4),
                    # arrowprops=dict(arrowstyle='<->', lw=1.0, color=coupling_color,
                                    # linestyle='--'))
        # ax.text(5.2, 5, coupling_label, fontsize=9, weight='bold',
                # color=coupling_color, va='center')

    # Information flow: Gen → Sel → output
    ax.annotate('', xy=(3.25, 2.5), xytext=(3.25, 6),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#333',
                                connectionstyle='arc3,rad=0.3'))
    # ax.text(4.5, 4.5, 'info\nflow', fontsize=5, ha='center', color='#333')

    # Output arrow
    ax.annotate('', xy=(8, 4.5), xytext=(4.5, 2.6),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.text(8.3, 4.5, r'$J$', ha='center', fontsize=12)


def plot_curves(ax, mode='decoupled'):
    """
    Plot J vs B_ref curves (linear assumption).

    mode: 'decoupled', 'negative', 'positive'
    """
    b_ref = np.linspace(1, 10, 50)

    # Base slope for fixed-selector curves
    base_slope = 3.0

    # Three fixed-selector curves (parallel lines with different intercepts)
    intercepts = [5, 12, 19]  # Different B_sel values give different intercepts
    fixed_curves = []

    for intercept in intercepts:
        j_fixed = intercept + base_slope * b_ref
        fixed_curves.append(j_fixed)
        ax.plot(b_ref, j_fixed, color='#92C5DE', alpha=0.3, lw=1.5, zorder=1)

    # Nested (co-scaled) curve
    if mode == 'decoupled':
        # Decoupled: same slope as fixed curves, coincides with middle fixed curve
        j_nested = intercepts[1] + base_slope * b_ref  # Middle fixed curve
        color = '#666'
        # No intersection points to mark (nested coincides with middle fixed curve everywhere)
    elif mode == 'negative':
        # Negative coupling: much lower slope (exaggerated)
        nested_slope = base_slope * 0.4  # More exaggerated: 1.2 vs 3.0
        nested_intercept = 22  # Move up higher to intersect with top line
        j_nested = nested_intercept + nested_slope * b_ref
        color = '#D6604D'
        # Mark intersections with fixed curves
        for intercept in intercepts:
            b_intersect = (intercept - nested_intercept) / (nested_slope - base_slope)
            if 1 <= b_intersect <= 10:
                j_intersect = nested_intercept + nested_slope * b_intersect
                ax.plot(b_intersect, j_intersect, 'o', color=color, markersize=4,
                       markeredgecolor='white', markeredgewidth=0.5, zorder=11)
    else:  # positive
        # Positive coupling: much higher slope (exaggerated)
        nested_slope = base_slope * 1.8  # More exaggerated: 5.4 vs 3.0
        nested_intercept = 0  # Move down to intersect with bottom line
        j_nested = nested_intercept + nested_slope * b_ref
        color = '#4393C3'
        # Mark intersections with fixed curves
        for intercept in intercepts:
            b_intersect = (intercept - nested_intercept) / (nested_slope - base_slope)
            if 1 <= b_intersect <= 10:
                j_intersect = nested_intercept + nested_slope * b_intersect
                ax.plot(b_intersect, j_intersect, 'o', color=color, markersize=4,
                       markeredgecolor='white', markeredgewidth=0.5, zorder=11)

    ax.plot(b_ref, j_nested, color=color, lw=2, zorder=10, label='Nested')

    ax.set_xlabel(r'$\mathcal{B}_\mathrm{ref}$')
    ax.set_ylabel(r'$J$')
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_xlim(1, 10)
    ax.set_ylim(0, 50)


def plot_three_panel_response():
    setup_rcparams()

    fig = plt.figure(figsize=(7, 4.5))

    # Three columns
    for i, (mode, title) in enumerate([('decoupled', 'a'),
                                        ('negative', 'b'),
                                        ('positive', 'c')]):
        # Architecture diagram (top)
        ax_arch = plt.subplot(2, 3, i + 1)
        draw_architecture(ax_arch, mode=mode)
        ax_arch.set_title(title, loc='left', fontweight='bold', pad=5)

        # Curves (bottom)
        ax_curve = plt.subplot(2, 3, i + 4)
        plot_curves(ax_curve, mode=mode)

    plt.tight_layout(pad=0.5, h_pad=1.5, w_pad=1.0)

    for ext in ['pdf', 'png']:
        path = os.path.join(OUT_DIR, f'figure_response.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    plot_three_panel_response()
