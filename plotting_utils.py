import matplotlib.pyplot as plt
import numpy as np
from IPython.display import set_matplotlib_formats


def initialize_params():

    set_matplotlib_formats('pdf', 'png')
    plt.rcParams['savefig.dpi'] = 75

    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.figsize'] = 10, 6
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['font.size'] = 16
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['legend.fontsize'] = 10

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "serif"
    plt.rcParams['font.serif'] = "cm"
    plt.rcParams[
        'text.latex.preamble'] = r"\usepackage{subdepth}, \usepackage{type1cm}"


def ndcg_vs_disparity_plot(plt_data_mats, names, join=False, ranges=None):
    if ranges:
        plt.xlim(ranges[0])
        plt.ylim(ranges[1])
    for i, plt_data_mat in enumerate(plt_data_mats):
        if not join:
            plt.scatter(
                plt_data_mats[i][:, 0],
                plt_data_mats[i][:, 1],
                marker="*",
                label=names[i])
        else:
            plt.plot(
                plt_data_mats[i][:, 0],
                plt_data_mats[i][:, 1],
                marker="*",
                label=names[i])
    plt.legend()
    plt.title("Utility-Fairness Trade-off")
    plt.xlabel("NDCG")
    plt.ylabel("Disparity")
    plt.grid()
    plt.show()


def plot_variance_bar_plot(matrix, lgroups):
    # dim1 is the different lambdas, dim2 is the runs
    means = np.mean(matrix, axis=1)
    variances = np.var(matrix, axis=1)
    print(means, lgroups)
    plt.figure()

    plt.semilogx(lgroups, means)
    plt.fill_between(
        lgroups,
        means - np.sqrt(variances),
        means + np.sqrt(variances),
        alpha=0.5)
    plt.show()


from symlog_axes import MinorSymLogLocator


def plot_multiaxis_plot(matrix1, matrix2, lgroups, ranges=None):
    fig, ax1 = plt.subplots()
    means = np.mean(matrix1, axis=1)
    stds = np.sqrt(np.var(matrix1, axis=1))
    stderrs = np.sqrt(np.var(matrix1, axis=1)) / np.sqrt(np.shape(matrix1)[1])

    ax1.plot(lgroups, means, marker='+', color='purple', label=r'Test NDCG@10')
    ax1.fill_between(
        lgroups, means - stderrs, means + stderrs, alpha=0.5, color='purple')
    ax1.set_xscale('symlog', linthreshx=0.1)
    ax1.yaxis.set_label_position('left')
    ax1.set_xlabel(r'$\lambda_{\rm ind}$', fontsize=20)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel("NDCG", color='purple')
    ax1.tick_params('y', colors='purple')
    if ranges:
        ax1.set_ylim(ranges[0])

    ax2 = ax1.twinx()
    means = np.mean(matrix2, axis=1)
    stds = np.sqrt(np.var(matrix2, axis=1))
    stderrs = np.sqrt(np.var(matrix2, axis=1)) / np.sqrt(np.shape(matrix2)[1])

    ax2.plot(lgroups, means, color='r', label=r'$\mathcal{D}_{\rm group}$')
    ax2.fill_between(
        lgroups, means - stderrs, means + stderrs, alpha=0.5, color='r')
    ax2.set_xscale('symlog', linthreshx=0.1)
    ax2.xaxis.set_minor_locator(MinorSymLogLocator(1e-1))
    ax2.yaxis.set_label_position('right')

    ax2.tick_params('y', color='r')
    if ranges:
        ax2.set_ylim(ranges[1])
    ax1.legend(loc=3, fontsize=12)
    ax2.legend(loc=4, fontsize=12)
    ax2.set_ylabel(r'$\mathcal{D}_{\rm group}$', fontsize=18, color='red')
    fig.tight_layout()
    plt.show()
