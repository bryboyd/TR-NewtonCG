import matplotlib.pyplot as plt
import numpy as np

from typing import List


def plot_basic_test(x_test,
                    y_true = None,
                    y_pred = None,
                    fig = None,
                    ax = None,
                    true_color = "blue",
                    pred_label = "Model Prediction",
                    pred_color = "orange",
                    eqn = "",
                    title = "",
                    fig_size = (8, 5),
                    out_file = ""):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    if y_true is not None:
        ax.plot(x_test,
                y_true,
                label=f"True {eqn}",
                c=true_color,
                linewidth=2)
    if y_pred is not None:
        ax.plot(x_test,
                y_pred,
                label=pred_label,
                c=pred_color,
                linewidth=1.5,
                linestyle="--")

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()

    if out_file:
        fig.savefig(out_file)
        plt.close()

    return fig, ax


def plot_pinn_test(test_pts,
                   test_vals,
                   fig = None,
                   ax = None,
                   fig_size = (7, 6),
                   cmap = "hot",
                   title = "",
                   out_file = ""):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
        
    # Need points in CPU for plotting
    test_pts = test_pts.detach().cpu().numpy()
    test_vals = test_vals.detach().cpu().numpy()

    sc = ax.scatter(test_pts[:, 0], test_pts[:, 1], c=test_vals, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    fig.colorbar(sc, ax=ax)
    fig.tight_layout()

    if out_file:
        fig.savefig(out_file)
        plt.close()
    
    return fig, ax


def plot_pinn_losses(epochs,
                     total_loss,
                     dom_loss = None,
                     bdry_loss = None,
                     fig = None,
                     ax = None,
                     fig_size = (6, 6),
                     title = "",
                     out_file = ""):
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    ax.plot(epochs, total_loss, label="Total loss")

    if dom_loss is not None:
        ax.plot(epochs, dom_loss, label="Domain loss")
    if bdry_loss is not None:
        ax.plot(epochs, bdry_loss, label="Domain loss")

    ax.grid()
    ax.set_title(title)
    fig.tight_layout()

    if out_file:
        fig.savefig(out_file)
        plt.close()
    
    return fig, ax


def plot_losses_list(epochs_list: List,
                     losses: List,
                     labels: List = None,
                     fig = None,
                     ax = None,
                     fig_size = (6, 6),
                     title = "",
                     out_file = ""):
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    for i, loss in enumerate(losses):
        if labels is not None:
            label = labels[i]
        else:
            label = ""

        ax.plot(epochs_list[i], loss, label=label)

    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()
    fig.tight_layout()

    if out_file:
        fig.savefig(out_file)
        plt.close()
    
    return fig, ax


def plot_epoch_data(epochs,
                    data,
                    fig = None,
                    ax = None,
                    fig_size = (6, 6),
                    x_log = False,
                    y_log = False,
                    y_label = "",
                    label = "",
                    show_legend = False,
                    title = "",
                    out_file = ""):
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    ax.plot(epochs, data, label=label)

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel(y_label)
    if show_legend:
        ax.legend()
    ax.grid()
    fig.tight_layout()

    if out_file:
        fig.savefig(out_file)
        plt.close()
    
    return fig, ax


def plot_time_comparison(fns,
                         adam_data,
                         tr_cg_data,
                         fig_size = (9, 6),
                         fig = None,
                         ax = None,
                         y_label = "Time",
                         title = "Time by Optimizer and Function",
                         out_file = ""):
    x = np.arange(len(fns)) 
    width = 0.35

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    # Bars
    bars_adam = ax.bar(x - width/2, adam_data, width, label="Adam", color="blue")
    bars_tr = ax.bar(x + width/2, tr_cg_data, width, label="TR Newton-CG", color="orange")

    ax.set_xlabel("Function")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(fns)
    ax.legend()

    # Annotate bars
    for bars in [bars_adam, bars_tr]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{int(height)}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # offset
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7.5)

    plt.tight_layout()

    if out_file:
        plt.savefig(out_file)
        plt.close()

    return fig, ax


if __name__ == "__main__":
    pass