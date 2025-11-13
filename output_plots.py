"""
Load models and plot Adam vs. TR Newton-CG in the same plot
"""
import numpy as np
import torch
import math
import os
import fnmatch

from src.plotting import *
from src.fcn import FCN

def plot_combined_convergence(x_test):
    y_true = fn_fn(x_test)

    adam_model = f"./models/{dir_date}/adam_model_{fn_name}.pt"
    tr_cg_model = f"./models/{dir_date}/tr_cg_model_{fn_name}.pt"

    # Adam convergence
    fcn_adam = FCN(1, 10, 10, 1)
    fcn_adam.eval()
    fcn_adam.load_state_dict(torch.load(adam_model, map_location=torch.device("cpu")))
    with torch.no_grad():
        y_hat_adam = fcn_adam(x_test)

    result_fig, result_ax = plot_basic_test(x_test,
                                            y_true=y_true,
                                            y_pred=y_hat_adam,
                                            true_color="darkgray",
                                            pred_color="blue",
                                            pred_label="Adam",
                                            eqn=fn_eqn)

    # TRNCG convergence
    fcn_tr_cg = FCN(1, 10, 10, 1)
    fcn_tr_cg.eval()
    fcn_tr_cg.load_state_dict(torch.load(tr_cg_model, map_location=torch.device("cpu")))
    with torch.no_grad():
        y_hat_tr_cg = fcn_tr_cg(x_test)

    result_fig, result_ax = plot_basic_test(x_test,
                                            y_pred=y_hat_tr_cg,
                                            fig=result_fig,
                                            ax=result_ax,
                                            pred_color="orange",
                                            pred_label="TR Newton-CG",
                                            eqn=fn_eqn,
                                            title=f"Comparison of Model Convergence to {fn_eqn}",
                                            out_file=f"./images/{dir_date}/model_convergence/{fn_name}.pdf")


def plot_loss_comparison(start_step=0, end_step=500):
    bigger_end = 0
    loss_col = 1 if fn_name != "pinn" else -1

    adam_max_epochs = adam_data.shape[0]
    if adam_max_epochs < end_step:
        end_step = adam_max_epochs
    adam_epochs = np.arange(start=start_step, stop=end_step)
    adam_loss = adam_data[start_step:end_step, loss_col]

    if end_step > bigger_end:
        bigger_end = end_step

    tr_cg_max_epochs = tr_cg_data.shape[0]
    if tr_cg_max_epochs < end_step:
        end_step = tr_cg_max_epochs
    tr_cg_epochs = np.arange(start=start_step, stop=end_step)
    tr_cg_loss = tr_cg_data[start_step:end_step, loss_col]

    if end_step > bigger_end:
        bigger_end = end_step

    plot_losses_list(epochs_list=[adam_epochs, tr_cg_epochs],
                    losses=[adam_loss, tr_cg_loss],
                    labels=["Adam", "TR Newton-CG"],
                    fig_size=(9, 6),
                    title=f"Loss Comparison for {fn_eqn} (Iteration {start_step} to {bigger_end})",
                    out_file=f"./images/{dir_date}/loss_comparison/{fn_name}_{start_step}-{bigger_end}.pdf")


def plot_tr_rad(start_step=0, end_step=500):
    """
    Plot TR radius per iteration
    """
    tr_cg_max_epochs = tr_cg_data.shape[0]
    if tr_cg_max_epochs < end_step:
        end_step = tr_cg_max_epochs
    tr_cg_epochs = np.arange(start=start_step, stop=end_step)
    
    if end_step == -1:
        end_step = tr_cg_max_epochs

    tr_cg_rad = tr_cg_data[start_step:end_step, -1]

    plot_epoch_data(tr_cg_epochs,
                    tr_cg_rad,
                    fig_size=(9, 6),
                    y_label = "Trust Region Radius",
                    title=f"Evolution of TR Radius for {fn_eqn} (Iteration {start_step} to {end_step})",
                    out_file=f"./images/{dir_date}/tr_evolution/{fn_name}_{start_step}-{end_step}.pdf")


def plot_grad_norms(start_step=0, end_step=500):
    """
    Plot norm of gradient of loss per iteration
    """
    adam_max_epochs = adam_data.shape[0]
    if adam_max_epochs < end_step:
        end_step = adam_max_epochs
    adam_epochs = np.arange(start=start_step, stop=end_step)
    adam_grad_norm = adam_data[start_step:end_step, 2]

    grad_fig, grad_ax = plot_epoch_data(adam_epochs,
                                        adam_grad_norm,
                                        fig_size=(9, 6),
                                        label="Adam",
                                        y_label = r"$\|\nabla f \|_2$",
                                        title=rf"Evolution of $\|\nabla f\|_2$ for {fn_eqn} (Iteration {start_step} to {end_step})")

    tr_cg_max_epochs = tr_cg_data.shape[0]
    if tr_cg_max_epochs < end_step:
        end_step = tr_cg_max_epochs
    tr_cg_epochs = np.arange(start=start_step, stop=end_step)

    tr_cg_grad_norm = tr_cg_data[start_step:end_step, 2]
    plot_epoch_data(tr_cg_epochs,
                    tr_cg_grad_norm,
                    fig=grad_fig,
                    ax=grad_ax,
                    label="TR Newton-CG",
                    show_legend=True,
                    fig_size=(9, 6),
                    y_label = r"$\|\nabla f \|_2$",
                    title=rf"Evolution of $\|\nabla f\|_2$ for {fn_eqn} (Iteration {start_step} to {end_step})",
                    out_file=f"./images/{dir_date}/grad_norm/{fn_name}_{start_step}-{end_step}.pdf")


def plot_tr_cg_tracker(start_step=0, end_step=500):
    tr_cg_max_epochs = tr_cg_data.shape[0]
    if tr_cg_max_epochs < end_step:
        end_step = tr_cg_max_epochs
    tr_cg_epochs = np.arange(start=start_step, stop=end_step)
    
    if end_step == -1:
        end_step = tr_cg_max_epochs

    rho = cg_tracker_data[start_step:end_step, 0]
    cg_counter = cg_tracker_data[start_step:end_step, 1]
    cg_dBd = cg_tracker_data[start_step:end_step, 2]

    # Rho
    plot_epoch_data(tr_cg_epochs,
                    rho,
                    fig_size=(9, 6),
                    y_label = r"$\rho$",
                    title=fr"Evolution of $\rho$ for {fn_eqn} (Iteration {start_step} to {end_step})",
                    out_file=f"./images/{dir_date}/tr_cg_tracker/rho_{fn_name}.pdf")
    
    # CG counter
    plot_epoch_data(tr_cg_epochs,
                    cg_counter,
                    fig_size=(9, 6),
                    y_label = "CG Iterations",
                    title=fr"Number of CG Iterations for {fn_eqn} (Iteration {start_step} to {end_step})",
                    out_file=f"./images/{dir_date}/tr_cg_tracker/cg_counter_{fn_name}.pdf")
    
    # CG dBd
    plot_epoch_data(tr_cg_epochs,
                    cg_dBd,
                    fig_size=(9, 6),
                    y_label = r"$d^T B d$",
                    title=fr"Evolution of Definiteness of Hessian for {fn_eqn} (Iteration {start_step} to {end_step})",
                    out_file=f"./images/{dir_date}/tr_cg_tracker/cg_dBd_{fn_name}.pdf")

    plt.show()


def plot_cg_dBd_all():
    max_epochs = cg_dBd_data.shape[0]
    epochs = np.arange(0, max_epochs)

    cg_dBd = cg_dBd_data[0:max_epochs]

    plot_epoch_data(epochs,
                    cg_dBd,
                    fig_size=(9, 6),
                    y_label = r"$d^T B d$",
                    title=fr"Evolution of Definiteness of Hessian for {fn_eqn} (Including CG Iterations)",
                    out_file=f"./images/{dir_date}/cg_dBd_all/{fn_name}.pdf")


def get_train_times(dir, opt_name, sort_by = None):
    pattern = f"{opt_name}_*_train_time.txt"
    train_time_data = []

    # Loop through matching files
    for filename in os.listdir(dir):
        if fnmatch.fnmatch(filename, pattern):
            # Extract the function name from the filename
            parts = filename.split("_")

            if parts[0] == "adam":
                fn_name = parts[1]
            elif parts[0] == "tr":
                fn_name = parts[2]
            
            # Crude filter
            if sort_by is not None and fn_name in sort_by:
                # Read training time
                file_path = os.path.join(dir, filename)
                with open(file_path, "r") as f:
                    content = f.read().strip()
                    try:
                        train_time = float(content)
                        train_time_data.append({"fn_name": fn_name, "train_time": train_time})
                    except ValueError:
                        print(f"Warning: Could not parse time from {filename}")

    if sort_by is not None:
        order = {name: i for i, name in enumerate(sort_by)}
        train_time_data = sorted(train_time_data, key=lambda d: order.get(d["fn_name"], float("inf")))

    return [d["train_time"] for d in train_time_data]


def get_num_iters(dir, opt_name, sort_by = None):
    pattern = f"{opt_name}_loss_*.dat"
    iters_data = []

    for filename in os.listdir(dir):
        if fnmatch.fnmatch(filename, pattern):
            f_name_only = filename.split(".")[0]
            parts = f_name_only.split("_")
            fn_name = parts[-1]

            if sort_by is not None and fn_name in sort_by:
                data = np.genfromtxt(os.path.join(dir, filename), delimiter=" ", skip_header=1)
                num_iters = data.shape[0]

                iters_data.append({"fn_name": fn_name, "num_iters": num_iters})
            
    if sort_by is not None:
        order = {name: i for i, name in enumerate(sort_by)}
        iters_data = sorted(iters_data, key=lambda d: order.get(d["fn_name"], float("inf")))

    return [int(d["num_iters"]) for d in iters_data]


if __name__ == "__main__":
    fns = [
        {
            "fn": lambda x: 5 * x + 3,
            "name": "linear",
            "eqn": r"$y = 5x + 3$"
        },
        {
            "fn": lambda x: (x - 0.5) ** 2,
            "name": "quadratic",
            "eqn": r"$y = (x - 0.5)^2$"
        },
        {
            "fn": lambda x: torch.exp(-x),
            "name": "exp",
            "eqn": r"$y = \exp(-x)$"
        },
        {
            "fn": lambda x: torch.sin(x - torch.tensor(math.pi, dtype=x.dtype)),
            "name": "sinx",
            "eqn": r"$y = \sin(x - \pi)$"
        },
        {
            "fn": lambda x: torch.sin(5 * x),
            "name": "sin5x",
            "eqn": r"$y = \sin(5x)$"
        },
        {
            "fn": lambda x: torch.sin(10 * x),
            "name": "sin10x",
            "eqn": r"$y = \sin(10x)$"
        },
        {
            "fn": None,
            "name": "pinn",
            "eqn": "Heat Eq'n"
        }
    ] 

    n_data = 10000
    test_pts = torch.linspace(-1, 1, n_data).unsqueeze(1)

    dir_date = "250420"

    # Create folders
    os.makedirs(f"./images/{dir_date}/model_convergence", exist_ok=True)
    os.makedirs(f"./images/{dir_date}/loss_comparison", exist_ok=True)
    os.makedirs(f"./images/{dir_date}/tr_evolution", exist_ok=True)
    os.makedirs(f"./images/{dir_date}/grad_norm", exist_ok=True)
    os.makedirs(f"./images/{dir_date}/tr_cg_tracker", exist_ok=True)
    os.makedirs(f"./images/{dir_date}/cg_dBd_all", exist_ok=True)
    os.makedirs(f"./images/{dir_date}/performance", exist_ok=True)

    for fn in fns:
        fn_name = fn["name"]
        fn_eqn = fn["eqn"]
        fn_fn = fn["fn"]

        adam_data = np.genfromtxt(f"./out_data/{dir_date}/adam_loss_{fn_name}.dat", skip_header=1)
        tr_cg_data = np.genfromtxt(f"./out_data/{dir_date}/tr_cg_loss_{fn_name}.dat", skip_header=1)
        cg_tracker_data = np.genfromtxt(f"./out_data/{dir_date}/tr_cg_tracker_{fn_name}.dat", skip_header=1)
        cg_dBd_data = np.genfromtxt(f"./out_data/{dir_date}/cg_dBd_{fn_name}.dat", skip_header=1)

        if fn_name != "pinn":
            plot_combined_convergence(test_pts)
        plot_loss_comparison(end_step=15000)
        plot_tr_rad()
        plot_grad_norms()
        plot_tr_cg_tracker(end_step=500)
        plot_cg_dBd_all()

    # For performance comparison
    fn_names = [fn["name"] for fn in fns]
    fn_eqns = [fn["eqn"] for fn in fns]

    # Training time comparison
    adam_train_times = get_train_times(f"./out_data/{dir_date}", opt_name="adam", sort_by=fn_names)
    tr_cg_train_times = get_train_times(f"./out_data/{dir_date}", opt_name="tr_cg", sort_by=fn_names)
    plot_time_comparison(fn_eqns,
                         adam_train_times,
                         tr_cg_train_times,
                         y_label = "Training time (s)",
                         title = "Training Time by Optimizer and Function",
                         out_file = f"./images/{dir_date}/performance/time_comparison.pdf")

    # Number of iterations comparison
    adam_iters = get_num_iters(f"./out_data/{dir_date}", opt_name="adam", sort_by=fn_names)
    tr_cg_iters = get_num_iters(f"./out_data/{dir_date}", opt_name="tr_cg", sort_by=fn_names)
    plot_time_comparison(fn_eqns,
                         adam_iters,
                         tr_cg_iters,
                         y_label = "Number of iterations",
                         title = "Number of Iterations by Optimizer and Function",
                         out_file = f"./images/{dir_date}/performance/iters_comparison.pdf")