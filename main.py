import torch
import torch.optim as optim
import numpy as np
from scipy.stats.qmc import LatinHypercube
import time
import os
import math

from src.fcn import FCN
from src.geom import generate_domain_pts
from src.loss import mse_loss
from src.optim import TrustRegionNewtonCG
from src.trainers import BasicTrainer, HeatPINNTrainer

print("Starting...")
print(f"Using GPU: {torch.cuda.is_available()}")

# Set device - use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use 64 bit floating point
torch.set_default_dtype(torch.float64)

# General RNG
rng = np.random.default_rng(42) 

# Training switches
train_adam_fns = False
train_tr_fns = False

train_adam_pinn = False
train_tr_pinn = True
train_combo_pinn = True

dir_date = "250420"

"""
Basic NN training (not PINN)
"""
if train_adam_fns or train_tr_fns:
    print("Starting functional training segment...")
    print()

    print("Defining shared parameters...")
    n_data = 10000
    x_test = torch.linspace(-1, 1, n_data, device=device).unsqueeze(1)

    train_fns = [
        {
            "fn": lambda x: 5 * x + 3,
            "name": "linear",
            "eqn": r"$y = 5x + 3$"
        },
        {
            "fn": lambda x: (x - 0.5) ** 2,
            "test": (x_test - 0.5) ** 2,
            "name": "quadratic",
            "eqn": r"$y = (x - 0.5)^2$"
        },
        {
            "fn": lambda x: torch.exp(-x),
            "name": "exp",
            "eqn": r"$y = \exp(-x)$"
        },
        {
            "fn": lambda x: torch.sin(x - torch.tensor(math.pi, device=x.device, dtype=x.dtype)),
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
    ]

    for train_fn in train_fns:
        # x ~ U(-1.1, 1.1) (0.1 margin so we can test (-1, 1))
        x_data = (torch.rand((n_data, 1), device=device) * 2.2) - 1.1

        y_target = train_fn["fn"](x_data).to(device)
        y_true = train_fn["fn"](x_test).to(device)
        fn_name = train_fn["name"]
        fn_eqn = train_fn["eqn"]

        # Make image folders
        os.makedirs(f"./images/adam_{fn_name}", exist_ok=True)
        os.makedirs(f"./images/tr_cg_{fn_name}", exist_ok=True)

        """
        ADAM
        """
        if train_adam_fns:
            print(f"Defining ADAM FCN for {fn_name}...")
            fcn_adam_basic = FCN(1, 10, 10, 1).to(device)
            print("Total weights: ", sum(p.numel() for name, p in fcn_adam_basic.named_parameters() if "weight" in name))
            print("Total parameters:", sum(p.numel() for p in fcn_adam_basic.parameters()))

            adam_opt_basic = optim.Adam(fcn_adam_basic.parameters(), lr=1e-3)

            max_epochs_adam = 25001
            basic_adam = BasicTrainer(nn=fcn_adam_basic,
                                      opt=adam_opt_basic,
                                      data_initial=x_data,
                                      target=y_target,
                                      test_data=x_test,
                                      y_true=y_true,
                                      loss_fn=mse_loss,
                                      max_epochs=max_epochs_adam,
                                      train_loss_file=f"./out_data/adam_loss_{fn_name}.dat",
                                      test_plot_out_dir=f"./images/adam_{fn_name}",
                                      model_out_file=f"./models/adam_model_{fn_name}.pt")

            print(f"Starting ADAM training for {fn_name}...")
            start_time = time.perf_counter()
            basic_adam.train(print_interval=1000,
                             test_interval=1000,
                             fn_eqn=fn_eqn,
                             plot_title=f"Model convergence to {fn_eqn} (Adam)")
            end_time = time.perf_counter()

            adam_basic_train_time =  end_time - start_time
            print(f"ADAM training took {adam_basic_train_time}")

            with open(f"./out_data/adam_{fn_name}_train_time.txt", "w") as f:
                f.write(f"{adam_basic_train_time}\n")


        """
        TR Newton-CG
        """
        if train_tr_fns:
            print(f"Defining TR Newton-CG FCN for {fn_name}...")
            fcn_tr_cg_basic = FCN(1, 10, 10, 1).to(device)
            print("Total weights: ", sum(p.numel() for name, p in fcn_tr_cg_basic.named_parameters() if "weight" in name))
            print("Total parameters:", sum(p.numel() for p in fcn_tr_cg_basic.parameters()))

            tr_cg_opt_basic = TrustRegionNewtonCG(model=fcn_tr_cg_basic,
                                                  tr_init=2,
                                                  tr_max=10.,
                                                  eta=1e-3,
                                                  cg_tol=1e-8,
                                                  cg_max_iter=500)

            max_epochs_tr_cg = 10001
            basic_tr_cg = BasicTrainer(nn=fcn_tr_cg_basic,
                                       opt=tr_cg_opt_basic,
                                       tr_optim=True,
                                       data_initial=x_data,
                                       target=y_target,
                                       test_data=x_test,
                                       y_true=y_true,
                                       loss_fn=mse_loss,
                                       max_epochs=max_epochs_tr_cg,
                                       train_loss_file=f"./out_data/tr_cg_loss_{fn_name}.dat",
                                       test_plot_out_dir=f"./images/tr_cg_{fn_name}",
                                       model_out_file=f"./models/tr_cg_model_{fn_name}.pt",
                                       cg_dBd_out_file=f"./out_data/cg_dBd_{fn_name}.dat",
                                       opt_track_out_file=f"./out_data/tr_cg_tracker_{fn_name}.dat")

            print(f"Starting TR Newton-CG training for {fn_name}...")
            start_time = time.perf_counter()
            basic_tr_cg.train(print_interval=1000,
                              test_interval=1000,
                              fn_eqn=fn_eqn,
                              plot_title=f"Model convergence to {fn_eqn} (TR Newton-CG)")
            end_time = time.perf_counter()

            tr_cg_basic_train_time =  end_time - start_time
            print(f"TR Newton-CG training took {tr_cg_basic_train_time}")

            with open(f"./out_data/tr_cg_{fn_name}_train_time.txt", "w") as f:
                f.write(f"{tr_cg_basic_train_time}\n")

    print("Finished functional training!")


"""
Start of PINN training
"""
if train_adam_pinn or train_tr_pinn or train_combo_pinn:
    print("Starting PINN segment...")
    print()
    print("Setting PDE parameters...")

    """
    PDE parameters
    """
    bdry_conds = [1, 0, 0, 1] # Order is clockwise from top
    initial_cond = 0.
    time_range = [0, 5]

    # Forcing (one per subdomain)
    q = [0, 0]

    # Diffusivity
    alpha = 1

    # Move to correct device
    bdry_conds = [torch.tensor(val, device=device) for val in bdry_conds]
    initial_cond = torch.tensor(initial_cond, device=device)
    q = [torch.tensor(val, device=device) for val in q]


    """
    Generate collocation points in domain
    """
    print("Generating collocation poitns...")

    # Generate 8000 domain points and 2000 boundary points for a total of 10000 points
    # Each point is a vector of 3 elements (x, y, t)
    dom_set, bdry_set = generate_domain_pts(n_dom=8000, n_bdry=2000, device=device)


    """
    Shared neural network parameters
    """
    print("Defining NN parameters...")

    # Regularization parameter
    lambda1 = 0.775

    # Test points
    n_test = 100000
    lhs_2d = LatinHypercube(d=2, rng=rng)
    test_pts = torch.tensor(lhs_2d.random(n_test), dtype=torch.float, device=device)

    """
    ADAM
    """
    if train_adam_pinn:
        print("Training PINN with ADAM...")

        # Make image folder
        os.makedirs("./images/adam_pinn", exist_ok=True)

        print("Defining FCN...")
        # FCN with 4 hidden layers of 24 neurons each
        # Total weights: 2400; total parameters: 2521
        fcn_adam = FCN(3, 24, 4, 1).to(device)
        print("Total weights: ", sum(p.numel() for name, p in fcn_adam.named_parameters() if "weight" in name))
        print("Total parameters:", sum(p.numel() for p in fcn_adam.parameters()))

        adam_opt = optim.Adam(fcn_adam.parameters(), lr=1e-3)

        # Initialize and train
        max_epochs_adam = 15001
        pinn_adam = HeatPINNTrainer(nn=fcn_adam,
                                    opt=adam_opt,
                                    bdry_conds=bdry_conds,
                                    initial_cond=initial_cond,
                                    pde_forcings=q,
                                    alpha=alpha,
                                    bdry_set=bdry_set,
                                    dom_set=dom_set,
                                    time_range=time_range,
                                    test_pts=test_pts,
                                    loss_fn=mse_loss,
                                    lamb=lambda1,
                                    max_epochs=max_epochs_adam,
                                    train_loss_file="./out_data/adam_loss_pinn.dat",
                                    model_out_file="./models/adam_model_pinn.pt",
                                    test_plot_out_dir="./images/adam_pinn/")

        print("Starting training...")
        start_time = time.perf_counter()
        pinn_adam.train(print_interval=1000, test_interval=1000, test_time=1, opt="ADAM")
        end_time = time.perf_counter()

        adam_train_time =  end_time - start_time
        print(f"ADAM training took {adam_train_time}")

        with open("./out_data/adam_pinn_train_time.txt", "w") as f:
            f.write(f"{adam_train_time}\n")


    """
    TR Newton-CG
    """
    if train_tr_pinn:
        print("Training PINN with TR Newton-CG...")

        # Make image folder
        os.makedirs(f"./images/{dir_date}/tr_cg_pinn", exist_ok=True)

        print("Defining FCN...")
        # FCN with 4 hidden layers of 24 neurons each
        fcn_tr_cg = FCN(3, 24, 4, 1).to(device)
        print("Total weights: ", sum(p.numel() for name, p in fcn_tr_cg.named_parameters() if "weight" in name))
        print("Total parameters:", sum(p.numel() for p in fcn_tr_cg.parameters()))

        tr_cg_opt = TrustRegionNewtonCG(fcn_tr_cg,
                                        tr_init=1,
                                        tr_max=5.,
                                        eta=5e-5,
                                        cg_tol=1e-10,
                                        cg_max_iter=2000)

        # Initialize and train
        max_epochs_tr_cg = 15001
        pinn_tr_cg = HeatPINNTrainer(nn=fcn_tr_cg,
                                     opt=tr_cg_opt,
                                     tr_optim=True,
                                     bdry_conds=bdry_conds,
                                     initial_cond=initial_cond,
                                     pde_forcings=q,
                                     alpha=alpha,
                                     bdry_set=bdry_set,
                                     dom_set=dom_set,
                                     time_range=time_range,
                                     test_pts=test_pts,
                                     loss_fn=mse_loss,
                                     lamb=lambda1,
                                     max_epochs=max_epochs_tr_cg,
                                     train_loss_file=f"./out_data/{dir_date}/tr_cg_loss_pinn.dat",
                                     model_out_file=f"./models/{dir_date}/tr_cg_model_pinn.pt",
                                     test_plot_out_dir=f"./images/{dir_date}/tr_cg_pinn/",
                                     cg_dBd_out_file=f"./out_data/{dir_date}/cg_dBd_pinn.dat",
                                     opt_track_out_file=f"./out_data/{dir_date}/tr_cg_tracker_pinn.dat")

        print("Starting training...")
        start_time = time.perf_counter()
        pinn_tr_cg.train(print_interval=1000, test_interval=1000, test_time=1, opt="TR Newton-CG")
        end_time = time.perf_counter()

        tr_cg_train_time = end_time - start_time
        print(f"TR Newton-CG training took {tr_cg_train_time}")

        with open(f"./out_data/{dir_date}/tr_cg_pinn_train_time.txt", "w") as f:
            f.write(f"{tr_cg_train_time}\n")


    """
    ADAM + TR Newton CG for PINN

    Train with ADAM for 2001 iterations,
    then train with TR Newton-CG for another 10001 iterations
    """
    if train_combo_pinn:
        print("Combination training segment...")
        print()

        # Make image folder
        os.makedirs(f"./images/{dir_date}/combo_pinn/adam", exist_ok=True)
        os.makedirs(f"./images/{dir_date}/combo_pinn/tr_cg", exist_ok=True)

        print("Training PINN with ADAM...")

        print("Defining FCN...")
        fcn_combo = FCN(3, 24, 4, 1).to(device)
        print("Total weights: ", sum(p.numel() for name, p in fcn_combo.named_parameters() if "weight" in name))
        print("Total parameters:", sum(p.numel() for p in fcn_combo.parameters()))

        adam_opt = optim.Adam(fcn_combo.parameters(), lr=1e-3)

        # Train for 2001 iterations only
        pinn_combo_adam = HeatPINNTrainer(nn=fcn_combo,
                                          opt=adam_opt,
                                          bdry_conds=bdry_conds,
                                          initial_cond=initial_cond,
                                          pde_forcings=q,
                                          alpha=alpha,
                                          bdry_set=bdry_set,
                                          dom_set=dom_set,
                                          time_range=time_range,
                                          test_pts=test_pts,
                                          loss_fn=mse_loss,
                                          lamb=lambda1,
                                          max_epochs=2001,
                                          train_loss_file=f"./out_data/{dir_date}/combo_adam_loss_pinn.dat",
                                          model_out_file=f"./models/{dir_date}/combo_adam_model_pinn.pt",
                                          test_plot_out_dir=f"./images/{dir_date}/combo_pinn/adam/")
            
        print("Starting training...")
        start_time = time.perf_counter()
        pinn_combo_adam.train(print_interval=1000, test_interval=1000, test_time=1, opt="Adam")
        end_time = time.perf_counter()

        adam_train_time =  end_time - start_time
        print(f"ADAM training took {adam_train_time}")

        with open(f"./out_data/{dir_date}/combo_adam_pinn_train_time.txt", "w") as f:
            f.write(f"{adam_train_time}\n")

        tr_cg_opt = TrustRegionNewtonCG(fcn_combo,
                                        tr_init=1,
                                        tr_max=5.,
                                        eta=5e-5,
                                        cg_tol=1e-10,
                                        cg_max_iter=2000)

        # Train for 2001 iterations only
        pinn_combo_tr_cg = HeatPINNTrainer(nn=fcn_combo,
                                           opt=tr_cg_opt,
                                           tr_optim=True,
                                           bdry_conds=bdry_conds,
                                           initial_cond=initial_cond,
                                           pde_forcings=q,
                                           alpha=alpha,
                                           bdry_set=bdry_set,
                                           dom_set=dom_set,
                                           time_range=time_range,
                                           test_pts=test_pts,
                                           loss_fn=mse_loss,
                                           lamb=lambda1,
                                           max_epochs=2001,
                                           train_loss_file=f"./out_data/{dir_date}/combo_tr_cg_loss_pinn.dat",
                                           model_out_file=f"./models/{dir_date}/combo_tr_cg_model_pinn.pt",
                                           test_plot_out_dir=f"./images/{dir_date}/combo_pinn/tr_cg")

        print("Starting training...")
        start_time = time.perf_counter()
        pinn_combo_tr_cg.train(print_interval=1000, test_interval=1000, test_time=1, opt="Adam + TR Newton-CG")
        end_time = time.perf_counter()

        tr_cg_train_time = end_time - start_time
        print(f"TR Newton-CG training took {tr_cg_train_time}")

        with open(f"./out_data/{dir_date}/combo_tr_cg_pinn_train_time.txt", "w") as f:
            f.write(f"{tr_cg_train_time}\n")

    print("Finished!")