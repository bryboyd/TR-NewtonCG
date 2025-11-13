import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import math

from src.utils import save_data
from src.plotting import plot_pinn_test, plot_basic_test
from types import FunctionType
from typing import List

class BasicTrainer():
    def __init__(self,
                 nn: nn.Module = None,
                 opt: optim.Optimizer = None,
                 tr_optim: bool = False,
                 data_initial: torch.Tensor = None,
                 target: torch.Tensor = None,
                 test_data: torch.Tensor = None,
                 y_true: torch.Tensor = None,
                 loss_fn: FunctionType = None,
                 loss_thresh: float = 1e-6,
                 max_epochs: int = 0,
                 train_loss_file: str = "",
                 model_out_file: str = "",
                 test_plot_out_dir: str = "",
                 opt_track_out_file: str = "",
                 cg_dBd_out_file: str = ""):
        self.nn = nn
        self.opt = opt
        self.tr_optim = tr_optim
        self.data = data_initial
        self.target = target
        self.test_data = test_data
        self.y_true = y_true
        self.loss_fn = loss_fn
        self.loss_thresh = loss_thresh
        self.max_epochs = max_epochs
        self.train_loss_file = train_loss_file
        self.model_out_file = model_out_file
        self.test_plot_out_dir = test_plot_out_dir
        self.opt_track_out_file = opt_track_out_file
        self.cg_dBd_out_file = cg_dBd_out_file

        self.epoch  = 0
        if self.tr_optim:
            self.loss_hist = {"epoch": [], "loss": [], "grad_norm": [], "tr": []}
        else:
            self.loss_hist = {"epoch": [], "loss": [], "grad_norm": []}


    @property
    def device(self):
        # Get device
        return next(self.nn.parameters()).device

    
    def reset_loss_hist(self) -> None:
        if self.tr_optim:
            self.loss_hist = {"epoch": [], "loss": [], "grad_norm": [], "tr": []}
        else:
            self.loss_hist = {"epoch": [], "loss": [], "grad_norm": []}

    
    def train(self,
              print_interval: int = 0,
              test_interval: int = 0,
              plot_test = True,
              fn_eqn = "",
              plot_title = "") -> None:
        self._loss = 0
        self._grad_norm = 0

        # Enforce for device safety
        self.data = self.data.to(self.device)
        self.target = self.target.to(self.device)
        if self.test_data is not None:
            self.test_data = self.test_data.to(self.device)

        # Reset our tracker
        self.reset_loss_hist()

        for self.epoch in range(self.max_epochs):
            # Training mode
            self.nn.train()

            def closure(out=None, backward=True):
                self.opt.zero_grad()

                if out is None:
                    out = self.nn(self.data)

                loss = self.loss_fn(out, self.target)
                self._loss = loss.detach().item()

                if backward:
                    loss.backward()

                    # Compute L2 norm of gradient
                    g_norm_sum_sq = 0.
                    for param in self.nn.parameters():
                        if param.grad is not None:
                            g_norm_sum_sq += param.grad.norm().item() ** 2
                    self._grad_norm = g_norm_sum_sq ** 0.5

                return loss

            if self.tr_optim:
                self.opt.step(closure=closure, model_args=self.data)
            else:
                self.opt.step(closure=closure)

            self.loss_hist["epoch"].append(self.epoch)
            self.loss_hist["loss"].append(self._loss)
            self.loss_hist["grad_norm"].append(self._grad_norm)
            if self.tr_optim:
                self.loss_hist["tr"].append(self.opt.tr)

            if print_interval > 0 and self.epoch % print_interval == 0:
                if self.tr_optim:
                    tr_rho = self.opt.tracker["rho"][-1]
                    cg_counter = self.opt.tracker["cg_counter"][-1]
                    cg_dBd = self.opt.tracker["cg_dBd"][-1]

                    print(f"Epoch: {self.epoch} | Loss: {self._loss} | Grad norm: {self._grad_norm} | TR radius: {self.opt.tr}")
                    print(f"Last TR rho: {tr_rho} | Last CG counter: {cg_counter} | Last CG dBd: {cg_dBd}")
                else:
                    print(f"Epoch: {self.epoch} | Loss: {self._loss} | Grad norm: {self._grad_norm}")
                print()

            # Test and plot if given directory and we don't turn plot_test off
            if test_interval > 0 and self.epoch % test_interval == 0 and self.test_plot_out_dir and plot_test:
                self.test_and_plot(plot_title, fn_eqn)

            # Stop iteration after threshold
            if self._loss < self.loss_thresh or math.isclose(self.loss_thresh, self._loss):
                # Plot too so we have latest version
                if self.test_plot_out_dir and plot_test:
                    self.test_and_plot(plot_title, fn_eqn)
                break

        # Save model and tracker at the end of training if paths are provided
        if self.tr_optim and  self.opt_track_out_file:
            sub_keys = ["rho", "cg_counter", "cg_dBd"]
            sub_tracker = {k: self.opt.tracker[k] for k in sub_keys if k in self.opt.tracker}
            save_data(data=sub_tracker,
                      out_file=self.opt_track_out_file)
        if self.tr_optim and self.cg_dBd_out_file:
            sub_keys = ["cg_dBd_all"]
            sub_tracker = {k: self.opt.tracker[k] for k in sub_keys if k in self.opt.tracker}
            save_data(data=sub_tracker,
                      out_file=self.cg_dBd_out_file)
        if self.train_loss_file:
            save_data(data=self.loss_hist,
                      out_file=self.train_loss_file)
        if self.model_out_file:
            torch.save(self.nn.state_dict(), self.model_out_file)


    def test_and_plot(self, title, eqn):
        self.nn.eval()
        with torch.no_grad():
            x_test = self.test_data.to(self.device)
            y_hat = self.nn(x_test).detach().cpu().numpy().flatten()
            x_test_np = x_test.cpu().numpy().flatten()
            y_true_np = self.y_true.cpu().numpy().flatten()

            sort_idx = np.argsort(x_test_np)
            x_sorted = x_test_np[sort_idx]
            y_true_sorted = y_true_np[sort_idx]
            y_hat_sorted = y_hat[sort_idx]

            plot_out_file = os.path.join(self.test_plot_out_dir, f"test_plot_epoch_{self.epoch}.pdf")
            plot_basic_test(x_sorted,
                            y_true_sorted,
                            y_hat_sorted,
                            eqn=eqn,
                            title=title,
                            out_file=plot_out_file)


class HeatPINNTrainer():
    def __init__(self,
                 nn: nn.Module = None,
                 opt: optim.Optimizer = None,
                 tr_optim: bool = False,
                 bdry_conds: List = None,
                 initial_cond: float = None,
                 pde_forcings: List = None,
                 alpha: float = 1.0,
                 bdry_set: torch.tensor = None,
                 dom_set: torch.tensor = None,
                 time_range: List = [0, 1],
                 test_pts: torch.tensor = None,
                 loss_fn: FunctionType = None,
                 lamb: float = 0.1,
                 max_epochs: int = None,
                 train_loss_file: str = "",
                 model_out_file: str = "",
                 test_plot_out_dir: str = "",
                 opt_track_out_file: str = "",
                 cg_dBd_out_file: str = ""):
        self.nn = nn
        self.opt = opt
        self.tr_optim = tr_optim
        self.bdry_conds = bdry_conds
        self.initial_cond = initial_cond
        self.pde_forcings = pde_forcings
        self.alpha = alpha
        self.bdry_set = bdry_set
        self.dom_set = dom_set
        self.time_range = time_range
        self.test_pts = test_pts
        self.loss_fn = loss_fn
        self.lamb = lamb
        self.max_epochs = max_epochs
        self.train_loss_file = train_loss_file
        self.model_out_file = model_out_file
        self.test_plot_out_dir = test_plot_out_dir
        self.opt_track_out_file = opt_track_out_file
        self.cg_dBd_out_file = cg_dBd_out_file

        self.epoch  = 0

        if self.tr_optim:
            self.loss_hist = {"epoch": [], "bdry": [], "dom": [], "total": [], "tr": []}
        else:
            self.loss_hist = {"epoch": [], "bdry": [], "dom": [], "total": []}


    @property
    def device(self):
        # Get device
        return next(self.nn.parameters()).device


    def reset_loss_hist(self) -> None:
        if self.tr_optim:
            self.loss_hist = {"epoch": [], "bdry": [], "dom": [], "total": [], "tr": []}
        else:
            self.loss_hist = {"epoch": [], "bdry": [], "dom": [], "total": []}

    
    def create_time_pts(self, size: int, t_type: str = "range") -> torch.tensor:
        if t_type == "zeros":
            return torch.zeros(size, device=self.device).unsqueeze(1)
        elif t_type == "ones":
            return torch.ones(size, device=self.device).unsqueeze(1)
        elif t_type == "range":
            start = self.time_range[0]
            end = self.time_range[1]
            step = (end - start) / size

            return torch.arange(start, end, step, device=self.device, requires_grad=True).unsqueeze(1)
        else:
            raise ValueError("t_type only accepts 'zeros', 'ones', or 'range'.")

    
    def _get_pde_residuals(self):
        residuals = []
        for dom_pts in self.dom_set:
            time_pts = self.create_time_pts(dom_pts.shape[0])
            dom_with_time = torch.hstack((dom_pts, time_pts)).requires_grad_()

            u_dom = self.nn(dom_with_time)

            grad_u = torch.autograd.grad(u_dom.sum(), dom_with_time, create_graph=True)[0]
            u_x, u_y, u_t = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]

            u_xx = torch.autograd.grad(u_x.sum(), dom_with_time, create_graph=True)[0][:, 0]
            u_yy = torch.autograd.grad(u_y.sum(), dom_with_time, create_graph=True)[0][:, 1]

            residual = u_t - self.alpha * (u_xx + u_yy)
            residuals.append(residual)

        return residuals


    def train(self,
              print_interval: int = 0,
              test_interval: int = 0,
              test_time: float = 1,
              plot_test: bool = True,
              opt: str = "") -> None:

        # Reset our tracker
        self.reset_loss_hist()

        # Make sure points are in specified device
        self.initial_cond = self.initial_cond.to(self.device)
        self.bdry_conds = [bc.to(self.device) for bc in self.bdry_conds]
        self.pde_forcings = [pf.to(self.device) for pf in self.pde_forcings]

        for self.epoch in range(self.max_epochs):
            # Enable training mode
            self.nn.train()

            # For tracking
            self._loss_bdry = 0
            self._loss_dom = 0
            self._loss_total = 0

            def closure(out=None, backward=True):
                """
                This function is called by optimizer and does:
                - Zero out gradients
                - Forward propagation
                    - Steps that look like self.nn(points)
                - Compute loss between output and target for all parts of PDE
                    - Steps that look like self.loss(a, b)
                - Compute gradients with backpropagation
                    - loss.backward()
                - Return computed loss
                """
                self.opt.zero_grad()

                # Boundary loss
                loss_bdry = torch.zeros(1, device=self.device)
                for i, bdry_pts in enumerate(self.bdry_set):
                    time_pts = self.create_time_pts(bdry_pts.shape[0])
                    input_pts = torch.hstack((bdry_pts, time_pts))
                    u_bdry = self.nn(input_pts)  # always forward here
                    loss_bdry += self.loss_fn(u_bdry, self.bdry_conds[i])

                # Initial condition loss
                loss_time = torch.zeros(1, device=self.device)
                for i, dom_pts in enumerate(self.dom_set):
                    zero_time = self.create_time_pts(dom_pts.shape[0], t_type="zeros")
                    dom_time_pts = torch.hstack((dom_pts, zero_time))
                    u_time = self.nn(dom_time_pts)  # always forward here
                    loss_time += self.loss_fn(u_time, self.initial_cond)

                # PDE residual loss
                loss_pde = torch.zeros(len(self.dom_set), device=self.device)
                for i, dom_pts in enumerate(self.dom_set):
                    time_pts = self.create_time_pts(dom_pts.shape[0])
                    dom_with_time = torch.hstack((dom_pts, time_pts)).requires_grad_()

                    if out is None:
                        u_dom = self.nn(dom_with_time)

                        grad_u = torch.autograd.grad(u_dom.sum(), dom_with_time, create_graph=True)[0]
                        u_x, u_y, u_t = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]

                        u_xx = torch.autograd.grad(u_x.sum(), dom_with_time, create_graph=True)[0][:, 0]
                        u_yy = torch.autograd.grad(u_y.sum(), dom_with_time, create_graph=True)[0][:, 1]

                        residual = u_t - self.alpha * (u_xx + u_yy)
                    else:
                        residual = out[i]  # precomputed residual (not u_dom)

                    loss_pde[i] = self.loss_fn(residual, self.pde_forcings[i])

                # Total loss
                loss = self.lamb * (loss_bdry + loss_time) + (1 - self.lamb) * loss_pde.sum()

                if backward:
                    loss.backward()

                # Tracker
                self._loss_bdry = loss_bdry.item() + loss_time.item()
                self._loss_dom = loss_pde.sum().item()
                self._loss_total = loss.item()

                return loss 
            
            if self.tr_optim:
                model_input = torch.vstack([
                    torch.hstack((dom_pts, self.create_time_pts(dom_pts.shape[0])))
                    for dom_pts in self.dom_set
                ]).detach().requires_grad_()

                self.opt.step(closure=closure, model_args=(model_input,))
            else:
                self.opt.step(closure=closure)

            # Update tracker
            self.loss_hist["epoch"].append(self.epoch)
            self.loss_hist["bdry"].append(self._loss_bdry)
            self.loss_hist["dom"].append(self._loss_dom)
            self.loss_hist["total"].append(self._loss_total)
            if self.tr_optim:
                self.loss_hist["tr"].append(self.opt.tr)

            if print_interval > 0 and self.epoch % print_interval == 0:
                if self.tr_optim:
                    print(f"Epoch: {self.epoch} | Boundary loss: {self._loss_bdry} | PDE loss: {self._loss_dom} | Total loss: {self._loss_total} | TR radius: {self.opt.tr}")
                else:
                    print(f"Epoch: {self.epoch} | Boundary loss: {self._loss_bdry} | PDE loss: {self._loss_dom} | Total loss: {self._loss_total}")

            # Test and plot if given directory and we don't turn plot_test off
            if test_interval > 0 and self.epoch % test_interval == 0 and self.test_plot_out_dir and plot_test:
                self.test_and_plot(test_time=test_time, opt=opt)

            # Save model and tracker at the end of training if paths are provided
            if self.tr_optim and  self.opt_track_out_file:
                sub_keys = ["rho", "cg_counter", "cg_dBd"]
                sub_tracker = {k: self.opt.tracker[k] for k in sub_keys if k in self.opt.tracker}
                save_data(data=sub_tracker,
                          out_file=self.opt_track_out_file)
            if self.tr_optim and self.cg_dBd_out_file:
                sub_keys = ["cg_dBd_all"]
                sub_tracker = {k: self.opt.tracker[k] for k in sub_keys if k in self.opt.tracker}
                save_data(data=sub_tracker,
                          out_file=self.cg_dBd_out_file)
            if self.train_loss_file:
                save_data(data=self.loss_hist,
                          out_file=self.train_loss_file)
            if self.model_out_file:
                torch.save(self.nn.state_dict(), self.model_out_file)


    def test_and_plot(self, test_time: float = 1, opt: str = ""):
        # Turn on eval mode
        self.nn.eval()
        with torch.inference_mode():
            t_pts = test_time * self.create_time_pts(self.test_pts.shape[0], t_type="ones")
            test_t_pts = torch.hstack((self.test_pts.to(self.device), t_pts))
            u_test = self.nn(test_t_pts)

            plot_out_file = os.path.join(self.test_plot_out_dir, f"test_plot_epoch_{self.epoch}.pdf")

            plot_pinn_test(test_pts=test_t_pts,
                           test_vals=u_test,
                           title=rf"Approximated $T(x, t)$ at $t = {test_time}$ ({opt})",
                           out_file=plot_out_file)
