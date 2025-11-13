import torch
from torch.func import functional_call
import math
from torch.optim import Optimizer

class TrustRegionNewtonCG(Optimizer):
    def __init__(self,
                 model,
                 tr_init = 1.,
                 tr_max = 5.,
                 eta = 1e-2,
                 cg_tol = 1e-8,
                 cg_max_iter = 1000):
        """
        Inputs:
        model       - Model
        tr_init     - Initial TR radius in (0, tr_max)
        tr_max      - Maximum TR radius > 0
        eta         - TR accept/reject threshold in [0, 1/4)
        cg_tol      - Newton-CG ||r|| tolerance
        cg_max_iter - Newton-CG max iterations
        """
        # TR parameters assertions
        assert tr_max > 0, "Maximum radius needs to be greater than 0"
        assert 0 < tr_init < tr_max, "Initial radius needs to be in (0, tr_max)"
        assert 0 <= eta < 0.25, "Accept/reject threshold must be in [0, 1/4)"

        # CG-Steihaug parameter assertion
        assert cg_tol > 0, "CG tolerance should be greater than 0"
        
        super().__init__(model.parameters(), {})

        self.model = model
        self.tr = tr_init
        self.tr_max = tr_max
        self.eta = eta
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter

        self.model_args = None
        self.tracker = {
            "rho": [], # rho per iteration
            "cg_counter": [], # how many times inner loop runs for per epoch
            "cg_dBd": [], # positive definiteness per epoch
            "cg_dBd_all": [] # positive definiteness for all iterations per epoch
        }

   
    def _get_hvp(self, p, closure):
        """
        Compute Hessian-vector product using torch's vhp(),
        which computes the vector-Hessian product (faster than hvp()).
        We enforce that p is 1D vector (i.e., just an array, not row or column),
        so hvp = vhp. We further assume that f is in C2.
        """
        # Flatten current parameter vector (x), and enable autograd
        x = self._get_params().detach().requires_grad_()
        p = p.detach()

        # Cache parameter structure
        named_params = dict(self.model.named_parameters())
        param_shapes = {k: v.shape for k, v in named_params.items()}

        def unflatten_to_param_dict(x_vec):
            """Convert flat vector to {name: tensor} dict for functional_call"""
            out = {}
            offset = 0
            for name, shape in param_shapes.items():
                numel = torch.tensor(shape).prod().item()
                out[name] = x_vec[offset:offset + numel].view(shape)
                offset += numel
            return out

        # Define loss function that depends on flat x
        def loss_fn(x_vec):
            params = unflatten_to_param_dict(x_vec)
            output = functional_call(self.model, params, args=self.model_args)
            return closure(output, backward=False)

        # Compute Hessian-vector product
        _, hvp = torch.autograd.functional.vhp(loss_fn, x, p)

        return hvp


    def _add_step(self, step):
        """
        Add step (p_k) to parameters;
        i.e. x_p = x_k + p_k
        So we when we run closure(), we'll obtain f(x_p)
        """
        offset = 0
        for group in self.param_groups:
            for param in group["params"]:
                numel = param.numel()
                start = offset
                stop = start + numel

                # Extract this param's slice from step
                param_step = step[start:stop].view_as(param)
                # Add to param
                param.data.add_(param_step)
                offset += numel


    def _set_params(self, x_vals):
        """
        Set model params to x_vals
        """
        offset = 0
        for group in self.param_groups:
            for param in group["params"]:
                numel = param.numel()
                start = offset
                stop = start + numel

                # Set parameter
                # param.data.copy_(x_vals[start:stop].view_as(param))
                param.data.copy_(x_vals[start:stop].view_as(param))
                offset += numel


    def _get_params(self):
        """
        Get model parameters

        Optimizer has access to model parameters through
        group["params"] in self.param_groups
        """
        flattened_params = []
        for group in self.param_groups:
            for param in group["params"]:
                # .view(-1) flattens the tensor into a 1D vector
                flattened_params.append(param.data.view(-1))

        # Return 1D vector Tensor instead of List
        return torch.cat(flattened_params)

    
    def _get_grad(self):
        """
        Get model gradients

        Optimizer has access to model gradients through
        group["params"].grad in self.param_groups
        """
        flattened_grads = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    # .view(-1) flattens the tensor into a 1D vector
                    flattened_grads.append(param.grad.view(-1))
                else:
                    flattened_grads.append(torch.zeros_like(param).view(-1))

        # Return 1D vector Tensor instead of List
        return torch.cat(flattened_grads)


    def step(self, closure, model_args):
        """
        This is our implementation of the Trust Region algorithm
        (Algorithm 4.1 in Nocedal & Wright). One difference with Alg 4.1
        is that step() is called once per iteration of model training,
        so we don't iterate here (i.e., model training is iterating for us)

        `closure` is a function that is passed in during model training
        and it computes the model's loss; i.e. f(x) = loss = closure(),
        where x is the current parameters.

        The TR subproblem is solved with CG Steihaug (self.cg_steihaug())
        """
        self.model_args = model_args

        # Quadratic model
        def quad_model(p, Bp):
            quad = f_x + g_x @ p + 0.5 * p @ Bp

            # Get float in case quad returns Tensor
            if isinstance(quad, torch.Tensor) and quad.dim() == 0:
                quad = quad.item()
            return quad

        # Compute f(x) and save a copy of it as f_x
        f_x = closure().item()
        g_x = self._get_grad() # Returns 1D vector as Tensor object

        # Get copy of original parameters
        x_current = self._get_params().clone()

        # Get step from CG Steihaug (step = p)
        step = self.cg_steihaug(grad=g_x, closure=closure)

        # Compute Hessian-vector product
        hessian_times_p = self._get_hvp(step, closure)

        # Add step to model parameters and compute f_p = f(x + step)
        self._add_step(step)
        f_p = closure().item()

        # Evaluate quadratic model at 0 and p
        m_0 = f_x # m(0) = f(x)
        m_p = quad_model(step, hessian_times_p)
        denom = m_0 - m_p

        # In case m_0 - m_p == 0, let rho = infty
        if abs(denom) < 1e-8:
            rho = float("inf")

        # Otherwise compute reduction ratio (actual / predicted)
        else:
            rho = (f_x - f_p) / denom

        # Track rho
        self.tracker["rho"].append(rho)

        # Update trust region radius
        # The last else sets tr_next = tr_current,
        # so we just don't update self.tr
        step_norm = torch.linalg.norm(step).item()
        if rho < 0.25:
            self.tr = 0.25 * step_norm

        # Use .isclose() to check approx. equality
        elif rho > 0.75 and math.isclose(step_norm, self.tr):
            self.tr = min(2 * self.tr, self.tr_max)

        # This is the "else" case in the book (i.e. if rho <= eta). 
        # If we don't accept, we actually need to roll back
        # the part where we added the step because we had to 
        # set x_next = x_current + step to compute f(x + p)
        # in the first place (in the else case, we do nothing
        # since the step has been added)
        if rho < self.eta or math.isclose(rho, self.eta):
            self._set_params(x_current)


    def cg_steihaug(self, grad, closure):
        """
        Implements CG Steihaug for Trust Region based on Algorithm 7.2 from
        Nocedal & Wright Numerical Optimization
        """
        z = torch.zeros_like(grad)
        r = grad.clone()
        d = -r

        if r.norm() < self.cg_tol:
            self.tracker["cg_counter"].append(0)
            return z
        
        rr = r.dot(r)
        for i in range(self.cg_max_iter):
            cg_count = i + 1
            Bd = self._get_hvp(d, closure)
            dBd = d.dot(Bd)
            self.tracker["cg_dBd_all"].append(dBd.item()) # track for all iterations

            # If dBd is not positive definite, up to some tolerance
            if dBd <  0 or torch.isclose(dBd, torch.zeros_like(dBd, device=dBd.device)):
                self.tracker["cg_dBd"].append(dBd.item())
                self.tracker["cg_counter"].append(cg_count)

                tau = self.find_tau(z, d)
                return z + tau * d

            alpha = rr / dBd
            z_new = z + alpha * d

            z_new_norm = z_new.norm()
            if z_new_norm > self.tr or torch.isclose(z_new_norm, torch.tensor(self.tr, dtype=z_new_norm.dtype, device=z_new_norm.device)):
                self.tracker["cg_dBd"].append(dBd.item())
                self.tracker["cg_counter"].append(cg_count)

                tau = self.find_tau(z, d)
                return z + tau * d

            r_new = r + alpha * Bd
            rr_new = r_new.dot(r_new)
            if rr_new.sqrt() < self.cg_tol:
                self.tracker["cg_dBd"].append(dBd.item())
                self.tracker["cg_counter"].append(cg_count)
                return z_new

            beta = rr_new / rr
            d = -r_new + beta * d

            z, r, rr = z_new, r_new, rr_new

        self.tracker["cg_dBd"].append(dBd.item())
        self.tracker["cg_counter"].append(cg_count)
        return z


    def find_tau(self, z, d):
        """
        Finds tau >= 0 such that ||z_j + tau d_j|| = delta_k, with p_k = z_j + tau d_j.
        Square both sides to get following:
            (z + tau d) dot (z + tau d) = delta^2
        which expands to:
            (d dot d) * tau^2 + 2 (z dot d) tau + (z dot z - delta^2) = 0.
        The above equation is quadratic in terms of tau, which leads to simple solving
        """
        a0 = d.dot(d)
        a1 = 2.0 * z.dot(d)
        a2 = z.dot(z) - self.tr**2

        # Solve for tau >= 0
        # Ensure we have real solution
        disc = a1**2 - 4*a0*a2

        # For GPU safety, move to the right device
        disc = disc.to(z.device) if not isinstance(disc, torch.Tensor) else disc

        if disc < 0:
            return torch.tensor(0.0, device=z.device)

        sqrt_disc = torch.sqrt(disc)
        tau_0 = (-a1 + sqrt_disc) / (2.0 * a0)
        tau_1 = (-a1 - sqrt_disc) / (2.0 * a0)

        candidates = []
        if tau_0 >= 0:
            candidates.append(tau_0)
        if tau_1 >= 0:
            candidates.append(tau_1)
        if not candidates:
            return torch.tensor(0.0, device=z.device)

        return min(candidates) # Return smallest candidate for tau


if __name__ == "__main__":
    from src.fcn import FCN
    from torch import nn

    """
    Basic test: just make sure things run...
    """
    print("*****************************************")
    print("*****************************************")
    print("Test 1: Make sure it runs")

    torch.manual_seed(42)
    x = torch.rand((3))

    # Instantiate network and optimizer
    fcn_tr_cg = FCN(3, 3, 2, 1)
    tr_cg_opt = TrustRegionNewtonCG(fcn_tr_cg)

    def closure(out=None, backward=True):
        # Zero out grad
        fcn_tr_cg.zero_grad() 

        # Forward propagate through model
        if out is None:
            out = fcn_tr_cg(x)

        # Dummy scalar loss
        loss = (out**2).sum()

        # Backpropagate
        if backward:
            loss.backward()

        return loss

    # In step(), optimizer calls closure()
    tr_cg_opt.step(closure, model_args=x)
    print("Test 1 OK")
    print("*****************************************")
    print("*****************************************")
    print()


    """
    Test 2: Solve Ax = b for symmetric A > 0
    """
    print("*****************************************")
    print("*****************************************")
    print("Test 2: Solve Ax = b for symmetric A > 0")
    torch.manual_seed(0)

    # Define A (symmetric positive definite) and b
    A = torch.tensor([[3., 2.],
                     [2., 6.]], requires_grad=False)
    b = torch.tensor([2., -8.], requires_grad=False)
    x_init = torch.zeros(2)

    # Define simple model: just a single parameter vector x
    class QuadModel(nn.Module):
        def __init__(self, x_init):
            super().__init__()
            self.x = nn.Parameter(x_init)  # start from zero

        def forward(self, _=None):
            # f(x) = 0.5 x^T A x - b^T x
            return 0.5 * self.x @ A @ self.x - b @ self.x

    model = QuadModel(x_init)
    optimizer = TrustRegionNewtonCG(model, tr_init=1.0, tr_max=10.0)

    def closure(out=None, backward=True):
        optimizer.zero_grad()
        loss = model()
        if backward:
            loss.backward()
        return loss

    # Train
    for i in range(20):
        loss = closure()
        print(f"[{i}] loss: {loss.item():.6f}, x: {model.x.data.tolist()}")
        optimizer.step(closure, model_args=None)

    print(f"x* = {torch.linalg.solve(A, b)}")
    print("*****************************************")
    print("*****************************************")
    print()


    """
    Test 3: Use our FCN
    """
    print("*****************************************")
    print("*****************************************")
    print("Test 3: Use FCN")

    # Define input and target data (tiny dataset)
    x_data = torch.randn(10, 3)  # 10 samples, 3 features
    y_target = torch.randn(10, 1)  # 10 targets, 1 output

    # Instantiate your FCN
    model = FCN(in_neurons=3, hid_neurons=5, n_hidden_layers=2, out_neurons=1)

    # Define Trust Region Optimizer
    optimizer = TrustRegionNewtonCG(model, tr_init=5.0, tr_max=10.0, eta=0.001)

    # Loss Function
    mse_loss = nn.MSELoss()

    def closure(out=None, backward=True):
        optimizer.zero_grad()

        if out is None:
            out = model(x_data)

        loss = mse_loss(out, y_target)

        if backward:
            loss.backward()

        return loss

    # Training loop
    for step in range(101):
        loss = closure()

        # Print every 100 steps
        if step % 1 == 0:
            print(f"Step {step:2d} | Loss: {loss.item():.6f} | TR Radius: {optimizer.tr:.4f}")

        optimizer.step(closure, model_args=x_data)

    print(optimizer.tracker)

    print("*****************************************")
    print("*****************************************")
    print()