import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube

def generate_domain_pts(n_dom=500, n_bdry=20, seed=42, device="cpu"):
    """
    Generate interior and boundary points for a 1x1 square domain 
    """
    rng = np.random.default_rng(seed)

    # Use Latin hypercube sampling to generate quasi-random boundary points
    # And make sure we have corners
    lhs_1d = LatinHypercube(d=1, rng=rng)

    bdry_rans = [0] * 4
    for i in range(4):
        bdry_rans[i] = np.zeros(n_bdry)
        bdry_rans[i][1:-1] = np.squeeze(lhs_1d.random(n_bdry - 2))
        bdry_rans[i][-1] = 1.

    top    = torch.stack([torch.from_numpy(bdry_rans[0]).to(dtype=torch.float, device=device), torch.ones(n_bdry, device=device)], dim=1)
    right  = torch.stack([torch.ones(n_bdry, device=device), torch.from_numpy(bdry_rans[1]).to(dtype=torch.float, device=device)], dim=1)
    bottom = torch.stack([torch.from_numpy(bdry_rans[2]).to(dtype=torch.float, device=device), torch.zeros(n_bdry, device=device)], dim=1)
    left   = torch.stack([torch.zeros(n_bdry, device=device), torch.from_numpy(bdry_rans[3]).to(dtype=torch.float, device=device)], dim=1)

    bdry_set = torch.stack([top, right, bottom, left], dim=0)

    # Use LHS to generate quasi-random domain points
    # We also store domain points in a 3D tensor, for when we have subdomains
    # The tensor shape is (# of domains x # of points per domain x spatial dim)
    lhs_2d = LatinHypercube(d=2, rng=rng)
    dom_set = torch.from_numpy(lhs_2d.random(n_dom)).to(dtype=torch.float, device=device).unsqueeze(0)
    
    return dom_set, bdry_set