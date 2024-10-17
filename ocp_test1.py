"""
Test case 1 for the SCVx* implemenmtation... (Oguri, 2023)
"""

import numpy as np
import cvxpy as cp

def ocp_cvx_AL(prob, verbose=False):
    """
    General function of the OCP.
    Constraints are handled by con_list (dict).
    Cleanest implementation. Want to merge to this one eventually... (05/06)
    """

    zbar = prob.zref 
    g_zbar, dg_zbar = compute_g(zbar)   

    z = cp.Variable((2,))
    ξ = cp.Variable((1,))
    
    
    con = []
    # convex constraints
    con += [ z >= -2, z <= 2 ]
    con += [- z[1] - 4/3 * z[0] - 2/3 <= 0]
    # nonconvex constraint   
    con += [ g_zbar + dg_zbar.T @ (z - zbar) == ξ ] 
    # trust region 
    con += [cp.norm(z - zbar, "inf") <= prob.rk]  # trust region constraint

    g = ξ   # column-wise vectorization 
    h = np.zeros((1))
    f0 = compute_f0(z)
    P  = compute_P(g, h, prob.pen_w, prob.pen_λ, prob.pen_μ)    
    cost = f0 + P 
    
    p = cp.Problem(cp.Minimize(cost), con)
    p.solve(solver=cp.CLARABEL, verbose=verbose)
    z_opt  = z.value 
    status = p.status
    f0_opt = f0.value
    P_opt  = P.value
    L_opt  = p.value
    ξ_opt  = ξ.value    
    ζ_opt  = h   

    sol = {"z": z_opt, "ξ": ξ_opt, "ζ": ζ_opt,  "status": status, "L": L_opt, "f0": f0_opt, "P": P_opt}
    
    return sol



def solve_cvx_AL(prob, verbose=False):  
    """
    Solving the convexified problem. 
    You may add any convexification process here (e.g., comptuation of state transition matrix in the nonlinear dynamics...).
    """

    sol = ocp_cvx_AL(prob, verbose=verbose)
    
    return sol 



# define objective functions 
        
def compute_f0(z, prob=None):
    """
    Objective function (written in CVXPY)
    """
    
    return cp.sum(z)    


def compute_P(g, h, pen_w, pen_λ, pen_μ):
    """
    Compute the (convex) penalty term for the argumented Lagrangian (written in CVXPY)
    NO NEED TO CHANGE THIS FUNCTION.
    """
    
    zero = cp.Constant((np.zeros(h.shape)))
    hp = cp.maximum(zero, h)
    
    P = pen_λ.T @ g + pen_μ.T @ hp + pen_w/2 * (cp.norm(g)**2 + cp.norm(hp)**2)
    
    return P


def compute_g(z, prob=None):
    """ 
    Returning nonconvex constraint value g and its gradient dg (written in NUMPY)
    """
    
    z1, z2 = z[0], z[1] 
    
    g  = np.array([z2 - z1**4 - 2*z1**3 + 1.2*z1**2 + 2*z1])
    dg = np.array([-4*z1**3 - 6*z1**2 + 2.4*z1 + 2, 1])
    
    return g, dg


def compute_h(z, prob=None):
    """
    Return inequality h and dh (written in NUMPY)
    """
    return np.zeros((1)) , None  


class NOCP_SCVX:
    """
    Nonconvex Optimal Control Problem (OCP) class. Tailored for the SCVx* implementation.
    """
    def __init__(self, 
                 ):
        
        # scp parameters 
        self.iter_max  = 100
        
        # SVCx parameters (DON'T CHANGE) =======================================
        self.α = np.array([2, 3])
        self.β = 2
        self.γ = 0.9
        self.ρ = np.array([0.0, 0.25, 0.7])
        self.r_minmax = np.array([1e-10, 10])
        self.r0 = 0.1
        self.w0 = 100
        self.ϵopt  = 1e-5
        self.ϵfeas = 1e-5
        # =======================================================
    
        # initial solution
        self.zref = np.array([1.5, 1.5])
        
        self.sol_0 = {"z": self.zref}
