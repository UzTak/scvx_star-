import numpy as np 
import scipy as sp 

from ocp_test1 import * 

def scvx_update_weights(z, prob):
    
    g, _ = compute_g(z, prob)
    h, _ = compute_h(z, prob)
    
    prob.pen_λ += prob.pen_w * g
    prob.pen_μ += prob.pen_w * h
    prob.pen_μ[prob.pen_μ < 0.0] = 0
    prob.pen_w *= prob.β
    
    return prob 


def scvx_update_delta(δ, γ, ΔJ):
    return γ * δ if δ < 1e10 else abs(ΔJ) 


def scvx_update_r(r, α, ρ, ρk, r_minmax):
    α1, α2 = α
    _, ρ1, ρ2 = ρ   # ρ0 < ρ1 < ρ2
    r_min, r_max = r_minmax
     
    if ρk < ρ1:
        return np.max([r/α1, r_min])
    elif ρk < ρ2:
        return r    # no change in the trust region 
    else:
        return np.min([α2*r, r_max])


def scvx_compute_dJdL(sol, sol_prev, prob, iter):
    """
    a dictionary "z" will contain all regualr and slack variables that are optimzied in the problem. 
    """
    
    w, λ, μ = prob.pen_w, prob.pen_λ, prob.pen_μ
    
    g_sol, _ = compute_g(sol["z"], prob)
    h_sol, _ = compute_h(sol["z"], prob)
    
    g_sol_prev = prob.g_sol_prev
    h_sol_prev = prob.h_sol_prev
    
    J0 = compute_f0(sol["z"],      prob).value + compute_P(g_sol,      h_sol,      w, λ, μ).value   
    J1 = compute_f0(sol_prev["z"], prob).value + compute_P(g_sol_prev, h_sol_prev, w, λ, μ).value  
    L  = compute_f0(sol["z"],      prob).value + compute_P(sol["ξ"],   sol["ζ"],   w, λ, μ).value 
    
    ΔJ = J1 - J0 
    ΔL = J1 - L 
    
    h_p = np.array([ele if ele >= 0 else 0 for ele in h_sol]) 
    χ = np.linalg.norm(np.hstack((g_sol, h_p)))
    
    # print("g_sol: ", np.round(np.linalg.norm(g_sol),4), ", h_sol: ", np.round(np.linalg.norm(h_p)))
    # print("g_sol_prev: ", np.round(np.linalg.norm(g_sol_prev),4), ", h_sol_prev: ", np.round(np.linalg.norm(h_sol_prev),4))
    # print("ξopt:", np.round(np.linalg.norm(ξopt),4), ", ζopt:", np.round(np.linalg.norm(ζopt),4))
    
    assert np.abs(L - sol["L"]) < 1e-6, "L must be equal to the one computed in cvxpy! "
    
    print("J_new: ", np.round(J0,4), ", J_prev: ", np.round(J1,4), ", L_new: ", np.round(L,4))
    
    if ΔL < 0 and iter!=0:
        # raise ValueError("ΔL must be positive! ")
        print("WARNING: ΔL must be positive! ")
        
    prob.g_sol_prev = g_sol
    prob.h_sol_prev = h_sol
    
    return ΔJ, ΔL, χ


# main routine 
def solve_scvx(prob):
    """
    General routine for SCvx*.
    Oguri, "Successive Convexification with Feasibility Guarantee via Augmented Lagrangian for Non-Convex Optimal Control Problems" (2023).  

    input variables: 
        z: optimization variables
        r: trust region
        w: penalty weight
        ϵopt: optimality tolerance
        ϵfeas: feasibility tolerance
        ρ0, ρ1, ρ2: trust region thresholds   
        α1, α2, β, γ: trust region / weight update parameters
    
        sol_0: initial solution dict
    
    return: 
        zopt: optimal solution
        λ: 
        μ:
        Jopt: optimal objective 
    """
    # extract the initial parameters 
    prob.rk, prob.pen_w = prob.r0, prob.w0           
    
    # make a initial solution dictionary    
    sol_prev = prob.sol_0 
        
    # initialization 
    k = 0 
    ΔJ, χ, δ = np.inf, np.inf, np.inf
    g, _ = compute_g(sol_prev["z"], prob)
    h, _ = compute_h(sol_prev["z"], prob)
    prob.g_sol_prev, prob.h_sol_prev = g, h 
    prob.pen_λ, prob.pen_μ = np.zeros(np.shape(g)), np.zeros(np.shape(h))   
    
    log = {"ϵopt":[], "ϵfeas":[], "f0":[], "P":[]}
    
    while abs(ΔJ) > prob.ϵopt or χ > prob.ϵfeas:  # both optimality and feasibility must converge 

        # ==== This part is a problem-specific component ================

        # solve the convexified problem (affinize if needed)
        sol = solve_cvx_AL(prob, verbose=False)
        
        # ============================================================
        
        if sol["status"] == "infeasible":
            print("infeasible iCS! terminating...")
            break
        
        ΔJ, ΔL, χ = scvx_compute_dJdL(sol, sol_prev, prob, k)

        if ΔL < prob.ϵfeas:  #  originally == 0 in the paper. Relaxing this to accomodate numerical issue 
            ρk = 1
        else:
            ρk = ΔJ / ΔL
                
        print(f"iter: {k}, ", "L(=f0+P):", np.round(sol["L"], 4),  ", f0:", np.round(sol["f0"],4), ", P:", np.round(sol["P"],4), f", χ: {χ:.3f},  r: {prob.rk:.6f}, w: {prob.pen_w:.1f}, ρk: {ρk:.3f}, ΔJ: {ΔJ:.3f}, ΔL: {ΔL:.3f}, δ: {δ:.3f}")
        
        log["ϵopt"].append(ΔJ)
        log["ϵfeas"].append(χ)
        log["f0"].append(sol["f0"])
        log["P"].append(sol["P"])
        
        if ρk >= prob.ρ[0]:
            
            print("reference is updated... ")  
            sol_prev = sol 
            prob.zref = sol["z"]
            
            if abs(ΔJ) < δ:
                print('======= weight update! =========')
                prob = scvx_update_weights(sol["z"], prob)
                δ = scvx_update_delta(δ, prob.γ, ΔJ) 
                
        prob.rk = scvx_update_r(prob.rk, prob.α, prob.ρ, ρk, prob.r_minmax)  
        
        k += 1
        
        if k > prob.iter_max:
            print("SCVx* did not converge... terminating...")
            break
        
    return sol, log 





if __name__ == "__main__":
    
    prob = NOCP_SCVX()
    
    sol, log = solve_scvx(prob)
    
    print("converged!")
    print("optimal solution: ", sol)


