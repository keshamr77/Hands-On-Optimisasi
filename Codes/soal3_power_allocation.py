# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Soal 3: Alokasi Daya Downlink
Parameter: g11=1.0, g22=1.2, q12=0.3, g21=0.4, σ²=0.1, P_max=1.0
Threshold γ ∈ {0.5, 1.0, 2.0, 2.5}

Mencakup:
- Formulasi masalah dan Lagrangian
- Penentuan titik KKT via Active Set
- Validasi dengan CVXPY
- Algoritma Primal-Dual
- Plot primal/dual variables vs iterasi
- Plot objektif primal dan dual vs iterasi
- Analisis power-limited regime
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os

# ============================================================
# Parameter Sistem
# ============================================================
g11 = 1.0    # Channel gain user 1 ke BS 1
g22 = 1.2    # Channel gain user 2 ke BS 2
q12 = 0.3    # Interference dari BS 1 ke user 2
g21 = 0.4    # Interference dari BS 2 ke user 1
sigma2 = 0.1 # Noise variance
P_max = 1.0  # Maximum total power

# ============================================================
# BAGIAN 1: Formulasi Masalah
# ============================================================

def print_formulation():
    """
    Formulasi masalah alokasi daya downlink.
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              FORMULASI MASALAH ALOKASI DAYA DOWNLINK             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║ MASALAH:                                                         ║
║   min   f(p₁, p₂) = p₁ + p₂                                    ║
║   s.t.  SINR₁ = g₁₁·p₁ / (g₂₁·p₂ + σ²) ≥ γ     (C1)         ║
║         SINR₂ = g₂₂·p₂ / (q₁₂·p₁ + σ²) ≥ γ     (C2)         ║
║         p₁ + p₂ ≤ P_max                            (C3)         ║
║         p₁ ≥ 0, p₂ ≥ 0                                          ║
║                                                                  ║
║ Transformasi ke bentuk standar (≤):                              ║
║   γ(g₂₁·p₂ + σ²) - g₁₁·p₁ ≤ 0                    (C1)         ║
║   γ(q₁₂·p₁ + σ²) - g₂₂·p₂ ≤ 0                    (C2)         ║
║   p₁ + p₂ - P_max ≤ 0                               (C3)         ║
║                                                                  ║
║ LAGRANGIAN:                                                      ║
║   L = p₁ + p₂                                                   ║
║     + λ₁[γ(g₂₁·p₂ + σ²) - g₁₁·p₁]                             ║
║     + λ₂[γ(q₁₂·p₁ + σ²) - g₂₂·p₂]                             ║
║     + λ₃[p₁ + p₂ - P_max]                                       ║
║                                                                  ║
║ KONDISI KKT:                                                     ║
║   ∂L/∂p₁ = 1 - λ₁·g₁₁ + λ₂·γ·q₁₂ + λ₃ = 0        (1)         ║
║   ∂L/∂p₂ = 1 + λ₁·γ·g₂₁ - λ₂·g₂₂ + λ₃ = 0        (2)         ║
║   λ₁[γ(g₂₁·p₂ + σ²) - g₁₁·p₁] = 0                  (3)         ║
║   λ₂[γ(q₁₂·p₁ + σ²) - g₂₂·p₂] = 0                  (4)         ║
║   λ₃[p₁ + p₂ - P_max] = 0                            (5)         ║
║   λ₁, λ₂, λ₃ ≥ 0                                     (6)         ║
╚══════════════════════════════════════════════════════════════════╝
""")

# ============================================================
# BAGIAN 2: Solusi KKT via Active Set Method
# ============================================================

def solve_kkt_active_set(gamma, verbose=True):
    """
    Penyelesaian titik KKT menggunakan metode Active Set.
    
    Strategi: Coba berbagai kombinasi active constraints.
    Minimal, constraint SINR biasanya aktif (equality).
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"ACTIVE SET METHOD untuk γ = {gamma}")
        print(f"{'='*60}")
    
    best_solution = None
    best_obj = np.inf
    
    # Kasus-kasus active set yang mungkin:
    # Label: (C1_active, C2_active, C3_active)
    active_set_cases = [
        (True, True, False),   # Hanya SINR aktif
        (True, True, True),    # Semua aktif
        (True, False, False),  # Hanya C1 aktif
        (False, True, False),  # Hanya C2 aktif
    ]
    
    for case_idx, (c1_active, c2_active, c3_active) in enumerate(active_set_cases):
        if verbose:
            print(f"\n  Percobaan {case_idx+1}: C1={'Aktif' if c1_active else 'Non-aktif'}, "
                  f"C2={'Aktif' if c2_active else 'Non-aktif'}, "
                  f"C3={'Aktif' if c3_active else 'Non-aktif'}")
        
        try:
            # Jika constraint aktif → equality, jika tidak → λ = 0
            
            # Bangun sistem persamaan
            # Variabel: p1, p2, λ1, λ2, λ3
            # Persamaan stasioneritas:
            #   1 - λ1*g11 + λ2*γ*q12 + λ3 = 0
            #   1 + λ1*γ*g21 - λ2*g22 + λ3 = 0
            # Plus constraint equalities (dari active constraints)
            
            if c1_active and c2_active and c3_active:
                # Semua 3 constraint aktif
                # 5 persamaan, 5 variabel
                # Dari C1: γ(g21*p2 + σ²) = g11*p1
                # Dari C2: γ(q12*p1 + σ²) = g22*p2
                # Dari C3: p1 + p2 = P_max
                
                # Dari C1: g11*p1 - γ*g21*p2 = γ*σ²
                # Dari C2: -γ*q12*p1 + g22*p2 = γ*σ²
                
                A_primal = np.array([
                    [g11, -gamma*g21],
                    [-gamma*q12, g22]
                ])
                b_primal = np.array([gamma*sigma2, gamma*sigma2])
                
                p = np.linalg.solve(A_primal, b_primal)
                p1, p2 = p[0], p[1]
                
                # Cek apakah p1+p2 = P_max (atau override)
                if abs(p1 + p2 - P_max) > 1e-6:
                    # SINR constraints menentukan p1, p2 yang tidak memenuhi power constraint equality
                    # Maka power constraint mungkin tidak aktif, atau infeasible
                    if p1 + p2 > P_max + 1e-6:
                        if verbose:
                            print(f"    → Infeasible: p1+p2 = {p1+p2:.6f} > P_max = {P_max}")
                        continue
                    # Jika p1+p2 < P_max, maka C3 tidak seharusnya aktif
                    if verbose:
                        print(f"    → Kontradiksi: p1+p2 = {p1+p2:.6f} ≠ P_max, C3 seharusnya tidak aktif")
                    continue
                
                # Dari stasioneritas, cari λ1, λ2, λ3
                # 1 - λ1*g11 + λ2*γ*q12 + λ3 = 0
                # 1 + λ1*γ*g21 - λ2*g22 + λ3 = 0
                # Kita punya 2 persamaan dan 3 unknown → perlu C3: p1+p2=P_max sebagai identitas,
                # tapi λ3 bebas
                
                A_dual = np.array([
                    [-g11, gamma*q12, 1],
                    [gamma*g21, -g22, 1]
                ])
                b_dual = np.array([-1, -1])
                
                # Underdetermined, tapi karena kita memaksakan semua aktif,
                # kita bisa selesaikan. Sebenarnya ini punya solusi unik jika kita
                # menambahkan fact bahwa p1+p2=P_max memang tercapai oleh SINR constraints
                
                # Gunakan least squares
                lam, _, _, _ = np.linalg.lstsq(A_dual, b_dual, rcond=None)
                lam1, lam2, lam3 = lam
                
            elif c1_active and c2_active and not c3_active:
                # C1 dan C2 aktif, C3 tidak aktif → λ3 = 0
                lam3 = 0.0
                
                # Dari C1: g11*p1 - γ*g21*p2 = γ*σ²
                # Dari C2: -γ*q12*p1 + g22*p2 = γ*σ²
                
                A_primal = np.array([
                    [g11, -gamma*g21],
                    [-gamma*q12, g22]
                ])
                b_primal = np.array([gamma*sigma2, gamma*sigma2])
                
                p = np.linalg.solve(A_primal, b_primal)
                p1, p2 = p[0], p[1]
                
                # Cek feasibility p1+p2 ≤ P_max
                if p1 + p2 > P_max + 1e-6:
                    if verbose:
                        print(f"    → Infeasible: p1+p2 = {p1+p2:.6f} > P_max = {P_max}")
                    continue
                
                # Dari stasioneritas (dengan λ3=0):
                # 1 - λ1*g11 + λ2*γ*q12 = 0
                # 1 + λ1*γ*g21 - λ2*g22 = 0
                
                A_dual = np.array([
                    [-g11, gamma*q12],
                    [gamma*g21, -g22]
                ])
                b_dual = np.array([-1, -1])
                
                lam_dual = np.linalg.solve(A_dual, b_dual)
                lam1, lam2 = lam_dual[0], lam_dual[1]
                
            elif c1_active and not c2_active and not c3_active:
                lam2 = 0.0
                lam3 = 0.0
                
                # Hanya C1 aktif:
                # 1 - λ1*g11 = 0 → λ1 = 1/g11
                lam1 = 1.0 / g11
                
                # 1 + λ1*γ*g21 = 0 → ini tidak nol untuk γ > 0 jadi tidak valid umumnya
                check = 1 + lam1 * gamma * g21
                if abs(check) > 1e-6:
                    if verbose:
                        print(f"    → Stasioneritas ∂L/∂p₂ tidak terpenuhi: {check:.6f} ≠ 0")
                    continue
                    
                # p1 dari C1: g11*p1 = γ*(g21*p2 + σ²)
                # Perlu p2 dari persamaan lain... underdetermined
                continue
                
            elif not c1_active and c2_active and not c3_active:
                lam1 = 0.0
                lam3 = 0.0
                
                lam2 = 1.0 / g22
                
                check = 1 + lam2 * gamma * q12
                if abs(check) > 1e-6:
                    if verbose:
                        print(f"    → Stasioneritas ∂L/∂p₁ tidak terpenuhi")
                    continue
                continue
            else:
                continue
            
            # Validasi
            feasible = True
            kkt_satisfied = True
            
            # Cek p ≥ 0
            if p1 < -1e-8 or p2 < -1e-8:
                feasible = False
                if verbose:
                    print(f"    → p negatif: p1={p1:.6f}, p2={p2:.6f}")
                continue
            
            p1 = max(0, p1)
            p2 = max(0, p2)
            
            # Cek λ ≥ 0
            if lam1 < -1e-6 or lam2 < -1e-6 or lam3 < -1e-6:
                if verbose:
                    print(f"    → λ negatif: λ1={lam1:.6f}, λ2={lam2:.6f}, λ3={lam3:.6f}")
                continue
            
            lam1 = max(0, lam1)
            lam2 = max(0, lam2)
            lam3 = max(0, lam3)
            
            # Cek feasibility
            sinr1 = g11 * p1 / (g21 * p2 + sigma2)
            sinr2 = g22 * p2 / (q12 * p1 + sigma2)
            
            if sinr1 < gamma - 1e-6 or sinr2 < gamma - 1e-6:
                if verbose:
                    print(f"    → SINR infeasible: SINR1={sinr1:.4f}, SINR2={sinr2:.4f}")
                continue
            
            if p1 + p2 > P_max + 1e-6:
                if verbose:
                    print(f"    → Power infeasible: p1+p2={p1+p2:.6f}")
                continue
            
            # Cek complementary slackness
            cs1 = lam1 * (gamma * (g21 * p2 + sigma2) - g11 * p1)
            cs2 = lam2 * (gamma * (q12 * p1 + sigma2) - g22 * p2)
            cs3 = lam3 * (p1 + p2 - P_max)
            
            obj = p1 + p2
            
            if verbose:
                print(f"    → VALID! p1={p1:.6f}, p2={p2:.6f}")
                print(f"      λ1={lam1:.6f}, λ2={lam2:.6f}, λ3={lam3:.6f}")
                print(f"      Objektif = {obj:.6f}")
                print(f"      SINR1 = {sinr1:.4f} ≥ {gamma}, SINR2 = {sinr2:.4f} ≥ {gamma}")
                print(f"      CS: λ1·h1={cs1:.8f}, λ2·h2={cs2:.8f}, λ3·h3={cs3:.8f}")
            
            if obj < best_obj:
                best_obj = obj
                best_solution = {
                    'p1': p1, 'p2': p2,
                    'lam1': lam1, 'lam2': lam2, 'lam3': lam3,
                    'obj': obj,
                    'sinr1': sinr1, 'sinr2': sinr2,
                    'active_set': (c1_active, c2_active, c3_active)
                }
        
        except np.linalg.LinAlgError:
            if verbose:
                print(f"    → Singular system")
            continue
    
    if best_solution is None and verbose:
        print(f"\n  [!] Tidak ditemukan solusi feasible untuk gamma = {gamma}")
        print(f"    Masalah mungkin INFEASIBLE.")
    
    return best_solution

# ============================================================
# BAGIAN 3: Validasi dengan CVXPY
# ============================================================

def solve_cvxpy(gamma, verbose=True):
    """Validasi solusi KKT menggunakan CVXPY."""
    p1 = cp.Variable(nonneg=True)
    p2 = cp.Variable(nonneg=True)
    
    objective = cp.Minimize(p1 + p2)
    
    constraints = [
        g11 * p1 >= gamma * (g21 * p2 + sigma2),
        g22 * p2 >= gamma * (q12 * p1 + sigma2),
        p1 + p2 <= P_max,
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    
    if verbose:
        print(f"\n  CVXPY Validation (γ = {gamma}):")
        print(f"    Status: {problem.status}")
        
    if problem.status in ['optimal', 'optimal_inaccurate']:
        p1_val = p1.value
        p2_val = p2.value
        
        # Dapatkan dual variables
        lam1_val = constraints[0].dual_value
        lam2_val = constraints[1].dual_value
        lam3_val = constraints[2].dual_value
        
        if verbose:
            print(f"    p1* = {p1_val:.6f}, p2* = {p2_val:.6f}")
            print(f"    f* = p1+p2 = {p1_val + p2_val:.6f}")
            print(f"    λ1 = {lam1_val:.6f}, λ2 = {lam2_val:.6f}, λ3 = {lam3_val:.6f}")
            
            sinr1 = g11 * p1_val / (g21 * p2_val + sigma2)
            sinr2 = g22 * p2_val / (q12 * p1_val + sigma2)
            print(f"    SINR1 = {sinr1:.4f}, SINR2 = {sinr2:.4f}")
        
        return {
            'p1': p1_val, 'p2': p2_val,
            'lam1': float(lam1_val) if lam1_val is not None else 0,
            'lam2': float(lam2_val) if lam2_val is not None else 0,
            'lam3': float(lam3_val) if lam3_val is not None else 0,
            'obj': p1_val + p2_val,
            'status': problem.status
        }
    else:
        if verbose:
            print(f"    Masalah {problem.status}")
        return {'status': problem.status, 'p1': None, 'p2': None}

# ============================================================
# BAGIAN 4: Algoritma Primal-Dual
# ============================================================

def primal_dual_algorithm(gamma, outer_iter=200, inner_iter=50, verbose=True):
    """
    Algoritma Primal-Dual menggunakan Method of Multipliers (Augmented Lagrangian).
    
    Outer loop: update dual variables (multipliers)
    Inner loop: minimize augmented Lagrangian w.r.t. primal variables
    
    L_aug = f(p) + sum_i lam_i * h_i(p) + (rho/2) * sum_i [max(0, h_i(p) + lam_i/rho)]^2
    
    Update dual: lam_i = max(0, lam_i + rho * h_i(p*))
    """
    # Parameter
    rho = 2.0           # penalty parameter
    alpha_p = 0.001     # primal step size for inner loop
    
    # Inisialisasi
    p1 = 0.3
    p2 = 0.3
    lam1 = 0.5
    lam2 = 0.5
    lam3 = 0.0
    
    # Storage untuk plotting
    p1_hist = [p1]
    p2_hist = [p2]
    lam1_hist = [lam1]
    lam2_hist = [lam2]
    lam3_hist = [lam3]
    primal_obj_hist = [p1 + p2]
    dual_obj_hist = []
    
    for outer in range(outer_iter):
        # === Inner loop: minimize augmented Lagrangian w.r.t. p ===
        for inner in range(inner_iter):
            # Constraint values
            h1 = gamma * (g21 * p2 + sigma2) - g11 * p1
            h2 = gamma * (q12 * p1 + sigma2) - g22 * p2
            h3 = p1 + p2 - P_max
            
            # Augmented multipliers: mu_i = lam_i + rho * max(0, h_i)
            # For augmented Lagrangian: penalty on max(0, h_i + lam_i/rho)
            mu1 = max(0, lam1 + rho * h1)
            mu2 = max(0, lam2 + rho * h2)
            mu3 = max(0, lam3 + rho * h3)
            
            # Gradient of augmented Lagrangian w.r.t. p
            # dh1/dp1 = -g11,  dh1/dp2 = gamma*g21
            # dh2/dp1 = gamma*q12,  dh2/dp2 = -g22
            # dh3/dp1 = 1,  dh3/dp2 = 1
            dL_dp1 = 1 + mu1 * (-g11) + mu2 * (gamma * q12) + mu3 * 1
            dL_dp2 = 1 + mu1 * (gamma * g21) + mu2 * (-g22) + mu3 * 1
            
            p1 = max(1e-8, p1 - alpha_p * dL_dp1)
            p2 = max(1e-8, p2 - alpha_p * dL_dp2)
        
        # === Outer: update dual variables ===
        h1 = gamma * (g21 * p2 + sigma2) - g11 * p1
        h2 = gamma * (q12 * p1 + sigma2) - g22 * p2
        h3 = p1 + p2 - P_max
        
        lam1 = max(0, lam1 + rho * h1)
        lam2 = max(0, lam2 + rho * h2)
        lam3 = max(0, lam3 + rho * h3)
        
        # Increase penalty gradually
        rho = min(rho * 1.01, 50.0)
        
        # Record
        primal_obj = p1 + p2
        dual_obj = primal_obj + lam1 * h1 + lam2 * h2 + lam3 * h3
        
        p1_hist.append(p1)
        p2_hist.append(p2)
        lam1_hist.append(lam1)
        lam2_hist.append(lam2)
        lam3_hist.append(lam3)
        primal_obj_hist.append(primal_obj)
        dual_obj_hist.append(dual_obj)
    
    if verbose:
        print(f"\n  Primal-Dual Method of Multipliers (gamma = {gamma}):")
        print(f"    Konvergensi setelah {outer_iter} outer iterasi:")
        print(f"    p1* = {p1:.6f}, p2* = {p2:.6f}")
        print(f"    lam1* = {lam1:.6f}, lam2* = {lam2:.6f}, lam3* = {lam3:.6f}")
        print(f"    f* = {p1+p2:.6f}")
    
    return {
        'p1_hist': p1_hist, 'p2_hist': p2_hist,
        'lam1_hist': lam1_hist, 'lam2_hist': lam2_hist, 'lam3_hist': lam3_hist,
        'primal_obj_hist': primal_obj_hist, 'dual_obj_hist': dual_obj_hist,
        'p1': p1, 'p2': p2,
        'lam1': lam1, 'lam2': lam2, 'lam3': lam3
    }

# ============================================================
# BAGIAN 5: Objektif Dual
# ============================================================

def dual_objective(lam1, lam2, lam3, gamma):
    """
    Objektif dual g(λ₁, λ₂, λ₃) = min_{p≥0} L(p, λ)
    
    L = (1 - λ₁·g₁₁ + λ₂·γ·q₁₂ + λ₃)·p₁ 
      + (1 + λ₁·γ·g₂₁ - λ₂·g₂₂ + λ₃)·p₂
      + λ₁·γ·σ² + λ₂·γ·σ² - λ₃·P_max
    
    L linear di p → minimisasi:
      Jika koefisien p_i > 0 → p_i = 0
      Jika koefisien p_i < 0 → p_i → -∞ (unbounded) → g = -∞
      Jika koefisien p_i = 0 → p_i bebas
    
    Untuk g(λ) > -∞, koefisien harus ≥ 0.
    Maka: g(λ) = λ₁·γ·σ² + λ₂·γ·σ² - λ₃·P_max
    """
    coeff_p1 = 1 - lam1 * g11 + lam2 * gamma * q12 + lam3
    coeff_p2 = 1 + lam1 * gamma * g21 - lam2 * g22 + lam3
    
    if coeff_p1 < -1e-10 or coeff_p2 < -1e-10:
        return -np.inf
    
    return lam1 * gamma * sigma2 + lam2 * gamma * sigma2 - lam3 * P_max

# ============================================================
# BAGIAN 6: Plotting
# ============================================================

def plot_primal_dual_results(gamma, pd_result, output_dir):
    """Plot hasil primal-dual: variabel dan objektif vs iterasi."""
    
    iters = range(len(pd_result['p1_hist']))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Algoritma Primal-Dual — γ = {gamma}', fontsize=15, fontweight='bold')
    
    # Plot 1: p1, p2 vs iterasi
    ax = axes[0, 0]
    ax.plot(iters, pd_result['p1_hist'], 'b-', linewidth=1.5, label='p₁')
    ax.plot(iters, pd_result['p2_hist'], 'r-', linewidth=1.5, label='p₂')
    ax.set_xlabel('Iterasi')
    ax.set_ylabel('Daya')
    ax.set_title('Variabel Primal (p₁, p₂)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: λ1, λ2, λ3 vs iterasi
    ax = axes[0, 1]
    ax.plot(iters, pd_result['lam1_hist'], 'b-', linewidth=1.5, label='λ₁')
    ax.plot(iters, pd_result['lam2_hist'], 'r-', linewidth=1.5, label='λ₂')
    ax.plot(iters, pd_result['lam3_hist'], 'g-', linewidth=1.5, label='λ₃')
    ax.set_xlabel('Iterasi')
    ax.set_ylabel('Pengali Lagrange')
    ax.set_title('Variabel Dual (λ₁, λ₂, λ₃)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Objektif primal dan dual vs iterasi
    ax = axes[1, 0]
    iters_dual = range(len(pd_result['dual_obj_hist']))
    ax.plot(list(iters), pd_result['primal_obj_hist'], 'b-', linewidth=1.5, label='Primal f(p₁,p₂)')
    ax.plot(list(iters_dual), pd_result['dual_obj_hist'], 'r--', linewidth=1.5, label='Dual g(λ)')
    ax.set_xlabel('Iterasi')
    ax.set_ylabel('Nilai Objektif')
    ax.set_title('Objektif Primal vs Dual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Duality gap
    ax = axes[1, 1]
    min_len = min(len(pd_result['primal_obj_hist']), len(pd_result['dual_obj_hist']))
    gap = [pd_result['primal_obj_hist'][i+1] - pd_result['dual_obj_hist'][i] 
           for i in range(min_len)]
    ax.plot(range(min_len), gap, 'purple', linewidth=1.5)
    ax.set_xlabel('Iterasi')
    ax.set_ylabel('Duality Gap')
    ax.set_title('Duality Gap (Primal - Dual)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'soal3_primal_dual_gamma_{gamma}.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# BAGIAN 7: Main
# ============================================================

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 80)
    print("SOAL 3: ALOKASI DAYA DOWNLINK")
    print(f"Parameter: g11={g11}, g22={g22}, q12={q12}, g21={g21}, sigma2={sigma2}, Pmax={P_max}")
    print("=" * 80)
    
    print_formulation()
    
    gamma_values = [0.5, 1.0, 2.0]
    results_kkt = {}
    results_cvxpy = {}
    results_pd = {}
    
    for gamma in gamma_values:
        print(f"\n{'#'*80}")
        print(f"# γ = {gamma}")
        print(f"{'#'*80}")
        
        # Active Set KKT
        kkt_sol = solve_kkt_active_set(gamma)
        results_kkt[gamma] = kkt_sol
        
        # CVXPY Validation
        cvxpy_sol = solve_cvxpy(gamma)
        results_cvxpy[gamma] = cvxpy_sol
        
        # Perbandingan
        if kkt_sol and cvxpy_sol['p1'] is not None:
            print(f"\n  Perbandingan KKT vs CVXPY (γ = {gamma}):")
            print(f"    {'':15} {'KKT':>12} {'CVXPY':>12} {'Selisih':>12}")
            print(f"    {'p1':15} {kkt_sol['p1']:>12.6f} {cvxpy_sol['p1']:>12.6f} {abs(kkt_sol['p1']-cvxpy_sol['p1']):>12.8f}")
            print(f"    {'p2':15} {kkt_sol['p2']:>12.6f} {cvxpy_sol['p2']:>12.6f} {abs(kkt_sol['p2']-cvxpy_sol['p2']):>12.8f}")
            print(f"    {'f*':15} {kkt_sol['obj']:>12.6f} {cvxpy_sol['obj']:>12.6f} {abs(kkt_sol['obj']-cvxpy_sol['obj']):>12.8f}")
        
        # Primal-Dual
        pd_sol = primal_dual_algorithm(gamma)
        results_pd[gamma] = pd_sol
        
        # Plot
        plot_primal_dual_results(gamma, pd_sol, output_dir)
    
    # ============================================================
    # Tabel Ringkasan
    # ============================================================
    print(f"\n{'='*80}")
    print("TABEL RINGKASAN SOLUSI KKT")
    print(f"{'='*80}")
    print(f"\n{'γ':>6} {'p1*':>10} {'p2*':>10} {'λ1*':>10} {'λ2*':>10} {'λ3*':>10} {'f*':>10}")
    print("-" * 70)
    for gamma in gamma_values:
        sol = results_kkt[gamma]
        if sol:
            print(f"{gamma:>6.1f} {sol['p1']:>10.6f} {sol['p2']:>10.6f} "
                  f"{sol['lam1']:>10.6f} {sol['lam2']:>10.6f} {sol['lam3']:>10.6f} {sol['obj']:>10.6f}")
        else:
            print(f"{gamma:>6.1f} {'INFEASIBLE':>10}")
    
    # ============================================================
    # Objektif Dual
    # ============================================================
    print(f"\n{'='*80}")
    print("OBJEKTIF DUAL g(λ₁*, λ₂*, λ₃*)")
    print(f"{'='*80}")
    print(f"""
Objektif dual diperoleh dari:
  g(λ) = min_{{p≥0}} L(p, λ)
  
  L = (1 - λ₁g₁₁ + λ₂γq₁₂ + λ₃)p₁ + (1 + λ₁γg₂₁ - λ₂g₂₂ + λ₃)p₂ 
      + λ₁γσ² + λ₂γσ² - λ₃P_max

Karena L linear di p, minimisasi terhadap p ≥ 0:
  - Jika koefisien p_i ≥ 0 → p_i* = 0 → kontribusi = 0
  - Jika koefisien p_i < 0 → g(λ) = -∞ (unbounded below)

Pada titik KKT, koefisien p_i = 0 (dari kondisi stasioneritas),
sehingga:
  g(λ*) = λ₁*γσ² + λ₂*γσ² - λ₃*P_max
""")
    
    for gamma in gamma_values:
        sol = results_kkt[gamma]
        if sol:
            g_dual = sol['lam1'] * gamma * sigma2 + sol['lam2'] * gamma * sigma2 - sol['lam3'] * P_max
            print(f"  γ = {gamma}: g(λ*) = {sol['lam1']:.4f}×{gamma}×{sigma2} + {sol['lam2']:.4f}×{gamma}×{sigma2} - {sol['lam3']:.4f}×{P_max}")
            print(f"           g(λ*) = {g_dual:.6f}")
            print(f"           f(p*) = {sol['obj']:.6f}")
            print(f"           Strong Duality: g(λ*) ≈ f(p*)? {'YA ✓' if abs(g_dual - sol['obj']) < 0.01 else 'TIDAK ✗'}")
            print()
    
    # ============================================================
    # Analisis γ = 2.5 (Infeasible)
    # ============================================================
    print(f"\n{'='*80}")
    print("ANALISIS γ = 2.5")
    print(f"{'='*80}")
    
    gamma_25 = 2.5
    kkt_25 = solve_kkt_active_set(gamma_25)
    cvxpy_25 = solve_cvxpy(gamma_25)
    
    # ============================================================
    # Compute p1+p2 for each gamma from KKT
    ptotal_05 = results_kkt[0.5]['p1'] + results_kkt[0.5]['p2']
    ptotal_10 = results_kkt[1.0]['p1'] + results_kkt[1.0]['p2']
    ptotal_20 = results_kkt[2.0]['p1'] + results_kkt[2.0]['p2']
    
    # Get primal-dual lambda3
    pd_lam3_05 = results_pd[0.5]['lam3']
    pd_lam3_10 = results_pd[1.0]['lam3']
    pd_lam3_20 = results_pd[2.0]['lam3']
    
    print(f"""
1. PERILAKU lam3* KETIKA gamma DINAIKKAN:
   - gamma = 0.5: lam3* (KKT) = {results_kkt[0.5]['lam3']:.4f}, p1+p2 = {ptotal_05:.4f} < P_max -> power constraint TIDAK aktif
   - gamma = 1.0: lam3* (KKT) = {results_kkt[1.0]['lam3']:.4f}, p1+p2 = {ptotal_10:.4f} < P_max -> power constraint TIDAK aktif
   - gamma = 2.0: lam3* (KKT) = {results_kkt[2.0]['lam3']:.4f}, p1+p2 = {ptotal_20:.4f} = P_max -> power constraint AKTIF (marginally)
   
   Dari Primal-Dual: lam3* = {pd_lam3_05:.4f}, {pd_lam3_10:.4f}, {pd_lam3_20:.4f}
   
   Semakin besar gamma, semakin besar daya minimum yang dibutuhkan untuk 
   memenuhi target SINR. Pada gamma=2.0, daya total tepat sama dengan P_max,
   sehingga constraint daya menjadi tight (aktif).

2. MENGAPA gamma = 2.0 MENGARAH PADA SISTEM POWER-LIMITED:
   Pada gamma = 2.0, solusi optimal memberikan p1+p2 = {ptotal_20:.4f} = P_max.
   Ini berarti seluruh budget daya telah digunakan.
   
   Dari persamaan SINR (equality):
     p1 = gamma(g21*p2 + sigma^2)/g11 = 2*(0.4*p2 + 0.1)/1.0
     p2 = gamma(q12*p1 + sigma^2)/g22 = 2*(0.3*p1 + 0.1)/1.2
   
   Solusi: p1 = 5/9 = 0.5556, p2 = 4/9 = 0.4444
   Total: p1 + p2 = 1.0 = P_max tepat!
   
   Sistem power-limited karena:
   - Tidak ada slack pada constraint daya
   - Jika gamma dinaikkan sedikit saja, SINR constraints membutuhkan 
     p1+p2 > P_max, yang infeasible
   - CVXPY mendeteksi lam3 = {results_cvxpy[2.0].get('lam3', 0):.4f} > 0 
     (degenerate case: constraint aktif tapi multiplier ~0 dari KKT karena
      SINR constraints saja sudah menentukan solusi unik)

3. ANALISIS gamma = 2.5:
   Pada gamma = 2.5, dari persamaan SINR aktif:
     g11*p1 - gamma*g21*p2 = gamma*sigma^2
     -gamma*q12*p1 + g22*p2 = gamma*sigma^2
   
   Substitusi numerik:
     1.0*p1 - 1.0*p2 = 0.25
     -0.75*p1 + 1.2*p2 = 0.25
   
   Solusi: p1 = {1.2*0.25 + 1.0*0.25:.4f}/{1.0*1.2 - 1.0*0.75:.4f}, p2 = ...
""")
    
    # Compute exact solution for gamma=2.5
    A_25 = np.array([[g11, -2.5*g21], [-2.5*q12, g22]])
    b_25 = np.array([2.5*sigma2, 2.5*sigma2])
    p_25 = np.linalg.solve(A_25, b_25)
    print(f"   Solusi numerik: p1 = {p_25[0]:.6f}, p2 = {p_25[1]:.6f}")
    print(f"   Total: p1 + p2 = {p_25[0]+p_25[1]:.6f} > P_max = {P_max}")
    print(f"   -> Masalah INFEASIBLE!")
    print(f"")
    print(f"   Tidak ada titik KKT yang valid karena tidak ada solusi feasible.")
    print(f"   Interpretasi: target SINR = 2.5 terlalu tinggi untuk dicapai")
    print(f"   dengan total daya P_max = {P_max}. Interferensi antar-user")
    print(f"   (q12={q12}, g21={g21}) membuat sistem tidak mampu memenuhi")
    print(f"   kedua target SINR secara bersamaan.")
    print()
    
    # Plot perbandingan λ3 vs γ
    fig, ax = plt.subplots(figsize=(8, 5))
    gammas = gamma_values
    lam3_values = [results_kkt[g]['lam3'] for g in gammas]
    ax.bar(range(len(gammas)), lam3_values, color=['#2196F3', '#FF9800', '#F44336'], 
           tick_label=[f'γ={g}' for g in gammas], width=0.5)
    ax.set_xlabel('Threshold γ', fontsize=12)
    ax.set_ylabel('λ₃*', fontsize=12)
    ax.set_title('Pengali Lagrange λ₃* vs Threshold γ', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(lam3_values):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'soal3_lambda3_vs_gamma.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nSemua plot disimpan di:", output_dir)
