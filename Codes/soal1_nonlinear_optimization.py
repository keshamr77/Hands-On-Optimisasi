# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Soal 1: Optimisasi Fungsi Non-Linear
f(x,y) = x^4 + y^4 - 3x^2 - 3y^2 + 2

Mencakup:
- Gradien dan Hessian
- Nilai eigen Hessian
- Titik stasioner dan klasifikasinya
- Steepest Descent & Newton Method
- Plot lintasan optimisasi
- Perbandingan konvergensi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# ============================================================
# BAGIAN 1: Definisi Fungsi, Gradien, dan Hessian
# ============================================================

def f(x, y):
    """Fungsi objektif f(x,y) = x^4 + y^4 - 3x^2 - 3y^2 + 2"""
    return x**4 + y**4 - 3*x**2 - 3*y**2 + 2

def grad_f(x, y):
    """
    Gradien nabla f(x,y):
    df/dx = 4x^3 - 6x
    df/dy = 4y^3 - 6y
    """
    dfdx = 4*x**3 - 6*x
    dfdy = 4*y**3 - 6*y
    return np.array([dfdx, dfdy])

def hessian_f(x, y):
    """
    Matriks Hessian H(x,y):
    d2f/dx2  = 12x^2 - 6
    d2f/dxdy = 0
    d2f/dydx = 0
    d2f/dy2  = 12y^2 - 6
    """
    h11 = 12*x**2 - 6
    h12 = 0.0
    h21 = 0.0
    h22 = 12*y**2 - 6
    return np.array([[h11, h12],
                     [h21, h22]])

# ============================================================
# BAGIAN 2: Analisis Titik Stasioner
# ============================================================

def analyze_stationary_points():
    """
    Menyelesaikan nabla f(x,y) = 0:
    4x^3 - 6x = 0  =>  2x(2x^2 - 3) = 0  =>  x = 0, ±sqrt(3/2)
    4y^3 - 6y = 0  =>  2y(2y^2 - 3) = 0  =>  y = 0, ±sqrt(3/2)
    
    Total: 3 x 3 = 9 titik stasioner
    """
    x_vals = [0.0, np.sqrt(3/2), -np.sqrt(3/2)]
    y_vals = [0.0, np.sqrt(3/2), -np.sqrt(3/2)]
    
    stationary_points = []
    for x in x_vals:
        for y in y_vals:
            stationary_points.append((x, y))
    
    print("=" * 80)
    print("ANALISIS TITIK STASIONER")
    print("=" * 80)
    print(f"\nFungsi: f(x,y) = x^4 + y^4 - 3x^2 - 3y^2 + 2")
    print(f"\nGradien: grad f(x,y) = [4x^3 - 6x, 4y^3 - 6y]")
    print(f"\nHessian: H(x,y) = [[12x^2 - 6, 0], [0, 12y^2 - 6]]")
    print(f"\nsqrt(3/2) = {np.sqrt(3/2):.6f}")
    print()
    
    print(f"{'No':<4} {'(x, y)':<30} {'f(x,y)':<12} {'lam1':<12} {'lam2':<12} {'Klasifikasi'}")
    print("-" * 80)
    
    for i, (x, y) in enumerate(stationary_points):
        fval = f(x, y)
        H = hessian_f(x, y)
        eigenvalues = np.linalg.eigvalsh(H)
        lam1, lam2 = eigenvalues[0], eigenvalues[1]
        
        # Klasifikasi berdasarkan nilai eigen
        if lam1 > 0 and lam2 > 0:
            classification = "Minimum Lokal"
        elif lam1 < 0 and lam2 < 0:
            classification = "Maksimum Lokal"
        elif lam1 * lam2 < 0:
            classification = "Saddle Point"
        else:
            classification = "Tidak Tentu"
        
        print(f"{i+1:<4} ({x:>8.4f}, {y:>8.4f})    {fval:>10.4f}  {lam1:>10.4f}  {lam2:>10.4f}  {classification}")
    
    return stationary_points

# ============================================================
# BAGIAN 3: Backtracking Line Search (Armijo)
# ============================================================

def backtracking_line_search(x, y, d, alpha_init=1.0, rho=0.5, c=1e-4):
    """
    Armijo backtracking line search.
    Mencari alpha sehingga f(x + alpha*d) <= f(x) + c*alpha*grad_f(x)'*d
    """
    alpha = alpha_init
    grad = grad_f(x, y)
    f_current = f(x, y)
    slope = grad @ d
    
    max_iter = 100
    for _ in range(max_iter):
        x_new = x + alpha * d[0]
        y_new = y + alpha * d[1]
        if f(x_new, y_new) <= f_current + c * alpha * slope:
            break
        alpha *= rho
    
    return alpha

# ============================================================
# BAGIAN 4: Steepest Descent
# ============================================================

def steepest_descent(x0, y0, tol=1e-8, max_iter=50000):
    """
    Algoritma Steepest Descent (Gradient Descent) dengan backtracking line search.
    Arah pencarian: d = -∇f(x,y)
    """
    x, y = x0, y0
    path = [(x, y)]
    
    for i in range(max_iter):
        g = grad_f(x, y)
        grad_norm = np.linalg.norm(g)
        
        if grad_norm < tol:
            break
        
        # Arah steepest descent
        d = -g
        
        # Line search
        alpha = backtracking_line_search(x, y, d)
        
        x = x + alpha * d[0]
        y = y + alpha * d[1]
        path.append((x, y))
    
    return np.array(path), i + 1

# ============================================================
# BAGIAN 5: Newton Method
# ============================================================

def newton_method(x0, y0, tol=1e-8, max_iter=1000):
    """
    Newton Method (sesuai Lecture 3, hal. 36-37):
    Arah pencarian: d = -H(x,y)^{-1} * nabla f(x,y)
    Step length: alpha = 1 (pure Newton step)
    Update rule: x^{k+1} = x^{k} + d^{k} = x^{k} - H^{-1} nabla f
    
    Jika Hessian tidak positif definit, gunakan regularisasi
    (Lecture 3, hal. 48-49: overcome singularity/non-PD Hessian).
    """
    x, y = x0, y0
    path = [(x, y)]
    
    for i in range(max_iter):
        g = grad_f(x, y)
        grad_norm = np.linalg.norm(g)
        
        if grad_norm < tol:
            break
        
        H = hessian_f(x, y)
        
        # Cek apakah Hessian positif definit
        eigvals = np.linalg.eigvalsh(H)
        
        if np.all(eigvals > 1e-10):
            # Pure Newton direction
            d = -np.linalg.solve(H, g)
        else:
            # Regularisasi agar positif definit (Lecture 3, hal. 48-49)
            tau = max(0, -min(eigvals) + 0.1)
            H_mod = H + tau * np.eye(2)
            d = -np.linalg.solve(H_mod, g)
        
        # Pure Newton step: alpha = 1 (Lecture 3, hal. 36)
        x = x + d[0]
        y = y + d[1]
        path.append((x, y))
    
    return np.array(path), i + 1

# ============================================================
# BAGIAN 6: Plotting & Perbandingan
# ============================================================

def plot_contour_with_trajectories(starting_points, output_dir):
    """Plot kontur fungsi dan lintasan kedua algoritma untuk setiap titik awal."""
    
    # Grid untuk kontur
    x_grid = np.linspace(-2.5, 2.5, 400)
    y_grid = np.linspace(-2.5, 2.5, 400)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)
    
    # Level kontur
    levels = np.linspace(-3, 10, 40)
    
    for idx, (x0, y0) in enumerate(starting_points):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Lintasan Optimisasi - Titik Awal ({x0}, {y0})', fontsize=15, fontweight='bold')
        
        # === Steepest Descent ===
        path_sd, iter_sd = steepest_descent(x0, y0)
        
        ax = axes[0]
        cs = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
        ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
        ax.plot(path_sd[:, 0], path_sd[:, 1], 'r.-', linewidth=1.2, markersize=3, label='Lintasan SD')
        ax.plot(path_sd[0, 0], path_sd[0, 1], 'go', markersize=10, label='Titik Awal', zorder=5)
        ax.plot(path_sd[-1, 0], path_sd[-1, 1], 'r*', markersize=15, label='Solusi Optimal', zorder=5)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Steepest Descent\nIterasi: {iter_sd}, f* = {f(path_sd[-1,0], path_sd[-1,1]):.6f}', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # === Newton Method ===
        path_nm, iter_nm = newton_method(x0, y0)
        
        ax = axes[1]
        cs = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
        ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
        ax.plot(path_nm[:, 0], path_nm[:, 1], 'b.-', linewidth=1.2, markersize=3, label='Lintasan NM')
        ax.plot(path_nm[0, 0], path_nm[0, 1], 'go', markersize=10, label='Titik Awal', zorder=5)
        ax.plot(path_nm[-1, 0], path_nm[-1, 1], 'b*', markersize=15, label='Solusi Optimal', zorder=5)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Newton Method\nIterasi: {iter_nm}, f* = {f(path_nm[-1,0], path_nm[-1,1]):.6f}', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'soal1_trajectory_start{idx+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n--- Titik Awal ({x0}, {y0}) ---")
        print(f"  Steepest Descent: {iter_sd} iterasi -> (x*, y*) = ({path_sd[-1,0]:.6f}, {path_sd[-1,1]:.6f}), f* = {f(path_sd[-1,0], path_sd[-1,1]):.6f}")
        print(f"  Newton Method  : {iter_nm} iterasi -> (x*, y*) = ({path_nm[-1,0]:.6f}, {path_nm[-1,1]:.6f}), f* = {f(path_nm[-1,0], path_nm[-1,1]):.6f}")
        
        if not np.allclose([path_sd[-1,0], path_sd[-1,1]], [path_nm[-1,0], path_nm[-1,1]], atol=0.1):
            print(f"  [!] Kedua algoritma KONVERGEN ke titik BERBEDA!")
        else:
            print(f"  [OK] Kedua algoritma konvergen ke titik yang sama.")

def plot_combined_comparison(starting_points, output_dir):
    """Plot gabungan perbandingan kedua algoritma."""
    
    x_grid = np.linspace(-2.5, 2.5, 400)
    y_grid = np.linspace(-2.5, 2.5, 400)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)
    levels = np.linspace(-3, 10, 40)
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Perbandingan Steepest Descent vs Newton Method', fontsize=16, fontweight='bold')
    
    for idx, (x0, y0) in enumerate(starting_points):
        path_sd, iter_sd = steepest_descent(x0, y0)
        path_nm, iter_nm = newton_method(x0, y0)
        
        ax = axes[idx]
        ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
        ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
        
        ax.plot(path_sd[:, 0], path_sd[:, 1], 'r.-', linewidth=1.5, markersize=4, label=f'SD ({iter_sd} iter)')
        ax.plot(path_nm[:, 0], path_nm[:, 1], 'b.-', linewidth=1.5, markersize=4, label=f'NM ({iter_nm} iter)')
        ax.plot(x0, y0, 'go', markersize=12, label='Start', zorder=5)
        ax.plot(path_sd[-1, 0], path_sd[-1, 1], 'r*', markersize=15, zorder=5)
        ax.plot(path_nm[-1, 0], path_nm[-1, 1], 'b*', markersize=15, zorder=5)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Titik Awal ({x0}, {y0})', fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'soal1_comparison_all.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# BAGIAN 7: Main
# ============================================================

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 80)
    print("SOAL 1: OPTIMISASI FUNGSI NON-LINEAR")
    print("f(x,y) = x^4 + y^4 - 3x^2 - 3y^2 + 2")
    print("=" * 80)
    
    # Bagian 1-4: Analisis titik stasioner
    stationary_points = analyze_stationary_points()
    
    # Bagian 5-9: Implementasi algoritma dan plotting
    starting_points = [(2.0, 2.0), (-0.3, 0.2), (-0.5, -1.5)]
    
    print("\n" + "=" * 80)
    print("PERBANDINGAN STEEPEST DESCENT vs NEWTON METHOD")
    print("=" * 80)
    
    plot_contour_with_trajectories(starting_points, output_dir)
    plot_combined_comparison(starting_points, output_dir)
    
    # Tabel ringkasan
    print("\n" + "=" * 80)
    print("TABEL RINGKASAN PERBANDINGAN")
    print("=" * 80)
    print(f"\n{'Titik Awal':<18} {'Metode':<20} {'Iterasi':<10} {'(x*, y*)':<30} {'f*':<12}")
    print("-" * 90)
    
    for x0, y0 in starting_points:
        path_sd, iter_sd = steepest_descent(x0, y0)
        path_nm, iter_nm = newton_method(x0, y0)
        
        sd_sol = f"({path_sd[-1,0]:.6f}, {path_sd[-1,1]:.6f})"
        nm_sol = f"({path_nm[-1,0]:.6f}, {path_nm[-1,1]:.6f})"
        
        print(f"({x0:>4.1f}, {y0:>4.1f})     {'Steepest Descent':<20} {iter_sd:<10} {sd_sol:<30} {f(path_sd[-1,0], path_sd[-1,1]):<12.6f}")
        print(f"{'':18} {'Newton Method':<20} {iter_nm:<10} {nm_sol:<30} {f(path_nm[-1,0], path_nm[-1,1]):<12.6f}")
        print()
    
    print("\n" + "=" * 80)
    print("ANALISIS PERBEDAAN LINTASAN")
    print("=" * 80)
    print("""
Penjelasan mengapa lintasan kedua algoritma dapat berbeda:

1. ARAH PENCARIAN BERBEDA:
   - Steepest Descent menggunakan arah d = -grad(f), yaitu arah negatif gradien.
     Arah ini selalu tegak lurus terhadap garis kontur di titik tersebut.
   - Newton Method menggunakan arah d = -H^{-1} grad(f), yang memperhitungkan 
     kelengkungan (curvature) fungsi melalui informasi orde kedua (Hessian).
     
2. KECEPATAN KONVERGENSI:
   - Steepest Descent memiliki konvergensi linear (Lec 3, hal. 53), sehingga 
     cenderung menghasilkan pola zig-zag (terutama di daerah lembah sempit).
   - Newton Method memiliki konvergensi kuadratik di dekat solusi optimal,
     sehingga konvergen jauh lebih cepat. Namun konvergensi kuadratik hanya
     berlaku ketika Hessian positif definit dan iterasi sudah dekat solusi.

3. OVERSHOOTING PADA PURE NEWTON (alpha = 1):
   - Pure Newton (alpha=1, tanpa line search, Lec 3 hal. 36) dapat menghasilkan
     langkah yang sangat besar ketika Hessian tidak positif definit.
   - Titik awal (-0.3, 0.2): Hessian negative definite (eig=[-5.52,-4.92]),
     dekat maximum lokal. Regularisasi menghasilkan langkah d=(-2.42, 11.68),
     iterasi pertama melompat ke y ~ 11.88 (jauh keluar area plot).
   - Titik awal (-0.5, -1.5): Hessian indefinite (eig=[-3.00, 21.00]),
     dekat saddle point. Regularisasi menghasilkan langkah d=(-25.0, 0.19),
     iterasi pertama melompat ke x ~ -25.5 (jauh keluar area plot).
   - Ini adalah kelemahan yang diketahui dari Newton Method (Lec 3, hal. 46-48).
     Meskipun melompat jauh, iterasi berikutnya kembali ke minimum karena di
     titik-titik jauh Hessian sudah positif definit.

4. SENSITIVITAS TERHADAP TITIK AWAL:
   - Pada titik awal (2.0, 2.0), SD konvergen ke (-1.2247,-1.2247) sedangkan
     Newton konvergen ke (1.2247, 1.2247) -- keduanya f* = -2.5 namun lokasi
     berbeda. Ini karena perbedaan arah pencarian menempatkan kedua algoritma
     pada basin of attraction yang berbeda.

5. REGULARISASI HESSIAN PADA DAERAH NON-CONVEX:
   - Ketika Hessian tidak positif definit, digunakan regularisasi H + tau*I
     (Lecture 3, hal. 48-49) dengan tau = max(0, -lambda_min + 0.1).
   - Regularisasi mengubah arah pencarian dan besarnya langkah, menyebabkan
     lintasan Newton berbeda drastis dari Steepest Descent.
""")
    
    print("Semua plot disimpan di:", output_dir)
    print("File: soal1_trajectory_start1.png, soal1_trajectory_start2.png,")
    print("      soal1_trajectory_start3.png, soal1_comparison_all.png")
