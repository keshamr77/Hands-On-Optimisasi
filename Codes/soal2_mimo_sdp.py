# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Soal 2: Sistem MIMO dan Relaksasi SDP
Deteksi Maximum-Likelihood (ML) untuk sistem MIMO 2×2 BPSK
dan relaksasi Semidefinite Programming (SDP).

Model: y = Hs + n
- H ∈ R^{2×2}: matriks kanal
- s = [s1, s2]^T, s_i ∈ {-1, 1} (BPSK)
- y ∈ R^{2×1}: sinyal terima
- n: AWGN
"""

import numpy as np
import cvxpy as cp
import itertools
import os

# ============================================================
# BAGIAN 1: Deteksi ML Eksak (Exhaustive Search)
# ============================================================

def ml_detection_exact(H, y):
    """
    Deteksi ML eksak via exhaustive search.
    min_{s_i ∈ {-1,1}} ||y - Hs||^2
    
    Untuk BPSK 2×2, ada 2^2 = 4 kemungkinan vektor s.
    """
    candidates = list(itertools.product([-1, 1], repeat=2))
    best_s = None
    best_cost = np.inf
    
    results = []
    for s_candidate in candidates:
        s = np.array(s_candidate, dtype=float)
        cost = np.linalg.norm(y - H @ s)**2
        results.append((s, cost))
        if cost < best_cost:
            best_cost = cost
            best_s = s.copy()
    
    return best_s, best_cost, results

# ============================================================
# BAGIAN 2: Formulasi dan Solusi SDP Relaxation
# ============================================================

def sdp_relaxation(H, y):
    """
    Relaksasi SDP dari masalah deteksi ML.
    
    FORMULASI:
    ----------
    Masalah asli: min ||y - Hs||^2  s.t. s_i ∈ {-1, 1}
    
    Ekspansi:
    ||y - Hs||^2 = y^T y - 2y^T H s + s^T H^T H s
                 = y^T y - 2y^T H s + tr(H^T H s s^T)
    
    Definisikan:
    - Q = H^T H
    - c = H^T y
    - W = s s^T  (rank-1 PSD matrix)
    
    Maka: ||y - Hs||^2 = y^T y - 2 c^T s + tr(Q W)
    
    Dengan memperkenalkan variabel tambahan:
    Definisikan ŝ = [s; 1] ∈ R^3 dan matriks:
    
    X = ŝ ŝ^T = [[W, s], [s^T, 1]]  ∈ R^{3×3}
    
    Konstrain:
    - X ≽ 0 (positive semidefinite)
    - X_{ii} = 1 untuk semua i (karena s_i^2 = 1 dan 1^2 = 1)
    - rank(X) = 1  ← ini yang direlaksasi!
    
    Relaksasi SDP menghapus konstrain rank-1:
    
    min  tr(C X)
    s.t. X ≽ 0
         X_{ii} = 1,  i = 1, 2, 3
    
    dimana: C = [[H^T H, -H^T y], [-y^T H, y^T y]]
    """
    n = H.shape[1]  # jumlah simbol = 2
    
    # Buat matriks C untuk fungsi objektif
    Q = H.T @ H
    c = H.T @ y
    
    # Matriks C (n+1) x (n+1)
    C = np.zeros((n+1, n+1))
    C[:n, :n] = Q
    C[:n, n] = -c.flatten()
    C[n, :n] = -c.flatten()
    C[n, n] = y.T @ y
    
    # Variabel SDP
    X = cp.Variable((n+1, n+1), symmetric=True)
    
    # Fungsi objektif: min tr(C @ X)
    objective = cp.Minimize(cp.trace(C @ X))
    
    # Konstrain
    constraints = [
        X >> 0,                          # X ≽ 0 (PSD)
    ]
    # Konstrain diagonal X_{ii} = 1
    for i in range(n+1):
        constraints.append(X[i, i] == 1)
    
    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    
    X_opt = X.value
    
    # Ekstrak solusi s dari X
    # Metode: ambil s dari kolom terakhir X (yaitu X[0:n, n])
    s_relaxed = X_opt[:n, n]
    
    # Rounding: s_detected = sign(s_relaxed)
    s_detected = np.sign(s_relaxed)
    
    # Hitung nilai objektif sebenarnya
    cost_detected = np.linalg.norm(y - H @ s_detected)**2
    
    return s_detected, cost_detected, X_opt, s_relaxed, problem.value

# ============================================================
# BAGIAN 3: Analisis Konveksitas
# ============================================================

def print_convexity_analysis():
    """Penjelasan konveksitas formulasi SDP."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              ANALISIS KONVEKSITAS FORMULASI SDP                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║ FUNGSI OBJEKTIF: tr(C·X)                                        ║
║ - Fungsi trace(C·X) adalah fungsi LINEAR terhadap X              ║
║ - Setiap fungsi linear bersifat konveks (dan juga konkaf)        ║
║ - Oleh karena itu, fungsi objektif SDP bersifat KONVEKS          ║
║                                                                  ║
║ KONSTRAIN:                                                       ║
║ 1) X ≽ 0 (positive semidefinite):                                ║
║    - Cone of PSD matrices merupakan CONVEX CONE                  ║
║    - Himpunan {X : X ≽ 0} adalah himpunan konveks               ║
║                                                                  ║
║ 2) X_{ii} = 1 (diagonal constraints):                           ║
║    - Ini adalah konstrain AFFINE (linear equality)               ║
║    - Himpunan solusi konstrain affine selalu konveks             ║
║                                                                  ║
║ KESIMPULAN:                                                      ║
║ Karena fungsi objektif linear (konveks) dan semua konstrain      ║
║ mendefinisikan himpunan konveks, maka formulasi SDP ini          ║
║ merupakan masalah OPTIMISASI KONVEKS.                            ║
║                                                                  ║
║ Relaksasi SDP memberikan LOWER BOUND pada masalah ML asli,      ║
║ karena himpunan feasible SDP (tanpa rank-1) merupakan            ║
║ SUPERSET dari himpunan feasible asli.                            ║
╚══════════════════════════════════════════════════════════════════╝
""")

# ============================================================
# BAGIAN 4: Main - Eksekusi kedua kasus
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SOAL 2: SISTEM MIMO 2x2 DAN RELAKSASI SDP")
    print("Model: y = Hs + n,  s_i in {-1, +1} (BPSK)")
    print("=" * 80)
    
    # --- Analisis Konveksitas ---
    print_convexity_analysis()
    
    # ========================
    # KASUS 1
    # ========================
    H1 = np.array([[1.2, -0.6],
                    [0.5,  1.0]])
    y1 = np.array([[-1.7],
                    [ 0.6]])
    
    print("=" * 80)
    print("KASUS 1")
    print(f"H = {H1.tolist()}")
    print(f"y = {y1.flatten().tolist()}")
    print("=" * 80)
    
    # ML Eksak
    s_ml1, cost_ml1, all_results1 = ml_detection_exact(H1, y1.flatten())
    print("\n--- Deteksi ML Eksak (Exhaustive Search) ---")
    print(f"{'s':<15} {'||y - Hs||^2':<15}")
    print("-" * 30)
    for s, cost in all_results1:
        marker = " <-- MINIMUM" if np.allclose(s, s_ml1) else ""
        print(f"[{s[0]:>2.0f}, {s[1]:>2.0f}]     {cost:>10.6f}{marker}")
    print(f"\nSolusi ML: s* = {s_ml1}, ||y - Hs*||^2 = {cost_ml1:.6f}")
    
    # SDP Relaxation
    s_sdp1, cost_sdp1, X_opt1, s_relaxed1, sdp_obj1 = sdp_relaxation(H1, y1.flatten())
    print("\n--- Relaksasi SDP ---")
    print(f"Solusi SDP relaksasi (sebelum rounding): ŝ = {s_relaxed1}")
    print(f"Solusi SDP (setelah rounding): s* = {s_sdp1}")
    print(f"Nilai objektif SDP (lower bound): {sdp_obj1:.6f}")
    print(f"Nilai objektif sebenarnya ||y - Hs*||^2: {cost_sdp1:.6f}")
    print(f"\nMatriks X optimal:")
    print(np.array2string(X_opt1, precision=4, suppress_small=True))
    
    # Bandingkan
    print(f"\n--- Perbandingan Kasus 1 ---")
    if np.allclose(s_ml1, s_sdp1):
        print(f"[OK] Kedua metode menghasilkan solusi SAMA: s* = {s_ml1}")
    else:
        print(f"[X] Kedua metode menghasilkan solusi BERBEDA!")
        print(f"  ML : s* = {s_ml1}")
        print(f"  SDP: s* = {s_sdp1}")
    
    # ========================
    # KASUS 2
    # ========================
    H2 = np.array([[1.0,  0.99],
                    [0.99, 1.0 ]])
    y2 = np.array([[ 0.2],
                    [-0.1]])
    
    print("\n" + "=" * 80)
    print("KASUS 2")
    print(f"H = {H2.tolist()}")
    print(f"y = {y2.flatten().tolist()}")
    print("=" * 80)
    
    # ML Eksak
    s_ml2, cost_ml2, all_results2 = ml_detection_exact(H2, y2.flatten())
    print("\n--- Deteksi ML Eksak (Exhaustive Search) ---")
    print(f"{'s':<15} {'||y - Hs||^2':<15}")
    print("-" * 30)
    for s, cost in all_results2:
        marker = " <-- MINIMUM" if np.allclose(s, s_ml2) else ""
        print(f"[{s[0]:>2.0f}, {s[1]:>2.0f}]     {cost:>10.6f}{marker}")
    print(f"\nSolusi ML: s* = {s_ml2}, ||y - Hs*||^2 = {cost_ml2:.6f}")
    
    # SDP Relaxation
    s_sdp2, cost_sdp2, X_opt2, s_relaxed2, sdp_obj2 = sdp_relaxation(H2, y2.flatten())
    print("\n--- Relaksasi SDP ---")
    print(f"Solusi SDP relaksasi (sebelum rounding): ŝ = {s_relaxed2}")
    print(f"Solusi SDP (setelah rounding): s* = {s_sdp2}")
    print(f"Nilai objektif SDP (lower bound): {sdp_obj2:.6f}")
    print(f"Nilai objektif sebenarnya ||y - Hs*||^2: {cost_sdp2:.6f}")
    print(f"\nMatriks X optimal:")
    print(np.array2string(X_opt2, precision=4, suppress_small=True))
    
    # Eigenvalue analysis
    eigvals_X2 = np.linalg.eigvalsh(X_opt2)
    print(f"\nNilai eigen X optimal: {eigvals_X2}")
    
    # Bandingkan
    print(f"\n--- Perbandingan Kasus 2 ---")
    if np.allclose(s_ml2, s_sdp2):
        print(f"[OK] Kedua metode menghasilkan solusi SAMA: s* = {s_ml2}")
    else:
        print(f"[X] Kedua metode menghasilkan solusi BERBEDA!")
        print(f"  ML : s* = {s_ml2}, cost = {cost_ml2:.6f}")
        print(f"  SDP: s* = {s_sdp2}, cost = {cost_sdp2:.6f}")
    
    # ========================
    # ANALISIS PERBANDINGAN
    # ========================
    print("\n" + "=" * 80)
    print("ANALISIS PERBANDINGAN DAN PENJELASAN")
    print("=" * 80)
    
    # Hitung condition number
    cond1 = np.linalg.cond(H1)
    cond2 = np.linalg.cond(H2)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ANALISIS PERBANDINGAN                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║ Kasus 1: H = [[1.2, -0.6], [0.5, 1.0]]                        ║
║   Condition number H: {cond1:.4f}                                      ║
║   → Kanal well-conditioned, kolom H hampir ortogonal            ║
║   → Relaksasi SDP menghasilkan solusi dekat rank-1              ║
║   → Rounding berhasil: SDP = ML ✓                               ║
║                                                                  ║
║ Kasus 2: H = [[1.0, 0.99], [0.99, 1.0]]                       ║
║   Condition number H: {cond2:.4f}                                     ║
║   → Kanal ILL-CONDITIONED: kolom H hampir paralel              ║
║   → Matriks H^T H hampir singular                               ║
║   → Relaksasi SDP lebih "loose" (gap lebih besar)               ║
║   → Rounding mungkin gagal menghasilkan solusi ML optimal       ║
║                                                                  ║
║ PENJELASAN:                                                      ║
║ - SDP relaxation menghapus konstrain rank(X) = 1                ║
║ - Untuk kanal well-conditioned, solusi SDP cenderung             ║
║   mendekati rank-1 → rounding tepat                              ║
║ - Untuk kanal ill-conditioned, solusi SDP bisa jauh dari        ║
║   rank-1 → rounding bisa menghasilkan suboptimal                ║
║ - Kasus 2 memiliki kolom kanal hampir identik, membuat          ║
║   pemisahan s1 dan s2 sangat sulit                               ║
║ - SDP selalu memberikan LOWER BOUND pada solusi ML optimal      ║
╚══════════════════════════════════════════════════════════════════╝
""")
