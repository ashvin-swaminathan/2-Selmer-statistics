#!/usr/bin/env python3
"""
matmethod_hyperfast_slurm.py

SLURM-optimized version of matmethod_hyperfast.py.

Improvements for cluster computing:
1. Command-line arguments for N value
2. Checkpointing to save/resume progress
3. SLURM-aware worker count
4. Unique output files per job
5. Graceful shutdown on SIGTERM (SLURM timeout)
"""

import sys
import os
import time
import signal
import pickle
import argparse
import itertools
import subprocess
import multiprocessing
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

# ===== SLURM ENVIRONMENT =====
SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
SLURM_CPUS = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

# ===== GLOBAL STATE FOR CHECKPOINTING =====
checkpoint_requested = False

def signal_handler(signum, frame):
    """Handle SIGTERM (sent by SLURM before killing job)"""
    global checkpoint_requested
    print(f"\n[SIGNAL] Received signal {signum}, will checkpoint and exit...")
    checkpoint_requested = True

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ===== SETTINGS =====
VERBOSE = False
# ====================

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:8.2f}s] {msg}", flush=True)

# --- Helper Functions ---

_memo = {}
def get_precomputed(key, func, *args):
    if key not in _memo: _memo[key] = func(*args)
    return _memo[key]

def sorted_vertices(n):
    return get_precomputed(f'sv_{n}', lambda n: sorted(list(itertools.product([0,1], repeat=n)), key=lambda v:(sum(v),v)), n)

def vindex(n):
    return get_precomputed(f'vi_{n}', lambda n: {v:i for i, v in enumerate(sorted_vertices(n))}, n)

def get_k_faces(n, k):
    if not (0 <= k <= n): return []
    key = f'faces_{n}_{k}'
    if key in _memo: return _memo[key]
    idx = vindex(n)
    faces = []
    for varying_axes in itertools.combinations(range(n), k):
        fixed_axes = [i for i in range(n) if i not in varying_axes]
        for fixed_values in itertools.product([0,1], repeat=n-k):
            base = {ax: val for ax,val in zip(fixed_axes, fixed_values)}
            verts = []
            for tv in itertools.product([0,1], repeat=k):
                m = base.copy()
                for ii, ax in enumerate(varying_axes): m[ax] = tv[ii]
                verts.append(idx[tuple(m[i] for i in range(n))])
            faces.append(tuple(sorted(verts)))
    faces = sorted(list(set(faces)))
    _memo[key] = faces
    return faces

# --- Compressed Config Representation ---

@dataclass(frozen=True)
class CompressedConfig:
    """
    Memory-efficient config representation. Instead of storing full 2^n tuple,
    we only store:
    - rank_counts: histogram of rank values (size n+1)
    - coeffs_eq: LP constraint coefficients for base 2 (size n)
    - coeffs_ineq: LP constraint coefficients for base 4 (size n)

    Memory: O(n) instead of O(2^n) per config!
    """
    rank_counts: tuple  # Length n+1: count of vertices with each rank
    coeffs_eq: tuple    # Length n: constraint coefficients for k=1..n with base 2
    coeffs_ineq: tuple  # Length n: constraint coefficients for k=1..n with base 4

def compute_compressed_config(F, n):
    """
    Given a full config F (vertex ranks), compute the compressed representation.
    """
    # 1. Compute rank histogram
    rank_counts = np.zeros(n + 1, dtype=np.int32)
    for rank in F:
        rank_counts[rank] += 1
    rank_counts = tuple(int(x) for x in rank_counts)

    # 2. Compute LP coefficients for all k
    coeffs_eq = []
    coeffs_ineq = []

    for k in range(1, n + 1):
        all_k_faces = get_k_faces(n, k)
        if not all_k_faces:
            coeffs_eq.append(0)
            coeffs_ineq.append(0)
            continue

        sum_base2 = 0
        sum_base4 = 0
        for face_indices in all_k_faces:
            min_rank = min(F[v_idx] for v_idx in face_indices)
            sum_base2 += 2 ** min_rank
            sum_base4 += 4 ** min_rank

        coeffs_eq.append(sum_base2)
        coeffs_ineq.append(sum_base4)

    return CompressedConfig(
        rank_counts=rank_counts,
        coeffs_eq=tuple(coeffs_eq),
        coeffs_ineq=tuple(coeffs_ineq)
    )

# --- Core Matrix Logic ---

def _require_tool(name):
    try:
        subprocess.run([name, "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except FileNotFoundError:
        raise RuntimeError(f"External tool '{name}' not found.")

def geng_stream(n):
    """Stream graphs one at a time - O(1) memory for enumeration"""
    proc = subprocess.Popen(["geng", "-q", str(n)], stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        g6 = line.strip()
        if g6: yield g6
    proc.wait()

def graph6_to_adj(g6):
    n_ord = ord(g6[0])
    if 63 <= n_ord <= 126:
        n = n_ord - 63
        offset = 1
        if n > 62:
            n = ((n_ord - 63) << 12) | ((ord(g6[2]) - 63) << 6) | (ord(g6[3]) - 63)
            offset = 4
    else: raise ValueError("Invalid g6")

    A = [[0]*n for _ in range(n)]
    bits_needed = n*(n-1)//2
    vals = [ord(c)-63 for c in g6[offset:]]
    bits = []
    for v in vals:
        for k in range(5,-1,-1): bits.append((v>>k)&1)

    idx = 0
    for j in range(1,n):
        for i in range(j):
            if idx < len(bits):
                A[i][j] = A[j][i] = bits[idx]
                idx += 1
    return A

def bit_matrix_rank_optimized(matrix_rows, n_cols):
    """Highly Optimized Rank over GF(2)."""
    if not matrix_rows: return 0
    rank = 0
    rows = list(matrix_rows)
    num_rows = len(rows)

    for col in range(n_cols):
        if rank >= num_rows: break

        pivot_row = -1
        mask = 1 << col

        for i in range(rank, num_rows):
            if rows[i] & mask:
                pivot_row = i
                break

        if pivot_row != -1:
            rows[rank], rows[pivot_row] = rows[pivot_row], rows[rank]
            pivot_val = rows[rank]

            for i in range(rank + 1, num_rows):
                if rows[i] & mask:
                    rows[i] ^= pivot_val

            rank += 1

    return rank

def build_config_via_matrix(A, n):
    M_rows = []
    for i in range(n):
        row_val = 0
        for j in range(n):
            if i != j and A[i][j] == 0:
                row_val |= (1 << j)
        M_rows.append(row_val)

    V = sorted_vertices(n)
    F = [0] * (1 << n)

    for idx_out, v_tuple in enumerate(V):
        T_indices = [i for i, bit in enumerate(v_tuple) if bit]
        k = len(T_indices)

        if k <= 1:
            F[idx_out] = k
            continue

        sub_rows = []
        keep_mask = 0
        for idx in T_indices:
            keep_mask |= (1 << idx)

        for r_idx in T_indices:
            val = M_rows[r_idx] & keep_mask
            sub_rows.append(val)

        mat_rank = bit_matrix_rank_optimized(sub_rows, n)
        F[idx_out] = k - mat_rank

    return tuple(F)

def process_graph_task_matrix(args):
    g6, n = args
    try:
        A = graph6_to_adj(g6)
        cfg = build_config_via_matrix(A, n)
        compressed = compute_compressed_config(cfg, n)
        return (True, g6, compressed)
    except Exception as e:
        return (False, g6, str(e))

# --- Checkpointing ---

def get_checkpoint_path(n, job_id):
    return Path(f"checkpoint_n{n}_{job_id}.pkl")

def save_checkpoint(n, job_id, processed_count, unique_configs, phase="graphs"):
    """Save current progress to checkpoint file (count-based, memory efficient)"""
    checkpoint_path = get_checkpoint_path(n, job_id)
    checkpoint_data = {
        'n': n,
        'phase': phase,
        'processed_count': processed_count,  # Just store the count, not all graph names
        'unique_configs': list(unique_configs),
        'timestamp': time.time()
    }

    # Write to temp file first, then rename (atomic on POSIX)
    temp_path = checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    temp_path.rename(checkpoint_path)

    log(f"Checkpoint saved: {processed_count} graphs processed, {len(unique_configs)} unique configs")

def load_checkpoint(n, job_id):
    """Load checkpoint if it exists"""
    checkpoint_path = get_checkpoint_path(n, job_id)
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        if data['n'] == n:
            # Handle both old format (processed_graphs) and new format (processed_count)
            if 'processed_count' in data:
                log(f"Resuming from checkpoint: {data['processed_count']} graphs already processed")
            else:
                # Convert old format to new
                data['processed_count'] = len(data.get('processed_graphs', []))
                log(f"Resuming from old checkpoint: {data['processed_count']} graphs already processed")
            return data
    return None

# --- Analysis & LP ---

def sum_exponentiated_min_ranks_over_k_faces(cfg, n, k):
    """Non-vectorized version for single-config calculations"""
    all_k_faces = get_k_faces(n, k)
    if not all_k_faces: return 0
    return sum(2**min(cfg[v_idx] for v_idx in face_indices) for face_indices in all_k_faces)


# --- LP Analysis & Column Generation (with complete pricing) ---

def _objective_coeff(rank_counts, m_target, shift, n):
    """
    Objective coefficient for column (i, shift):
      c_{i,shift} = (#vertices with rank m_target in shifted config) / 2^n.
    Base ranks range is 0..n; shift is >=0.
    """
    base_rank = m_target - shift
    if 0 <= base_rank <= n:
        return rank_counts[base_rank] / (1 << n)
    return 0.0


def _compute_alpha_beta_for_config(i, duals_eq, duals_ub, A_matrix_eq, A_matrix_ineq, num_rels):
    """
    Compute alpha_i and beta_i for tail reduced-cost analysis:
      rc_{i,k} = c_{i,k} - dual_mass - 2^k alpha_i - 4^k beta_i
    where:
      alpha_i = sum_r dual_eq[r] * C(i,r)
      beta_i  = sum_r (mu_r - sigma_r) * D(i,r)
    """
    alpha = 0.0
    for r in range(num_rels):
        alpha += duals_eq[r] * A_matrix_eq[r][i]

    beta = 0.0
    for r in range(num_rels):
        mu = duals_ub[r]
        sigma = duals_ub[num_rels + r]
        beta += (mu - sigma) * A_matrix_ineq[r][i]

    return alpha, beta


def _reduced_cost(i, shift, duals_eq, duals_ub, A_matrix_eq, A_matrix_ineq,
                  rank_counts_i, m_target, n, num_rels):
    """
    Reduced cost for column (i, shift) in the full infinite LP.
    """
    cost = _objective_coeff(rank_counts_i, m_target, shift, n)

    dual_mass = float(duals_eq[num_rels])
    alpha, beta = _compute_alpha_beta_for_config(i, duals_eq, duals_ub, A_matrix_eq, A_matrix_ineq, num_rels)

    # Build exact reduced cost:
    # rc = cost - dual_mass - 2^shift*alpha - 4^shift*beta
    return cost - dual_mass - (2.0 ** shift) * alpha - (4.0 ** shift) * beta


def _tail_candidate_shifts(alpha, beta, dual_mass, m_target):
    """
    Produce a finite candidate set of shifts k >= m_target+1 that contains
    a minimizer of the tail reduced cost (where objective is 0).

    Tail reduced cost (objective=0) is:
      rc_tail(k) = -dual_mass - alpha*2^k - beta*4^k.

    Let t = 2^k, then rc_tail(t) = -dual_mass - alpha*t - beta*t^2.

    Cases:
    - beta > 0: concave down in t; rc_tail -> -infty as t->infty (improving columns exist).
    - beta = 0:
        - alpha > 0: rc_tail -> -infty; improving columns exist.
        - alpha <=0: monotone nondecreasing in k; best is k=m+1.
    - beta < 0:
        - alpha <=0: rc_tail increases in k; best is k=m+1.
        - alpha > 0: convex parabola; minimizer at t* = alpha/(-2beta).
          Candidate integer k are floor(log2 t*), ceil(log2 t*), plus boundary k=m+1.
    """
    k0 = m_target + 1
    cands = {k0}

    if beta > 0.0:
        # diverges to -infty; we'll handle by explicit improvement search in pricing
        return sorted(cands)

    if beta == 0.0:
        if alpha > 0.0:
            # diverges to -infty; handled in pricing search
            return sorted(cands)
        return sorted(cands)

    # beta < 0
    if alpha <= 0.0:
        return sorted(cands)

    # alpha > 0, beta < 0: convex quadratic in t
    t_star = alpha / (-2.0 * beta)
    if t_star <= 0:
        return sorted(cands)

    k_star = np.log2(t_star)
    k_floor = int(np.floor(k_star))
    k_ceil = int(np.ceil(k_star))

    for kk in (k_floor, k_ceil):
        if kk >= k0:
            cands.add(kk)
        else:
            cands.add(k0)

    # also check neighbors (very cheap, avoids edge effects)
    for kk in (k_floor - 1, k_floor + 1, k_ceil - 1, k_ceil + 1):
        if kk >= k0:
            cands.add(kk)

    return sorted(cands)


def _find_improving_tail_shift_if_diverges(alpha, beta, dual_mass, m_target):
    """
    If tail reduced cost diverges to -infty (beta>0 or beta==0 and alpha>0),
    find a finite k >= m+1 such that rc_tail(k) < 0 deterministically.

    Since divergence is exponential, this terminates quickly in practice.
    """
    k = m_target + 1
    while True:
        rc_tail = -dual_mass - alpha * (2.0 ** k) - beta * (4.0 ** k)
        if rc_tail < 0.0:
            return k
        k += 1


def _cert_slack(duals_eq, duals_ub, A_matrix_eq, A_matrix_ineq, B_vector_eq, B_vector_ineq_upper, B_vector_ineq_lower):
    """
    Principled floating-point slack for certificate checking:
    we scale machine epsilon by a magnitude bound from duals and coefficients.

    This is *not* an iteration/tolerance heuristic; it's a guard for roundoff in dot-products.
    """
    eps = np.finfo(float).eps

    mags = [1.0]
    mags.append(float(np.max(np.abs(duals_eq))) if len(duals_eq) else 1.0)
    mags.append(float(np.max(np.abs(duals_ub))) if len(duals_ub) else 1.0)

    # coefficient magnitudes
    if A_matrix_eq:
        mags.append(float(max(max(abs(x) for x in row) for row in A_matrix_eq)) if A_matrix_eq[0] else 1.0)
    if A_matrix_ineq:
        mags.append(float(max(max(abs(x) for x in row) for row in A_matrix_ineq)) if A_matrix_ineq[0] else 1.0)

    mags.append(float(np.max(np.abs(B_vector_eq))) if len(B_vector_eq) else 1.0)
    mags.append(float(np.max(np.abs(B_vector_ineq_upper))) if len(B_vector_ineq_upper) else 1.0)
    mags.append(float(np.max(np.abs(B_vector_ineq_lower))) if len(B_vector_ineq_lower) else 1.0)

    scale = max(mags)
    # 512*eps is a conservative multiple for a few dozen floating ops
    return 512.0 * eps * scale


def certificate_check_infinite_lp(n, m_target, compressed_configs, A_matrix_eq, A_matrix_ineq,
                                  duals_eq, duals_ub, slack):
    """
    Certificate check for the infinite LP at termination.

    We verify:
      (1) For every config i, the tail cannot improve:
          - if beta_i > 0, or beta_i==0 and alpha_i>0, then tail diverges to -infty
            => would imply an improving column exists => FAIL.
      (2) For every config i, min reduced cost over the *complete pricing set* is >= -slack:
          - all k in [0..m_target] (finite)
          - all tail candidates from _tail_candidate_shifts (finite)
            (plus, in diverging cases, we'd already fail in (1)).
    """
    num_configs = len(compressed_configs)
    num_rels = n

    dual_mass = float(duals_eq[num_rels])

    # 1) tail divergence check
    for i in range(num_configs):
        alpha, beta = _compute_alpha_beta_for_config(i, duals_eq, duals_ub, A_matrix_eq, A_matrix_ineq, num_rels)
        if beta > slack:
            return (False, f"FAIL: config {i} has beta={beta} > 0 => tail improves without bound.")
        if abs(beta) <= slack and alpha > slack:
            return (False, f"FAIL: config {i} has beta≈0 and alpha={alpha} > 0 => tail improves without bound.")

    # 2) reduced cost nonnegativity over complete pricing set
    worst = 0.0
    worst_loc = None

    for i in range(num_configs):
        rank_counts_i = compressed_configs[i].rank_counts
        alpha, beta = _compute_alpha_beta_for_config(i, duals_eq, duals_ub, A_matrix_eq, A_matrix_ineq, num_rels)

        # check all k in [0..m_target]
        for k in range(0, m_target + 1):
            rc = _reduced_cost(i, k, duals_eq, duals_ub, A_matrix_eq, A_matrix_ineq, rank_counts_i, m_target, n, num_rels)
            if rc < worst:
                worst = rc
                worst_loc = (i, k, "finite")

        # check tail finite candidates
        tail_cands = _tail_candidate_shifts(alpha, beta, dual_mass, m_target)
        for k in tail_cands:
            rc = _reduced_cost(i, k, duals_eq, duals_ub, A_matrix_eq, A_matrix_ineq, rank_counts_i, m_target, n, num_rels)
            if rc < worst:
                worst = rc
                worst_loc = (i, k, "tail")

    if worst < -slack:
        return (False, f"FAIL: negative reduced cost detected rc={worst} at {worst_loc} (slack={slack}).")

    return (True, f"OK: all checked reduced costs >= -{slack}, worst={worst} at {worst_loc}.")


def run_lp_analysis(n, compressed_configs, f_rels):
    if not compressed_configs: return
    log(f"Building LP constraints for n={n}...")

    f_rels.write(f"--- N={n} ---\n")

    log(f"  Extracting pre-computed coefficients from {len(compressed_configs)} configs...")

    A_matrix_eq = [[] for _ in range(n)]
    A_matrix_ineq = [[] for _ in range(n)]

    for cfg in compressed_configs:
        for k in range(n):
            A_matrix_eq[k].append(cfg.coeffs_eq[k])
            A_matrix_ineq[k].append(cfg.coeffs_ineq[k])

    zero_cfg = [0] * (1 << n)
    B_vector_eq = []
    B_vector_ineq_upper = []
    B_vector_ineq_lower = []

    for k in range(1, n + 1):
        face_count = sum_exponentiated_min_ranks_over_k_faces(zero_cfg, n, k)
        rhs = face_count*(1+2**(1-k))
        B_vector_eq.append(rhs)
        f_rels.write(f"k={k}: sum(2^min) = {rhs}\n")

        term = 1 + 2**(1 - k)
        rhs_ineq = face_count * term * (1 + 2**(2 - k))
        lhs_ineq = face_count * max(term**2, 6 * term - 8)
        B_vector_ineq_upper.append(rhs_ineq)
        B_vector_ineq_lower.append(lhs_ineq)
        f_rels.write(f"k={k}: {lhs_ineq:.2f} <= sum(4^min) <= {rhs_ineq:.2f}\n")

    print(f"--- LP OPTIMIZATION N={n} ---")

    num_configs = len(compressed_configs)
    num_rels = n
    divisor = (1 << n)

    for m_target in range(10):
        if checkpoint_requested:
            log("Checkpoint requested during LP, exiting...")
            return

        print(f"  Solving for rank={m_target}...")

        # Column generation: start with a finite set that guarantees feasibility
        # and includes all shifts that can affect the objective, plus a tail representative.
        active_vars_map = {}

        # Initialize with all k in [0..m_target] for all configs, plus k=m+1.
        init_shifts = list(range(0, m_target + 1)) + [m_target + 1]
        for i in range(num_configs):
            for shift in init_shifts:
                active_vars_map[(i, shift)] = len(active_vars_map)

        while True:
            num_vars = len(active_vars_map)

            c = np.zeros(num_vars, dtype=float)
            A_eq = np.zeros((num_rels + 1, num_vars), dtype=float)
            b_eq = np.zeros(num_rels + 1, dtype=float)

            A_ub = np.zeros((2 * num_rels, num_vars), dtype=float)
            b_ub = np.concatenate([np.array(B_vector_ineq_upper, dtype=float),
                                   -np.array(B_vector_ineq_lower, dtype=float)])

            # Fill columns
            for (i, shift), var_idx in active_vars_map.items():
                c[var_idx] = _objective_coeff(compressed_configs[i].rank_counts, m_target, shift, n)

                # mass constraint
                A_eq[num_rels, var_idx] = 1.0

                scale2 = float(2 ** shift)
                scale4 = float(4 ** shift)
                for r in range(num_rels):
                    A_eq[r, var_idx] = A_matrix_eq[r][i] * scale2
                    val = A_matrix_ineq[r][i] * scale4
                    A_ub[r, var_idx] = val
                    A_ub[num_rels + r, var_idx] = -val

            b_eq[num_rels] = 1.0
            b_eq[:num_rels] = np.array(B_vector_eq, dtype=float)

            res = linprog(
                c,
                A_ub=A_ub, b_ub=b_ub,
                A_eq=A_eq, b_eq=b_eq,
                bounds=(0, None),
                method='highs'
            )

            if not res.success:
                print(f"    [LP Error] {res.message}")
                break

            duals_eq = np.array(res.eqlin.marginals, dtype=float)   # length num_rels+1
            duals_ub = np.array(res.ineqlin.marginals, dtype=float) # length 2*num_rels
            dual_mass = float(duals_eq[num_rels])

            # Complete pricing oracle for the infinite LP:
            # - check all finite shifts k in [0..m_target]
            # - handle tail k>=m_target+1 using finite candidate set (or explicit improving k if diverges)
            min_rc = 0.0
            best_new = None

            for i in range(num_configs):
                rank_counts_i = compressed_configs[i].rank_counts
                alpha, beta = _compute_alpha_beta_for_config(i, duals_eq, duals_ub, A_matrix_eq, A_matrix_ineq, num_rels)

                # (A) finite range: k in [0..m_target]
                for k in range(0, m_target + 1):
                    if (i, k) in active_vars_map:
                        continue
                    rc = _reduced_cost(i, k, duals_eq, duals_ub, A_matrix_eq, A_matrix_ineq,
                                       rank_counts_i, m_target, n, num_rels)
                    if rc < min_rc:
                        min_rc = rc
                        best_new = (i, k)

                # (B) tail: k >= m_target+1
                # If tail diverges to -infty, there MUST be an improving column unless solver is inconsistent.
                if beta > 0.0 or (beta == 0.0 and alpha > 0.0):
                    k_improve = _find_improving_tail_shift_if_diverges(alpha, beta, dual_mass, m_target)
                    if (i, k_improve) not in active_vars_map:
                        rc = _reduced_cost(i, k_improve, duals_eq, duals_ub, A_matrix_eq, A_matrix_ineq,
                                           rank_counts_i, m_target, n, num_rels)
                        if rc < min_rc:
                            min_rc = rc
                            best_new = (i, k_improve)
                    continue

                # Otherwise, finite candidate tail shifts suffice
                tail_cands = _tail_candidate_shifts(alpha, beta, dual_mass, m_target)
                for k in tail_cands:
                    if (i, k) in active_vars_map:
                        continue
                    rc = _reduced_cost(i, k, duals_eq, duals_ub, A_matrix_eq, A_matrix_ineq,
                                       rank_counts_i, m_target, n, num_rels)
                    if rc < min_rc:
                        min_rc = rc
                        best_new = (i, k)

            # Termination: pricing oracle found no improving column for the infinite LP
            if best_new is None:
                # Certificate check (machine-epsilon scaled slack)
                slack = _cert_slack(duals_eq, duals_ub, A_matrix_eq, A_matrix_ineq,
                                    np.array(B_vector_eq, dtype=float),
                                    np.array(B_vector_ineq_upper, dtype=float),
                                    np.array(B_vector_ineq_lower, dtype=float))
                ok, msg = certificate_check_infinite_lp(
                    n=n,
                    m_target=m_target,
                    compressed_configs=compressed_configs,
                    A_matrix_eq=A_matrix_eq,
                    A_matrix_ineq=A_matrix_ineq,
                    duals_eq=duals_eq,
                    duals_ub=duals_ub,
                    slack=slack
                )
                if not ok:
                    print(f"    CERTIFICATE FAILED: {msg}")
                    print(f"    (HiGHS returned a solution, but the dual certificate did not verify.)")
                    break

                print(f"    Optimal (infinite LP): Prop >= {res.fun:.6f}  |  Certificate: {msg}")
                break

            # Add best improving column and continue
            active_vars_map[best_new] = len(active_vars_map)

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description='SLURM-optimized Matrix Method for Hypercubes')
    parser.add_argument('n', type=int, help='Dimension n to compute')
    parser.add_argument('--resume', type=str, default=None,
                        help='Job ID to resume from (uses that job\'s checkpoint)')
    parser.add_argument('--checkpoint-interval', type=int, default=10000,
                        help='Save checkpoint every N graphs (default: 10000)')
    args = parser.parse_args()

    n = args.n
    job_id = args.resume if args.resume else SLURM_JOB_ID
    checkpoint_interval = args.checkpoint_interval

    log(f"SLURM Job ID: {SLURM_JOB_ID}")
    log(f"Using {SLURM_CPUS} CPUs")
    log(f"Computing n={n}")

    try:
        _require_tool("geng")
    except RuntimeError as e:
        print(f"[FATAL] {e}"); sys.exit(1)

    # Output files with job ID
    output_dir = Path(f"output_n{n}_{SLURM_JOB_ID}")
    output_dir.mkdir(exist_ok=True)

    cubes_file = output_dir / "hypercubes_output.txt"
    rels_file = output_dir / "relations_output.txt"

    # Check for existing checkpoint
    checkpoint = load_checkpoint(n, job_id)

    if checkpoint and checkpoint['phase'] == 'lp':
        # Graph processing complete, just need to redo LP
        log("Checkpoint indicates graph processing complete, skipping to LP...")
        unique_compressed_configs = set(checkpoint['unique_configs'])
    else:
        # Resume from checkpoint if available
        if checkpoint:
            skip_count = checkpoint['processed_count']
            unique_compressed_configs = set(checkpoint['unique_configs'])
            log(f"Will skip first {skip_count} graphs from stream")
        else:
            skip_count = 0
            unique_compressed_configs = set()

        # Process graphs using streaming (O(1) memory for enumeration)
        t_gen = time.time()
        count = skip_count
        last_checkpoint = count

        def graph_stream_with_skip():
            """Stream graphs, skipping already-processed ones"""
            skipped = 0
            for g6 in geng_stream(n):
                if skipped < skip_count:
                    skipped += 1
                    if skipped % 100000 == 0:
                        log(f"Skipping... {skipped}/{skip_count}")
                    continue
                yield (g6, n)

        with multiprocessing.Pool(processes=SLURM_CPUS) as pool:
            for success, g6, compressed_cfg in pool.imap_unordered(
                    process_graph_task_matrix, graph_stream_with_skip(), chunksize=100):

                if checkpoint_requested:
                    log("Checkpoint requested, saving and exiting...")
                    save_checkpoint(n, job_id, count, unique_compressed_configs, phase="graphs")
                    sys.exit(0)

                count += 1

                if success:
                    unique_compressed_configs.add(compressed_cfg)

                if count % 5000 == 0:
                    log(f"Graphs processed: {count}...")

                # Periodic checkpoint
                if count - last_checkpoint >= checkpoint_interval:
                    save_checkpoint(n, job_id, count, unique_compressed_configs, phase="graphs")
                    last_checkpoint = count

        log(f"Graphs processed: {count}. Done. ({time.time()-t_gen:.2f}s)")

        # Save checkpoint before LP (LP is fast, but good to have)
        save_checkpoint(n, job_id, count, unique_compressed_configs, phase="lp")

    final_configs = list(unique_compressed_configs)
    log(f"Unique configs: {len(final_configs)}")
    log(f"Memory saved: ~{len(final_configs) * ((1 << n) - 3*n) * 8 / (1024**2):.1f} MB per config type")

    # Write results
    with open(cubes_file, "w") as f_cubes:
        f_cubes.write(f"N={n} Count={len(final_configs)}\n")
        for cfg in final_configs:
            f_cubes.write(f"Ranks: {cfg.rank_counts} | Eq: {cfg.coeffs_eq[:3]}... | Ineq: {cfg.coeffs_ineq[:3]}...\n")

    with open(rels_file, "w") as f_rels:
        run_lp_analysis(n, final_configs, f_rels)

    # Clean up checkpoint on successful completion
    checkpoint_path = get_checkpoint_path(n, job_id)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log("Checkpoint removed (job completed successfully)")

    log("DONE!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
