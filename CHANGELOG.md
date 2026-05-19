# Changelog

All notable changes to the SRCM package are documented here.

---

## [v0.0.4] — 2026-05-19

### Changed

**Inner-loop performance optimisations (`engine.py`)**

Four targeted changes to reduce overhead in the Gillespie hot path:

1. **Removed `state.assert_consistent` from the inner loop.** Previously called twice per Gillespie event (inside `build_propensity_vector` and `apply_event`). State shapes cannot change mid-simulation, so this was pure redundant validation. The single call in `run` at initialisation is sufficient.

2. **Removed duplicate negative-propensity scan.** `_simulate_interval` was running `np.any(propensity < 0)` (an O(N) scan) every step before calling `gillespie_draw`, which performs the identical check internally and raises on failure. The outer scan was eliminated; `gillespie_draw` remains the safety net.

3. **Vectorised `pde_rhs` and precomputed `D/dx²`.** The per-species Python loop calling `L @ C[s]` separately for each species is replaced by a single batched BLAS call `(L @ C.T).T`, scaled by a precomputed `(n_species, 1)` array `_D_over_dx2`. Called 4× per timestep (RK4), the saving compounds.

4. **Vectorised CD/DC propensity blocks and precomputed gamma rates.** The two Python `for`-loops over species in `build_propensity_vector` (with per-iteration `rate_for` calls that each invoke `np.isscalar`) are replaced by two numpy array expressions using a precomputed `(n_species, 1)` column vector `_gamma_arr`.

---

## [v0.0.3] — 2026-05-18

### Added

**Post-PDE-step front cleanup (`enable_front_cleanup`)**

Introduces `_post_pde_cleanup()`, a γ-independent mechanism called once per timestep immediately after each RK4 step. It addresses a fundamental instability in hybrid solvers for pushed-wave systems (e.g. FKPP): any sub-particle PDE mass that diffuses into the stochastic zone is never absorbed by Gillespie CD events fast enough when γ is small, and when γ is large the SSA zone collapses — in both limits the wavespeed drifts toward the deterministic PDE speed, contradicting the Brunet–Derrida prediction.

**Algorithm.** After each RK4 step, for every SSA compartment $k$ and species $s$:

1. Compute $Y_k = \sum_{j \in k} u_j \, dx$ (integrated PDE mass).
2. Compute combined mass $\mathcal{M}_k = N_k + Y_k$.
3. If $\mathcal{M}_k < \theta_\mathrm{CD}$ (stochastic regime) **and** $Y_k \in (0, 1)$ (sub-particle):
   - Zero the PDE slice: $u_j \leftarrow 0$ for all $j \in k$.
   - Create a discrete particle with probability $Y_k$.

Expected mass is conserved: $\mathbb{E}[\text{particles created}] = Y_k$.

**Why this works.** The Brunet–Derrida wavespeed correction requires a hard cutoff at concentrations below $1/N$ — the stochastic discreteness is what slows the wave relative to the PDE. In the SRCM, this cutoff must be enforced at the PDE level too, not only through Gillespie events. The cleanup ensures that no sub-particle PDE mass can participate in RK4 diffusion/reaction steps in the stochastic zone, regardless of $\gamma$. The Gillespie CD events remain active for $Y_k \geq 1$ and coexist with the cleanup for $Y_k < 1$ (where the cleanup dominates because it fires every $\Delta t$).

**New `SRCMEngine` field:**

| Field | Type | Default | Description |
|---|---|---|---|
| `enable_front_cleanup` | `bool` | `True` | Toggle post-PDE-step cleanup. Set `False` to recover pre-v0.0.3 behaviour for comparison. |

### Changed

**Per-cell step-mass CD removal (commit `73bf87d`)**

Replaces the previous probabilistic CD logic with a cleaner per-cell formulation. When a CD Gillespie event fires for compartment $k$:

$$u_j \leftarrow u_j - \min\!\left(\tfrac{1}{h},\, u_j\right) \quad \forall\, j \in k$$

Total mass removed: $M = \sum_j \min(1/h, u_j)\,dx \in [0, 1]$. A discrete particle is created with probability $M$. This guarantees the PDE slice never goes negative and removes at most one particle worth of mass per event regardless of the fine-grid concentration profile.

---

## [v0.0.2] — 2026-05-14

### Changed

**PDE dynamics suppressed below concentration threshold**

When a PDE cell concentration $u_j < 1/h$ (below the single-particle resolution threshold), the cell is excluded from PDE reaction–diffusion dynamics during the RK4 step. This prevents the deterministic PDE from propagating genuinely sub-resolution mass, which would otherwise seed spurious wave propagation ahead of the stochastic front.

This was motivated by FKPP wavespeed experiments showing that even with probabilistic CD conversion (v0.0.1), tiny PDE concentrations at the wave tip caused the measured speed to creep upward toward $c^* = 2\sqrt{D\lambda}$ when $\gamma$ was small.

**Note:** This version suppresses sub-threshold PDE dynamics but does not remove the sub-particle mass from the grid between Gillespie events. The full fix for γ-independent front behaviour arrives in v0.0.3.

---

## [v0.0.1] — 2026-05-14

### Added

Initial release of the SRCM package. Full hybrid SSA–PDE solver implementing the Spatial Regime Conversion Method of Cameron, Smith & Yates (2025).

**Core engine (`SRCMEngine`)**

- Coupled SSA/PDE state on a dual grid: coarse SSA compartments of size $h$, fine PDE grid of spacing $dx = h / P$ where $P$ is `pde_multiple`.
- RK4 time integration for the PDE.
- Gillespie SSA for diffusion, conversion, and hybrid reaction events.
- Block-structured propensity vector layout:
  - Blocks 0 to $S-1$: SSA diffusion (one block per species)
  - Blocks $S$ to $2S-1$: CD conversion (PDE → SSA)
  - Blocks $2S$ to $3S-1$: DC conversion (SSA → PDE)
  - Blocks $3S$ onward: hybrid reactions (mixed SSA/PDE propensities)

**Two-threshold hysteresis conversion**

Conversion between regimes is controlled by two thresholds with $\theta_\mathrm{DC} > \theta_\mathrm{CD}$:

| Combined mass $\mathcal{M}_k$ | Conversion allowed |
|---|---|
| $\mathcal{M}_k > \theta_\mathrm{DC}$ | DC: SSA → PDE at rate $\gamma N_k$ |
| $\theta_\mathrm{CD} \leq \mathcal{M}_k \leq \theta_\mathrm{DC}$ | None (hysteresis band) |
| $\mathcal{M}_k < \theta_\mathrm{CD}$ | CD: PDE → SSA at rate $\gamma Y_k$ |

A single symmetric threshold is also supported (`threshold` parameter).

**Probabilistic CD conversion (`cd_removal_mode="probabilistic"`)**

When $Y_k < 1$ (less than one particle's worth of PDE mass): zero the compartment and create a discrete particle with probability $Y_k$. When $Y_k \geq 1$: use boolean removal (remove one particle worth of mass uniformly, always create a particle). This replaces an earlier hard boolean scheme and prevents the PDE from holding genuine sub-particle mass indefinitely.

**`MassProjector`**

Utility class for integrating fine-grid PDE concentrations onto coarse SSA compartments and computing high-concentration masks (used to gate CD events via `sufficient_mask`).

**`ConversionParams`**

Frozen dataclass holding conversion rate $\gamma$ and thresholds. Supports per-species arrays for both rate and thresholds. Provides `DC_mask`, `CD_mask`, and `sufficient_pde_mass_mask` helpers.

**Run modes**

| Method | Returns |
|---|---|
| `run()` | Full time-series for a single realisation |
| `run_repeats()` | Ensemble mean SSA + PDE time-series |
| `run_trajectories()` | All individual trajectories (shape `(repeats, ...)`) |
| `run_repeats_final()` | Final frame only from each repeat (memory-efficient) |

All `run_*` methods support `parallel=True` via `joblib`.

**Pure SSA solver**

Standalone stochastic simulator (`srcm.ssa_solver`) for reference comparisons without any PDE component.

**PDE solver**

Standalone deterministic PDE solver (`srcm.pde_solver`) for pure macroscopic runs.

**CLI**

`srcm-inspect <file.npz>`: prints a summary of saved simulation results.

**`SimulationResults`**

Container for `(time, ssa, pde, domain, species)` with convenience properties for computing combined mass and domain coordinates.
