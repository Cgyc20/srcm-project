# SRCM — Spatial Regime Conversion Method

A Python package for simulating spatial reaction–diffusion systems using the Spatial Regime Conversion Method (SRCM): a hybrid solver that couples stochastic particle dynamics (SSA) and deterministic continuum dynamics (PDE) within a single adaptive spatial framework.

**Author:** Charlie Cameron  
**Reference:** Cameron, C. G., Smith, C. A., & Yates, C. A. (2025). *The Spatial Regime Conversion Method.* Mathematics, 13(21), 3406. [https://doi.org/10.3390/math13213406](https://doi.org/10.3390/math13213406)

---

![Schematic](figures/schematic.png)

---

## Overview

Many biological and physical systems exhibit spatially varying stochasticity. Near a wave front, or where particle numbers are low, stochastic fluctuations are essential to the dynamics. Deep in the bulk, where particle numbers are high, a deterministic PDE is an accurate and far cheaper description.

The SRCM maintains both representations simultaneously on the same domain and converts mass between them based on local density. The result is a solver that captures stochastic front dynamics (e.g. the Brunet–Derrida wavespeed correction in FKPP) while remaining computationally efficient in the high-density bulk.

---

## Methodology

### Dual grid

The domain $[0, L]$ is discretised on two grids:

- **Coarse SSA grid**: $K$ compartments of width $h = L/K$. Particle counts $N_k^{(s)} \in \mathbb{Z}_{\geq 0}$ per species $s$.
- **Fine PDE grid**: $KP$ cells of width $dx = h/P$. Concentrations $u_j^{(s)} \geq 0$. The integer $P$ is `pde_multiple`.

Both grids are live simultaneously. The **combined mass** in compartment $k$ for species $s$ is:

$$\mathcal{M}_k^{(s)} = N_k^{(s)} + Y_k^{(s)}, \qquad Y_k^{(s)} = \sum_{j \in k} u_j^{(s)}\,dx$$

### Regime conversion

Conversion between discrete and continuous representations is controlled by two thresholds $\theta_\mathrm{CD} < \theta_\mathrm{DC}$ (hysteresis band prevents chattering):

| Condition | Event | Rate |
|---|---|---|
| $\mathcal{M}_k > \theta_\mathrm{DC}$ | DC: one SSA particle → PDE mass | $\gamma N_k$ |
| $\mathcal{M}_k < \theta_\mathrm{CD}$ | CD: PDE mass → one SSA particle | $\gamma Y_k$ |
| $\theta_\mathrm{CD} \leq \mathcal{M}_k \leq \theta_\mathrm{DC}$ | None | — |

$\gamma$ is the conversion rate. Both thresholds can be set per-species.

### Within a timestep $\Delta t$

Each timestep follows three stages:

1. **Gillespie interval** $[t, t+\Delta t)$: SSA diffusion, conversion (CD/DC), and hybrid reactions are simulated with the PDE held frozen. The propensity vector has the block structure:

$$\mathbf{a} = \bigl[\underbrace{a_\mathrm{diff}}_{\text{SSA diffusion}} \,\big|\, \underbrace{a_\mathrm{CD}}_{\text{PDE}\to\text{SSA}} \,\big|\, \underbrace{a_\mathrm{DC}}_{\text{SSA}\to\text{PDE}} \,\big|\, \underbrace{a_\mathrm{hybrid}}_{\text{mixed reactions}}\bigr]$$

2. **RK4 step**: the PDE advances from $t$ to $t+\Delta t$ via fourth-order Runge–Kutta. Diffusion and macroscopic reactions are integrated together.

3. **Post-PDE front cleanup** *(v0.0.3+)*: enforces the Brunet–Derrida $N$-cutoff (see below).

### CD conversion: probabilistic removal

When a CD event fires for compartment $k$, the PDE slice is modified cell by cell:

$$u_j \leftarrow u_j - \min\!\left(\frac{1}{h},\, u_j\right)$$

The total mass removed $M = \sum_j \min(1/h, u_j)\,dx \in [0,1]$. A discrete particle is created with probability $M$. This guarantees the PDE never goes negative and removes at most one particle per event.

### Front cleanup and the Brunet–Derrida cutoff

For pushed-wave systems such as FKPP, the stochastic wavespeed is slower than the deterministic PDE speed by a logarithmic correction (Brunet & Derrida 1997). This correction requires that concentrations below $1/N$ are effectively zero — the discreteness of particles provides the cutoff.

In a hybrid solver, the PDE propagates mass every RK4 step regardless of what happens in the Gillespie loop. Any sub-particle PDE mass that diffuses into the stochastic zone between Gillespie events will drive deterministic (PDE-speed) propagation — causing the measured wavespeed to drift toward $c^* = 2\sqrt{D\lambda}$ when $\gamma$ is small, or toward a similarly wrong speed when $\gamma$ is large (because rapid CD/DC cycling collapses the SSA zone).

**The fix (v0.0.3):** after every RK4 step, scan the stochastic zone ($\mathcal{M}_k < \theta_\mathrm{CD}$). For any compartment with $Y_k \in (0,1)$:

$$u_j \leftarrow 0 \;\forall j \in k, \qquad N_k \mathrel{+}= \mathbf{1}[\xi < Y_k], \quad \xi \sim \mathrm{Uniform}(0,1)$$

This absorbs leaked tip mass within one $\Delta t$, independently of $\gamma$. Expected mass is conserved. The Gillespie CD events handle $Y_k \geq 1$; the cleanup handles $Y_k < 1$.

---

## Installation

### From GitHub

```bash
pip install "git+https://github.com/Cgyc20/srcm-project.git"
```

### Pinned to a specific version tag

```bash
pip install "git+https://github.com/Cgyc20/srcm-project.git@v0.0.3"
```

With `uv`:

```bash
uv add "git+https://github.com/Cgyc20/srcm-project.git" --tag v0.0.3
```

### Editable install (development / local path)

```bash
pip install -e /path/to/srcm-project
```

With `uv` (in `pyproject.toml`):

```toml
[tool.uv.sources]
srcm = { path = "../srcm-project", editable = true }
```

An editable install reads directly from the source files on disk — any changes take effect immediately without reinstalling.

---

## Usage

### 1. Define the domain and reactions

```python
import numpy as np
from srcm import Simulation

sim = Simulation(species=["U"])

sim.define_rates(birth=10.0, competition=0.01)
sim.define_diffusion(U=0.01)

# SSA: birth and competition reactions
sim.add_reaction(reactants={}, products={"U": 1}, rate="birth")
sim.add_reaction(reactants={"U": 2}, products={}, rate="competition")
```

### 2. Define PDE reaction terms

The PDE reaction term is a function of concentration arrays and a rates dict. It should return $dC/dt$ from reactions only (diffusion is handled internally).

```python
def fkpp_reactions(C, rates):
    U = C[0]
    dU = rates["birth"] * U - rates["competition"] * U**2
    return np.array([dU])

sim.set_pde_reactions(fkpp_reactions)
```

### 3. Define conversion parameters

```python
sim.define_conversion(
    DC_threshold=70,   # above this: SSA particles convert to PDE
    CD_threshold=60,   # below this: PDE mass converts to SSA particles
    rate=10.0,         # gamma: conversion rate
)
```

For a single symmetric threshold use `threshold=N` instead. Both accept per-species arrays.

### 4. Run

```python
results = sim.run(
    L=10.0,            # domain length
    K=100,             # number of SSA compartments
    pde_multiple=8,    # PDE cells per compartment
    total_time=10.0,
    dt=0.005,
    init_counts={"U": 10},   # initial SSA particles
    repeats=200,
    mode="mean",              # "single" | "mean" | "trajectories" | "final"
    cd_removal_mode="probabilistic",
    enable_front_cleanup=True,
)
```

### 5. Save and inspect

```python
from srcm.results import save_npz
save_npz(results, "fkpp_run.npz")
```

```bash
srcm-inspect fkpp_run.npz
```

---

## Engine parameters

Direct access to the engine gives full control:

```python
from srcm.hybrid_solver.engine import SRCMEngine
from srcm.hybrid_solver.conversion import ConversionParams
from srcm.hybrid_solver.domain import Domain

domain = Domain(length=10.0, n_ssa=100, pde_multiple=8)
conversion = ConversionParams(DC_threshold=70, CD_threshold=60, rate=10.0)

engine = SRCMEngine(
    reactions=reaction_system,
    pde_reaction_terms=fkpp_reactions,
    diffusion_rates={"U": 0.01},
    domain=domain,
    conversion=conversion,
    reaction_rates={"birth": 10.0, "competition": 0.01},
    cd_removal_mode="probabilistic",   # "standard" | "probabilistic"
    enable_front_cleanup=True,          # v0.0.3+: enforce N-cutoff after each RK4 step
)

results = engine.run(
    initial_ssa=initial_ssa,
    initial_pde=initial_pde,
    time=10.0,
    dt=0.005,
    seed=42,
)
```

### Key `SRCMEngine` fields

| Field | Description |
|---|---|
| `cd_removal_mode` | `"standard"`: remove one particle mass uniformly. `"probabilistic"`: per-cell $\min(1/h, u_j)$ removal with probabilistic particle creation. |
| `enable_front_cleanup` | If `True` (default), zero sub-particle PDE mass in the stochastic zone after each RK4 step. Makes wavespeed independent of $\gamma$ for pushed-wave systems. |

---

## Solvers

The package provides three standalone solvers:

| Module | Description |
|---|---|
| `srcm.hybrid_solver` | Full SRCM hybrid SSA–PDE engine |
| `srcm.ssa_solver` | Pure stochastic (SSA/RDME) simulation, no PDE |
| `srcm.pde_solver` | Pure deterministic PDE solver |

Use the pure SSA for reference benchmarks (wavespeed, moments) and the PDE for the deterministic limit.

---

## Limitations

- One-dimensional spatial domains only.
- Reactions limited to order ≤ 2 in the SSA.
- Explicit (RK4) PDE time stepping: timestep $\Delta t$ must satisfy the diffusion CFL condition $\Delta t \leq dx^2 / (2D)$.
- The post-PDE cleanup (v0.0.3) is designed for pushed waves where a single stochastic cutoff determines the speed. For systems without a sharp stochastic front it can be disabled with `enable_front_cleanup=False`.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a full version history.

---

## Reference

> Cameron, C. G., Smith, C. A., & Yates, C. A. (2025).
> *The Spatial Regime Conversion Method.* Mathematics, 13(21), 3406.
> [https://doi.org/10.3390/math13213406](https://doi.org/10.3390/math13213406)
