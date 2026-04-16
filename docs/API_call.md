Here you go in clean Markdown — ready to drop into a README or notes 👇

---

# Simulation Framework API (Clean Design)

## Imports

```python
from srcm import Simulation, save_npz
```

---

## Core Idea

1. Define a simulation
2. Configure the system (diffusion, rates, reactions)
3. *(Optional)* add hybrid components
4. Run (SSA or Hybrid)
5. Save results

---

## 1. Create Simulation

```python
sim = Simulation(species=["U", "V"], boundary="zero-flux")
```

---

## 2. Define System (Common to SSA + Hybrid)

```python
sim.diffusion(U=Du, V=Dv)

sim.rates(
    r_11=...,
    r_12=...,
    r_2=...,
    r_3=...,
)

sim.reaction({"U": 2}, {"U": 3}, "r_11")
sim.reaction({"U": 2}, {"U": 2, "V": 1}, "r_12")
sim.reaction({"U": 1, "V": 1}, {"V": 1}, "r_2")
sim.reaction({"V": 1}, {}, "r_3")
```

---

## 3. Hybrid-only Additions (Optional)


```pythons
single_threshold = {"U":20, "V":10} #Single threshold
double_threshold = {"U": [20, 30], "V": [10,15]} double threshold
sim.conversion(
    threshold=single_threshold/double_threshold,
    rate=...,
    removal = "bool"/"prob"
)


sim.pde(
    lambda U, V, r: (
        ...,  # dU/dt
        ...,  # dV/dt
    )
)
```

---

## 4A. Run SSA

```python
results, meta = sim.run_ssa(
    L=...,
    K=...,
    total_time=...,
    dt=...,
    init_counts=...,
    repeats=...,
    parallel=...,
)
```

---

## 4B. Run Hybrid

```python
results, meta = sim.run_hybrid(
    L=...,
    K=...,
    pde_multiple=...,
    total_time=...,
    dt=...,
    init_counts=...,
    repeats=...,
    parallel=...,
)
```

---

## 5. Save Results

```python
save_npz(results, "output.npz", meta=meta)
```

---

## Mental Model

* `Simulation` → system definition + configuration
* `diffusion`, `rates`, `reaction` → core system
* `conversion`, `pde` → hybrid extensions
* `run_ssa` vs `run_hybrid` → execution mode
* `save_npz` → output

---

## Key Principle

> Define the system once → choose how to simulate it later.
