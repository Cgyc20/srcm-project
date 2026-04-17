from ..src.srcm import SRCMModel
import numpy as np


model = SRCMModel(["U", "V"], boundary="zero-flux")

model.diffusion(U=0.1, V=0.2)

model.rates(
    r_11=1.0,
    r_12=0.8,
    r_2=0.4,
    r_3=0.1,
)

model.reaction({"U": 2}, {"U": 3}, rate="r_11")
model.reaction({"U": 2}, {"U": 2, "V": 1}, rate="r_12")
model.reaction({"U": 1, "V": 1}, {"V": 1}, rate="r_2")
model.reaction({"V": 1}, {}, rate="r_3")

model.conversion(
    threshold={"U": [20, 30], "V": [10, 15]},
    rate=1.0,
    removal="prob",
)

model.pde(lambda U, V, r: (
    r["r_11"] * U**2 - r["r_2"] * U * V,
    r["r_12"] * U**2 + r["r_2"] * U * V - r["r_3"] * V,
))

results, meta = model.run(
    L=25.0,
    K=100,
    pde_multiple=4,
    total_time=50.0,
    dt=0.1,
    init_counts={
        "U": np.zeros(100, dtype=int),
        "V": np.zeros(100, dtype=int),
    },
    repeats=20,
    output="mean",
    parallel=True,
)