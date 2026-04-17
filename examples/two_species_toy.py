from srcm import SRCMModel
from srcm.results.io import ResultsIO
import numpy as np


def main():
    K = 40

    model = SRCMModel(["U", "V"], boundary="zero-flux")

    model.diffusion(U=0.02, V=0.25)

    model.rates(
        a=1.0,
        b=0.04,
        c=0.03,
        d=0.02,
    )

    # simple toy reactions
    model.reaction({}, {"U": 1}, rate="a")
    model.reaction({"U": 1}, {"U": 2}, rate="b")
    model.reaction({"U": 1, "V": 1}, {"V": 1}, rate="c")
    model.reaction({"V": 1}, {}, rate="d")

    # hybrid-only configuration
    model.conversion(
        threshold={"U": [10, 20], "V": [10, 20]},
        rate=1.0,
        removal="prob",
    )

    model.pde(lambda U, V, r: (
        r["a"] + r["b"] * U - r["c"] * U * V,
        r["c"] * U * V - r["d"] * V,
    ))

    init_U = np.full(K, 5, dtype=int)
    init_V = np.full(K, 3, dtype=int)

    # small central perturbation
    init_U[K // 2 - 2:K // 2 + 2] += 20
    init_V[K // 2 - 2:K // 2 + 2] += 10

    init_counts = {
        "U": init_U,
        "V": init_V,
    }

    # ------------------------------------------------------------
    # Hybrid run
    # ------------------------------------------------------------
    hybrid_results, hybrid_meta = model.run(
        mode="hybrid",
        L=40.0,
        K=K,
        pde_multiple=4,
        total_time=80.0,
        dt=0.1,
        init_counts=init_counts,
        repeats=50,
        output="mean",
        parallel=True,
        progress=True,
    )

    ResultsIO.save(
        results=hybrid_results,
        path="output/two_species_toy_hybrid_prob.npz",
        meta=hybrid_meta,
    )

    print("Saved output/two_species_toy_hybrid_prob.npz")

    # ------------------------------------------------------------
    # Pure SSA run
    # ------------------------------------------------------------
    ssa_results, ssa_meta = model.run(
        mode="ssa",
        L=40.0,
        K=K,
        total_time=80.0,
        dt=0.1,
        init_counts=init_counts,
        repeats=50,
        output="mean",
        parallel=True,
        progress=True,
    )

    ResultsIO.save(
        results=ssa_results,
        path="output/two_species_toy_ssa.npz",
        meta=ssa_meta,
    )

    print("Saved output/two_species_toy_ssa.npz")


if __name__ == "__main__":
    main()