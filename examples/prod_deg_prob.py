from srcm import SRCMModel
from srcm.results.io import ResultsIO
import numpy as np


def main():
    K = 40

    model = SRCMModel(["U"], boundary="zero-flux")

    model.diffusion(U=0.08)

    model.rates(
        alpha=4.0,   # production
        beta=0.15,   # degradation
    )

    model.reaction({}, {"U": 1}, rate="alpha")
    model.reaction({"U": 1}, {}, rate="beta")

    model.conversion(
        threshold={"U": [10, 20]},
        rate=1.0,
        removal="prob",
    )

    model.pde(lambda U, r: (
        r["alpha"] - r["beta"] * U,
    ))

    init_U = np.zeros(K, dtype=int)
    init_U[K // 2 - 3:K // 2 + 3] = 40

    results, meta = model.run(
        L=30.0,
        K=K,
        pde_multiple=4,
        total_time=60.0,
        dt=0.1,
        init_counts={"U": init_U},
        repeats=100,
        output="mean",
        parallel=True,
        progress=True,
    )

    ResultsIO.save(
        results=results,
        path="output/production_degradation_prob.npz",
        meta=meta,
    )

    print("Saved output/production_degradation_prob.npz")


if __name__ == "__main__":
    main()