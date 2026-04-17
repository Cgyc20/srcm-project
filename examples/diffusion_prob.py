from srcm import SRCMModel
from srcm.results.io import ResultsIO
import numpy as np


def main():
    K = 40
    centre = K // 2

    model = SRCMModel(["U"], boundary="zero-flux")

    model.diffusion(U=0.1)

    # No reactions
    model.rates()
    model.conversion(
        threshold={"U": [5, 10]},
        rate=1.0,
        removal="prob",
    )

    # No reaction terms in PDE
    model.pde(lambda U, r: (
        np.zeros_like(U),
    ))

    # All particles initially in the central compartment
    init_U = np.zeros(K, dtype=int)
    init_U[centre] = 200

    results, meta = model.run(
        L=25.0,
        K=K,
        pde_multiple=4,
        total_time=50.0,
        dt=0.1,
        init_counts={"U": init_U},
        repeats=100,
        output="mean",
        parallel=True,
        progress=True
    )

    ResultsIO.save(
        results=results,
        path="output/diffusion_one_species_central_peak.npz",
        meta=meta,
    )

    print("Run complete.")
    print("Saved to output/diffusion_one_species_central_peak.npz")
    print("SSA shape:", results.ssa.shape)
    print("PDE shape:", results.pde.shape)


if __name__ == "__main__":
    main()