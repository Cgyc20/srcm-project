import numpy as np
from srcm import PDEModel


def main():
    K = 160
    L = 25.0
    h_pde = L/K
    centre = K // 2

    model = PDEModel(["U"], boundary="zero-flux")
    model.diffusion(U=0.1)

    # No reaction term
    model.pde(lambda U, x: (
        np.zeros_like(U),
    ))

    init_U = np.zeros(K, dtype=float)
    init_U[centre] = 200.0/h_pde

    results, meta = model.run(
        L=L,
        K=K,
        total_time=50.0,
        dt=0.1,
        init={"U": init_U},
    )

    PDEModel.save(
        results=results,
        path="output/diffusion_PDE.npz",
        meta=meta,
    )

    print("Run complete.")
    print("Saved to output/diffusion_PDE.npz")
    print("U shape:", results.U.shape)   # (Nt, K)
    print("x shape:", results.x.shape)   # (K,)
    print("t shape:", results.t.shape)   # (Nt,)


if __name__ == "__main__":
    main()