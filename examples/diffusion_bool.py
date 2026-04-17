import numpy as np
from srcm import SRCMModel, PDEModel
from srcm.results.io import ResultsIO


def main():
    # ============================================================
    # Shared parameters
    # ============================================================
    K_ssa = 40
    pde_multiple = 4
    K_pde = K_ssa * pde_multiple

    L = 25.0
    total_time = 50.0
    dt = 0.1

    h_ssa = L / K_ssa
    h_pde = L / K_pde
    centre_ssa = K_ssa // 2

    # Fine-grid indices corresponding to the central SSA box
    start = centre_ssa * pde_multiple
    end = start + pde_multiple

    # ============================================================
    # Initial conditions
    # ============================================================
    # SSA initial condition: 200 particles in centre SSA compartment
    init_U_ssa = np.zeros(K_ssa, dtype=int)
    init_U_ssa[centre_ssa] = 200

    # PDE initial condition on fine grid:
    # same total mass, spread uniformly across the whole central SSA box
    init_U_pde = np.zeros(K_pde, dtype=float)
    init_U_pde[start:end] = 200.0 / h_ssa

    print("Initial PDE mass:", np.sum(init_U_pde) * h_pde)

    # ============================================================
    # HYBRID + SSA model
    # ============================================================
    hybrid_model = SRCMModel(["U"], boundary="zero-flux")

    hybrid_model.diffusion(U=0.1)
    hybrid_model.rates()

    hybrid_model.conversion(
        threshold={"U": [5, 10]},
        rate=1.0,
        removal="bool",
    )

    # No PDE reaction term
    hybrid_model.pde(lambda U, r: (
        np.zeros_like(U),
    ))

    # ----------------------------
    # Hybrid
    # ----------------------------
    hybrid_results, hybrid_meta = hybrid_model.run(
        mode="hybrid",
        L=L,
        K=K_ssa,
        pde_multiple=pde_multiple,
        total_time=total_time,
        dt=dt,
        init_counts={"U": init_U_ssa},
        init_pde=None,
        repeats=100,
        output="mean",
        parallel=True,
        progress=True,
    )

    ResultsIO.save(
        results=hybrid_results,
        path="output/diffusion_hybrid_bool.npz",
        meta=hybrid_meta,
    )

    print("Saved output/diffusion_hybrid_bool.npz")
    print("Hybrid SSA shape:", hybrid_results.ssa.shape)
    print("Hybrid PDE shape:", hybrid_results.pde.shape)

    # ----------------------------
    # Pure SSA
    # ----------------------------
    ssa_results, ssa_meta = hybrid_model.run(
        mode="ssa",
        L=L,
        K=K_ssa,
        total_time=total_time,
        dt=dt,
        init_counts={"U": init_U_ssa},
        repeats=100,
        output="mean",
        parallel=True,
        progress=True,
    )

    ResultsIO.save(
        results=ssa_results,
        path="output/diffusion_ssa.npz",
        meta=ssa_meta,
    )

    print("Saved output/diffusion_ssa.npz")
    print("SSA shape:", ssa_results.ssa.shape)

    # ============================================================
    # PURE PDE model
    # ============================================================
    pde_model = PDEModel(["U"], boundary="zero-flux")

    pde_model.diffusion(U=0.1)

    # No reaction term
    pde_model.pde(lambda U, x: (
        np.zeros_like(U),
    ))

    pde_results, pde_meta = pde_model.run(
        L=L,
        K=K_pde,  # IMPORTANT: fine PDE grid, not coarse SSA grid
        total_time=total_time,
        dt=dt,
        init={"U": init_U_pde},
    )

    PDEModel.save(
        results=pde_results,
        path="output/diffusion_pde.npz",
        meta=pde_meta,
    )

    print("Saved output/diffusion_pde.npz")
    print("PDE U shape:", pde_results.U.shape)
    print("PDE x shape:", pde_results.x.shape)
    print("PDE t shape:", pde_results.t.shape)


if __name__ == "__main__":
    main()