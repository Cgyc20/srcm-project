from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from ..hybrid_solver import (
    Domain,
    ConversionParams,
    HybridReactionSystem,
    SimulationResults,
    SRCMEngine,
)

# Later, when your pure SSA backend is in place, import it here, e.g.
# from ..ssa_solver import SSAEngine, SSAReactionSystem

UserRHSFn = Callable[..., Union[Sequence[np.ndarray], np.ndarray]]


@dataclass
class SRCMModel:
    """
    User-facing API for building and running SRCM models.

    Supports:
      - mode="hybrid"
      - mode="ssa"   (backend hook prepared; plug in your pure stochastic solver)

    Public usage
    ------------
    model = SRCMModel(["U", "V"], boundary="zero-flux")

    model.diffusion(U=0.1, V=0.2)
    model.rates(r_11=1.0, r_12=0.5, r_2=2.0, r_3=0.1)

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
        r["r_12"] * U**2 - r["r_2"] * U * V,
        r["r_2"] * U * V - r["r_3"] * V,
    ))

    results, meta = model.run(
        mode="hybrid",
        L=25.0,
        K=100,
        pde_multiple=4,
        total_time=100.0,
        dt=0.1,
        init_counts={"U": U0, "V": V0},
        repeats=100,
        output="mean",
        parallel=True,
    )
    """

    species: List[str]
    boundary: str = "zero-flux"

    _diffusion_rates: Optional[Dict[str, float]] = None
    _reaction_rates: Dict[str, float] = field(default_factory=dict)
    _rhs_user: Optional[UserRHSFn] = None

    # store user reactions neutrally, not as a hybrid-specific object
    _reaction_specs: List[Dict[str, Any]] = field(default_factory=list)

    _conversion_threshold: Any = None
    _conversion_rate: Any = None
    _removal_mode: str = "bool"

    def __post_init__(self):
        if not self.species:
            raise ValueError("species must be non-empty")
        if len(set(self.species)) != len(self.species):
            raise ValueError("species must be unique")
        if self.boundary not in ("zero-flux", "periodic"):
            raise ValueError("boundary must be 'zero-flux' or 'periodic'")

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    def diffusion(self, **rates: float) -> "SRCMModel":
        missing = [sp for sp in self.species if sp not in rates]
        extra = [sp for sp in rates if sp not in self.species]

        if missing:
            raise ValueError(f"Missing diffusion rates for species: {missing}")
        if extra:
            raise ValueError(f"Unknown species in diffusion rates: {extra}")

        self._diffusion_rates = {sp: float(rates[sp]) for sp in self.species}
        return self

    def rates(self, **rates: float) -> "SRCMModel":
        self._reaction_rates.update({str(k): float(v) for k, v in rates.items()})
        return self

    def reaction(
        self,
        reactants: Dict[str, int],
        products: Dict[str, int],
        *,
        rate: str,
        label: Optional[str] = None,
    ) -> "SRCMModel":
        self._reaction_specs.append(
            {
                "reactants": dict(reactants),
                "products": dict(products),
                "rate_name": str(rate),
                "label": label,
            }
        )
        return self

    def conversion(
        self,
        *,
        threshold: Any,
        rate: Any = 1.0,
        removal: str = "bool",
    ) -> "SRCMModel":
        """
        Configure hybrid conversion.

        threshold formats
        -----------------
        Single-threshold:
            threshold={"U": 20, "V": 10}

        Double-threshold / hysteresis:
            threshold={"U": [20, 30], "V": [10, 15]}

        Convention for double threshold:
            [CD_threshold, DC_threshold]

        removal
        -------
        "bool" : standard / sufficient-mass removal rule
        "prob" : probabilistic removal rule
        """
        if removal not in ("bool", "prob"):
            raise ValueError("removal must be 'bool' or 'prob'")

        self._conversion_threshold = threshold
        self._conversion_rate = rate
        self._removal_mode = removal
        return self

    def pde(self, fn: UserRHSFn) -> "SRCMModel":
        self._rhs_user = fn
        return self

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _require_diffusion(self) -> None:
        if self._diffusion_rates is None:
            raise ValueError("diffusion(...) must be set before run()")

    def _require_hybrid_ready(self) -> None:
        self._require_diffusion()
        if self._rhs_user is None:
            raise ValueError("pde(...) must be set before run(mode='hybrid')")
        if self._conversion_threshold is None:
            raise ValueError("conversion(...) must be set before run(mode='hybrid')")

    def _require_ssa_ready(self) -> None:
        self._require_diffusion()
        if len(self._reaction_specs) == 0:
            raise ValueError("At least one reaction(...) must be set before run(mode='ssa')")

    def _normalise_per_species(
        self,
        x: Any,
        *,
        name: str,
    ) -> float | List[float]:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)

        if isinstance(x, dict):
            missing = [sp for sp in self.species if sp not in x]
            extra = [sp for sp in x.keys() if sp not in self.species]
            if missing:
                raise ValueError(f"Missing {name} for species: {missing}")
            if extra:
                raise ValueError(f"Unknown species in {name}: {extra}")
            return [float(x[sp]) for sp in self.species]

        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) != len(self.species):
                raise ValueError(f"{name} must have length {len(self.species)}")
            return [float(v) for v in x]

        raise TypeError(f"Unsupported type for {name}: {type(x)}")

    def _build_conversion(self) -> ConversionParams:
        thr = self._conversion_threshold
        rate = self._conversion_rate

        if isinstance(thr, dict):
            missing = [sp for sp in self.species if sp not in thr]
            extra = [sp for sp in thr.keys() if sp not in self.species]
            if missing:
                raise ValueError(f"Missing threshold for species: {missing}")
            if extra:
                raise ValueError(f"Unknown species in threshold: {extra}")

            first_val = thr[self.species[0]]
            if isinstance(first_val, (list, tuple, np.ndarray)):
                cd = []
                dc = []
                for sp in self.species:
                    pair = thr[sp]
                    if len(pair) != 2:
                        raise ValueError(
                            f"threshold[{sp}] must have length 2: [CD_threshold, DC_threshold]"
                        )
                    cd.append(float(pair[0]))
                    dc.append(float(pair[1]))
            else:
                vals = [float(thr[sp]) for sp in self.species]
                cd = vals
                dc = vals

        elif isinstance(thr, (int, float, np.integer, np.floating)):
            cd = float(thr)
            dc = float(thr)

        elif isinstance(thr, (list, tuple, np.ndarray)):
            if len(thr) == 2 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in thr):
                cd = float(thr[0])
                dc = float(thr[1])
            elif len(thr) == len(self.species):
                vals = [float(v) for v in thr]
                cd = vals
                dc = vals
            else:
                raise ValueError(
                    "threshold list/tuple must be either length 2 ([CD, DC]) "
                    f"or length {len(self.species)}"
                )
        else:
            raise TypeError(f"Unsupported threshold type: {type(thr)}")

        rate_val = self._normalise_per_species(rate, name="rate")

        return ConversionParams(
            DC_threshold=dc,
            CD_threshold=cd,
            rate=rate_val,
        )

    def _build_domain(
        self,
        *,
        L: float,
        K: int,
        pde_multiple: int,
    ) -> Domain:
        return Domain(
            length=float(L),
            n_ssa=int(K),
            pde_multiple=int(pde_multiple),
            boundary=str(self.boundary),
        )

    def _build_pde_terms(self) -> Callable[[np.ndarray, Dict[str, float]], np.ndarray]:
        if self._rhs_user is None:
            raise RuntimeError("pde(...) must be set before build")

        n = len(self.species)
        user_fn = self._rhs_user

        def pde_terms(C: np.ndarray, rates_: Dict[str, float]) -> np.ndarray:
            args = [C[i] for i in range(n)]
            out = user_fn(*args, rates_)  # type: ignore[misc]

            if isinstance(out, np.ndarray):
                arr = out.astype(float, copy=False)
                if arr.shape != C.shape:
                    raise ValueError(
                        f"pde(...) returned array with shape {arr.shape}, expected {C.shape}"
                    )
                return arr

            out = tuple(out)
            if len(out) != n:
                raise ValueError(f"pde(...) returned {len(out)} outputs, expected {n}")

            arr = np.array(out, dtype=float)
            if arr.shape != C.shape:
                raise ValueError(
                    f"pde(...) returned array with shape {arr.shape}, expected {C.shape}"
                )
            return arr

        return pde_terms

    def _build_hybrid_reaction_system(self) -> HybridReactionSystem:
        rs = HybridReactionSystem(species=list(self.species))

        for spec in self._reaction_specs:
            rs.add_reaction_original(
                reactants=spec["reactants"],
                products=spec["products"],
                rate=0.0,
                rate_name=spec["rate_name"],
            )

        for rec in getattr(rs, "pure_reactions", []):
            rn = rec.get("rate_name", None)
            if rn is not None and rn in self._reaction_rates:
                rec["rate"] = float(self._reaction_rates[rn])

        return rs

    def _build_hybrid_engine(
        self,
        *,
        L: float,
        K: int,
        pde_multiple: int,
    ) -> SRCMEngine:
        self._require_hybrid_ready()

        domain = self._build_domain(L=L, K=K, pde_multiple=pde_multiple)
        conversion = self._build_conversion()
        pde_terms = self._build_pde_terms()
        reactions = self._build_hybrid_reaction_system()

        return SRCMEngine(
            reactions=reactions,
            pde_reaction_terms=pde_terms,
            diffusion_rates=self._diffusion_rates,  # type: ignore[arg-type]
            domain=domain,
            conversion=conversion,
            reaction_rates=dict(self._reaction_rates),
            cd_removal_mode="probabilistic" if self._removal_mode == "prob" else "standard",
        )

    def _build_ssa_backend(
        self,
        *,
        L: float,
        K: int,
    ):
        """
        Hook for your pure stochastic backend.

        Replace this with construction of your SSA reaction system + engine.
        """
        self._require_ssa_ready()

        domain = Domain(
            length=float(L),
            n_ssa=int(K),
            pde_multiple=1,
            boundary=str(self.boundary),
        )

        raise NotImplementedError(
            "SSA backend not yet wired in. Next step: build Reaction/SSA engine "
            "from self._reaction_specs and return an object matching the run API."
        )

    def _coerce_init_counts(
        self,
        init_counts: Union[np.ndarray, Dict[str, np.ndarray]],
        *,
        K: int,
    ) -> np.ndarray:
        n = len(self.species)

        if isinstance(init_counts, np.ndarray):
            arr = init_counts.astype(int, copy=False)
            if arr.shape != (n, K):
                raise ValueError(f"init_counts must have shape {(n, K)}, got {arr.shape}")
            return arr

        if isinstance(init_counts, dict):
            missing = [sp for sp in self.species if sp not in init_counts]
            extra = [sp for sp in init_counts.keys() if sp not in self.species]
            if missing:
                raise ValueError(f"Missing init_counts for species: {missing}")
            if extra:
                raise ValueError(f"Unknown species in init_counts: {extra}")

            arr = np.zeros((n, K), dtype=int)
            for i, sp in enumerate(self.species):
                vals = np.asarray(init_counts[sp], dtype=int)
                if vals.shape != (K,):
                    raise ValueError(
                        f"init_counts['{sp}'] must have shape {(K,)}, got {vals.shape}"
                    )
                arr[i] = vals
            return arr

        raise TypeError("init_counts must be a numpy array or dict[str, array]")

    def _coerce_init_pde(
        self,
        init_pde: Optional[Union[np.ndarray, Dict[str, np.ndarray]]],
        *,
        n_pde: int,
    ) -> np.ndarray:
        n = len(self.species)

        if init_pde is None:
            return np.zeros((n, n_pde), dtype=float)

        if isinstance(init_pde, np.ndarray):
            arr = init_pde.astype(float, copy=False)
            if arr.shape != (n, n_pde):
                raise ValueError(f"init_pde must have shape {(n, n_pde)}, got {arr.shape}")
            return arr

        if isinstance(init_pde, dict):
            missing = [sp for sp in self.species if sp not in init_pde]
            extra = [sp for sp in init_pde.keys() if sp not in self.species]
            if missing:
                raise ValueError(f"Missing init_pde for species: {missing}")
            if extra:
                raise ValueError(f"Unknown species in init_pde: {extra}")

            arr = np.zeros((n, n_pde), dtype=float)
            for i, sp in enumerate(self.species):
                vals = np.asarray(init_pde[sp], dtype=float)
                if vals.shape != (n_pde,):
                    raise ValueError(
                        f"init_pde['{sp}'] must have shape {(n_pde,)}, got {vals.shape}"
                    )
                arr[i] = vals
            return arr

        raise TypeError("init_pde must be None, a numpy array, or dict[str, array]")

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        mode: str = "hybrid",
        L: float,
        K: int,
        pde_multiple: int = 4,
        total_time: float,
        dt: float,
        init_counts: Union[np.ndarray, Dict[str, np.ndarray]],
        init_pde: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        repeats: int = 1,
        output: str = "single",
        parallel: bool = False,
        n_jobs: int = -1,
        prefer: str = "processes",
        progress: bool = True,
        seed: int = 0,
    ):
        """
        Run the model.

        Parameters
        ----------
        mode : {"hybrid", "ssa"}
        output : {"single", "mean", "trajectories", "final"}

        Notes
        -----
        - For mode="hybrid", conversion(...) and pde(...) must be configured.
        - For mode="ssa", those are ignored, and the pure stochastic backend is used.
        """
        if mode not in ("hybrid", "ssa"):
            raise ValueError("mode must be one of: 'hybrid', 'ssa'")
        if output not in ("single", "mean", "trajectories", "final"):
            raise ValueError("output must be one of: 'single', 'mean', 'trajectories', 'final'")

        if mode == "ssa":
            # prepare shape-checked counts
            ssa0 = self._coerce_init_counts(init_counts, K=int(K))

            # pure SSA backend hook
            backend = self._build_ssa_backend(L=L, K=K)

            # once wired in, backend should mirror the hybrid backend interface
            # keeping this explicit for now
            raise NotImplementedError(
                "SSA run path is prepared, but backend calls still need wiring."
            )

        # hybrid mode
        engine = self._build_hybrid_engine(L=L, K=K, pde_multiple=pde_multiple)
        domain = engine.domain

        ssa0 = self._coerce_init_counts(init_counts, K=domain.K)
        pde0 = self._coerce_init_pde(init_pde, n_pde=domain.n_pde)

        if output == "single":
            res = engine.run(
                initial_ssa=ssa0,
                initial_pde=pde0,
                time=float(total_time),
                dt=float(dt),
                seed=int(seed),
            )
            return res, self.metadata(mode=mode, domain=domain, output=output, repeats=1)

        if output == "mean":
            if repeats <= 0:
                raise ValueError("repeats must be > 0 for output='mean'")
            res = engine.run_repeats(
                initial_ssa=ssa0,
                initial_pde=pde0,
                time=float(total_time),
                dt=float(dt),
                repeats=int(repeats),
                seed=int(seed),
                parallel=bool(parallel),
                n_jobs=int(n_jobs),
                prefer=str(prefer),
                progress=bool(progress),
            )
            return res, self.metadata(mode=mode, domain=domain, output=output, repeats=int(repeats))

        if output == "trajectories":
            if repeats <= 0:
                raise ValueError("repeats must be > 0 for output='trajectories'")
            res = engine.run_trajectories(
                initial_ssa=ssa0,
                initial_pde=pde0,
                time=float(total_time),
                dt=float(dt),
                repeats=int(repeats),
                seed=int(seed),
                parallel=bool(parallel),
                n_jobs=int(n_jobs),
                prefer=str(prefer),
                progress=bool(progress),
            )
            return res, self.metadata(mode=mode, domain=domain, output=output, repeats=int(repeats))

        if repeats <= 0:
            raise ValueError("repeats must be > 0 for output='final'")
        final_ssa, final_pde, t_final = engine.run_repeats_final(
            initial_ssa=ssa0,
            initial_pde=pde0,
            time=float(total_time),
            dt=float(dt),
            repeats=int(repeats),
            seed=int(seed),
            parallel=bool(parallel),
            n_jobs=int(n_jobs),
            prefer=str(prefer),
            progress=bool(progress),
            save_path=None,
        )
        return (
            final_ssa,
            final_pde,
            t_final,
        ), self.metadata(mode=mode, domain=domain, output=output, repeats=int(repeats))

    # ------------------------------------------------------------------
    # metadata / inspection
    # ------------------------------------------------------------------
    def metadata(
        self,
        *,
        mode: Optional[str] = None,
        domain: Optional[Domain] = None,
        output: Optional[str] = None,
        repeats: Optional[int] = None,
    ) -> dict:
        out = {
            "model": "SRCMModel",
            "mode": mode,
            "species": list(self.species),
            "boundary": self.boundary,
            "diffusion_rates": None if self._diffusion_rates is None else dict(self._diffusion_rates),
            "reaction_rates": dict(self._reaction_rates),
            "removal": self._removal_mode,
            "output": output,
            "repeats": repeats,
            "reactions": list(self._reaction_specs),
        }

        if domain is not None:
            out["domain"] = {
                "length": float(domain.length),
                "n_ssa": int(domain.K),
                "pde_multiple": int(domain.pde_multiple),
                "boundary": str(domain.boundary),
            }

        if self._conversion_threshold is not None:
            out["conversion_threshold"] = self._conversion_threshold
        if self._conversion_rate is not None:
            out["conversion_rate"] = self._conversion_rate

        return out

    def hybrid_labels(self) -> List[str]:
        rs = self._build_hybrid_reaction_system()
        return [hr.label for hr in rs.hybrid_reactions]

    def describe_reactions(self) -> None:
        rs = self._build_hybrid_reaction_system()
        if hasattr(rs, "describe_full"):
            rs.describe_full()
        else:
            rs.describe()