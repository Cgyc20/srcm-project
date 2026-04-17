from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from ..hybrid_solver import Domain, ConversionParams, HybridReactionSystem, SimulationResults, SRCMEngine


UserRHSFn = Callable[..., Union[Sequence[np.ndarray], np.ndarray]]


@dataclass
class SRCMModel:
    """
    User-facing API for building and running SRCM hybrid models.

    Typical usage
    -------------
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
    _reactions: Optional[HybridReactionSystem] = None

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

        self._reactions = HybridReactionSystem(species=list(self.species))

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
        if self._reactions is None:
            raise RuntimeError("Internal reaction system not initialised")

        # add_reaction_original stores symbolic rate_name and can later be filled numerically
        self._reactions.add_reaction_original(
            reactants=reactants,
            products=products,
            rate=0.0,
            rate_name=str(rate),
        )

        # If your reaction system supports labels directly, wire that in there instead.
        # For now, label is accepted for API consistency.
        _ = label
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
    def _require_ready(self) -> None:
        if self._diffusion_rates is None:
            raise ValueError("diffusion(...) must be set before run()")
        if self._rhs_user is None:
            raise ValueError("pde(...) must be set before run()")
        if self._conversion_threshold is None:
            raise ValueError("conversion(...) must be set before run()")
        if self._reactions is None:
            raise RuntimeError("Internal reaction system not initialised")

    def _normalise_per_species(
        self,
        x: Any,
        *,
        name: str,
        allow_pairs: bool = False,
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

            if allow_pairs and all(isinstance(x[sp], (list, tuple, np.ndarray)) for sp in self.species):
                return x  # handled elsewhere
            return [float(x[sp]) for sp in self.species]

        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) != len(self.species):
                raise ValueError(f"{name} must have length {len(self.species)}")
            return [float(v) for v in x]

        raise TypeError(f"Unsupported type for {name}: {type(x)}")

    def _build_conversion(self) -> ConversionParams:
        thr = self._conversion_threshold
        rate = self._conversion_rate

        # Single threshold per species -> use same value for both CD and DC
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
                # global [CD, DC]
                cd = float(thr[0])
                dc = float(thr[1])
            elif len(thr) == len(self.species):
                # single threshold per species
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
                raise ValueError(
                    f"pde(...) returned {len(out)} outputs, expected {n}"
                )

            arr = np.array(out, dtype=float)
            if arr.shape != C.shape:
                raise ValueError(
                    f"pde(...) returned array with shape {arr.shape}, expected {C.shape}"
                )
            return arr

        return pde_terms

    def _update_reaction_rates_on_system(self) -> None:
        if self._reactions is None:
            raise RuntimeError("Internal reaction system not initialised")

        for rec in getattr(self._reactions, "pure_reactions", []):
            rn = rec.get("rate_name", None)
            if rn is not None and rn in self._reaction_rates:
                rec["rate"] = float(self._reaction_rates[rn])

    def _build_engine(
        self,
        *,
        L: float,
        K: int,
        pde_multiple: int,
    ) -> SRCMEngine:
        self._require_ready()
        self._update_reaction_rates_on_system()

        domain = self._build_domain(L=L, K=K, pde_multiple=pde_multiple)
        conversion = self._build_conversion()
        pde_terms = self._build_pde_terms()

        return SRCMEngine(
            reactions=self._reactions,  # type: ignore[arg-type]
            pde_reaction_terms=pde_terms,
            diffusion_rates=self._diffusion_rates,  # type: ignore[arg-type]
            domain=domain,
            conversion=conversion,
            reaction_rates=dict(self._reaction_rates),
            cd_removal_mode="probabilistic" if self._removal_mode == "prob" else "standard",
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
        output : {"single", "mean", "trajectories", "final"}
            - "single"       -> one SimulationResults
            - "mean"         -> averaged SimulationResults over repeats
            - "trajectories" -> trajectory ensemble SimulationResults
            - "final"        -> (final_ssa, final_pde, t_final)
        """
        if output not in ("single", "mean", "trajectories", "final"):
            raise ValueError("output must be one of: 'single', 'mean', 'trajectories', 'final'")

        engine = self._build_engine(L=L, K=K, pde_multiple=pde_multiple)
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
            return res, self.metadata(domain=domain, output=output, repeats=1)

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
            return res, self.metadata(domain=domain, output=output, repeats=int(repeats))

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
            return res, self.metadata(domain=domain, output=output, repeats=int(repeats))

        # output == "final"
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
        ), self.metadata(domain=domain, output=output, repeats=int(repeats))

    # ------------------------------------------------------------------
    # metadata / inspection
    # ------------------------------------------------------------------
    def metadata(
        self,
        *,
        domain: Optional[Domain] = None,
        output: Optional[str] = None,
        repeats: Optional[int] = None,
    ) -> dict:
        out = {
            "model": "SRCMModel",
            "species": list(self.species),
            "boundary": self.boundary,
            "diffusion_rates": None if self._diffusion_rates is None else dict(self._diffusion_rates),
            "reaction_rates": dict(self._reaction_rates),
            "removal": self._removal_mode,
            "output": output,
            "repeats": repeats,
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
        if self._reactions is None:
            raise RuntimeError("Internal reaction system not initialised")
        return [hr.label for hr in self._reactions.hybrid_reactions]

    def describe_reactions(self) -> None:
        if self._reactions is None:
            raise RuntimeError("Internal reaction system not initialised")
        if hasattr(self._reactions, "describe_full"):
            self._reactions.describe_full()
        else:
            self._reactions.describe()