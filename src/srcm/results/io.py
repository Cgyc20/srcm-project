from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import json

import numpy as np

from srcm import Domain
from srcm import SimulationResults

PathLike = Union[str, Path]


def _domain_to_metadata(domain: Domain) -> Dict[str, Any]:
    return {
        "length": float(domain.length),
        "n_ssa": int(domain.K),
        "pde_multiple": int(domain.pde_multiple),
        "boundary": str(domain.boundary),
    }


def _domain_from_metadata(meta: Dict[str, Any]) -> Domain:
    dom = meta["domain"]
    return Domain(
        length=float(dom["length"]),
        n_ssa=int(dom["n_ssa"]),
        pde_multiple=int(dom["pde_multiple"]),
        boundary=str(dom["boundary"]),
    )


def _json_safe(value: Any) -> Any:
    """
    Convert common non-JSON-native values to safe equivalents.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    return value


def _infer_result_kind(ssa: np.ndarray, pde: Optional[np.ndarray]) -> str:
    """
    Infer result kind from array ranks.

    Supported:
      ssa.ndim == 3 -> single_or_mean_timeseries
          expected SSA shape: (S, K, T)
          expected PDE shape: (S, Npde, T)

      ssa.ndim == 4 -> trajectories
          expected SSA shape: (R, S, K, T)
          expected PDE shape: (R, S, Npde, T)

      ssa.ndim == 3 and time scalar/len1 won't matter here; still timeseries

      final-only is typically stored as:
          SSA shape: (R, S, K)
          PDE shape: (R, S, Npde)
    """
    if ssa.ndim == 4:
        return "trajectories"
    if ssa.ndim == 3:
        return "timeseries"
    raise ValueError(f"Unsupported SSA shape {ssa.shape}")


def _ensure_pde_for_timeseries(res: SimulationResults) -> np.ndarray:
    pde = getattr(res, "pde", None)
    if pde is not None:
        return pde

    ssa = res.ssa
    time = res.time
    n_species = int(ssa.shape[0])
    n_steps = int(time.shape[0])
    n_pde = int(res.domain.n_pde)
    return np.zeros((n_species, n_pde, n_steps), dtype=float)


def _ensure_pde_for_trajectories(res: SimulationResults) -> np.ndarray:
    pde = getattr(res, "pde", None)
    if pde is not None:
        return pde

    R, S, _, T = res.ssa.shape
    n_pde = int(res.domain.n_pde)
    return np.zeros((R, S, n_pde, T), dtype=float)


class ResultsIO:
    """
    Unified save/load wrapper for SRCM results.

    Supports:
      - single timeseries results
      - mean timeseries results
      - trajectory ensembles
      - final-only arrays

    File layout:
      <path>.npz  : arrays + embedded meta_json
    """

    @staticmethod
    def save(
        results: Optional[SimulationResults] = None,
        path: PathLike = "results.npz",
        *,
        final_ssa: Optional[np.ndarray] = None,
        final_pde: Optional[np.ndarray] = None,
        t_final: Optional[float] = None,
        domain: Optional[Domain] = None,
        species: Optional[list[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        result_kind: Optional[str] = None,
    ) -> None:
        """
        Save either:
          1. a SimulationResults object
          2. final-only arrays from run_repeats_final

        Examples
        --------
        ResultsIO.save(results=res, path="out/run1.npz")

        ResultsIO.save(
            path="out/final_only.npz",
            final_ssa=final_ssa,
            final_pde=final_pde,
            t_final=t_final,
            domain=engine.domain,
            species=engine.reactions.species,
            meta={"run_type": "final_only"},
        )
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta_out: Dict[str, Any] = dict(meta or {})

        if results is not None:
            if domain is None:
                domain = results.domain
            if species is None:
                species = list(results.species)

            ssa = results.ssa
            kind = result_kind or _infer_result_kind(ssa, getattr(results, "pde", None))

            if kind == "timeseries":
                pde = _ensure_pde_for_timeseries(results)
                time = results.time
            elif kind == "trajectories":
                pde = _ensure_pde_for_trajectories(results)
                time = results.time
            else:
                raise ValueError(f"Unsupported inferred result kind '{kind}'")

            payload: Dict[str, Any] = {
                "format_version": np.array(1),
                "result_kind": np.array(kind),
                "time": time,
                "ssa": ssa,
                "pde": pde,
                "species": np.array(list(species), dtype=object),
                "domain_length": float(domain.length),
                "n_ssa": int(domain.K),
                "pde_multiple": int(domain.pde_multiple),
                "boundary": str(domain.boundary),
            }

            meta_out.setdefault("species", list(species))
            meta_out.setdefault("domain", _domain_to_metadata(domain))
            meta_out.setdefault("result_kind", kind)

            payload["meta_json"] = json.dumps(_json_safe(meta_out))
            np.savez_compressed(path, **payload)
            print(f"Saved results to {path}")
            return

        # final-only mode
        if final_ssa is None or final_pde is None or t_final is None or domain is None or species is None:
            raise ValueError(
                "For final-only saving you must provide final_ssa, final_pde, "
                "t_final, domain, and species."
            )

        payload = {
            "format_version": np.array(1),
            "result_kind": np.array("final_only"),
            "final_ssa": final_ssa,
            "final_pde": final_pde,
            "t_final": np.array(float(t_final)),
            "species": np.array(list(species), dtype=object),
            "domain_length": float(domain.length),
            "n_ssa": int(domain.K),
            "pde_multiple": int(domain.pde_multiple),
            "boundary": str(domain.boundary),
        }

        meta_out.setdefault("species", list(species))
        meta_out.setdefault("domain", _domain_to_metadata(domain))
        meta_out.setdefault("result_kind", "final_only")

        payload["meta_json"] = json.dumps(_json_safe(meta_out))
        np.savez_compressed(path, **payload)
        print(f"Saved final-only results to {path}")

    @staticmethod
    def load(path: PathLike) -> Tuple[Any, Dict[str, Any]]:
        """
        Load results from a single .npz file.

        Returns
        -------
        (obj, meta)

        obj is:
          - SimulationResults for timeseries / trajectories
          - dict with final_ssa/final_pde/t_final/domain/species for final_only
        """
        data = np.load(str(path), allow_pickle=True)

        meta: Dict[str, Any] = {}
        if "meta_json" in data.files:
            try:
                meta = json.loads(str(data["meta_json"]))
            except (json.JSONDecodeError, TypeError):
                meta = {}

        domain = Domain(
            length=float(data["domain_length"]),
            n_ssa=int(data["n_ssa"]),
            pde_multiple=int(data["pde_multiple"]),
            boundary=str(data["boundary"]),
        )
        species = [str(s) for s in data["species"].tolist()]
        result_kind = str(data["result_kind"]) if "result_kind" in data.files else "timeseries"

        if result_kind in ("timeseries", "trajectories"):
            res = SimulationResults(
                time=data["time"],
                ssa=data["ssa"],
                pde=data["pde"] if "pde" in data.files else None,
                domain=domain,
                species=species,
            )
            return res, meta

        if result_kind == "final_only":
            out = {
                "final_ssa": data["final_ssa"],
                "final_pde": data["final_pde"],
                "t_final": float(data["t_final"]),
                "domain": domain,
                "species": species,
            }
            return out, meta

        raise ValueError(f"Unknown result_kind '{result_kind}' in {path}")

    @staticmethod
    def save_from_engine_run(
        path: PathLike,
        *,
        results: Optional[SimulationResults] = None,
        final_tuple: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
        domain: Optional[Domain] = None,
        species: Optional[list[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Thin convenience wrapper for engine outputs.

        Use with either:
          - results=<SimulationResults>
          - final_tuple=(final_ssa, final_pde, t_final)
        """
        if results is not None:
            ResultsIO.save(results=results, path=path, meta=meta)
            return

        if final_tuple is not None:
            if domain is None or species is None:
                raise ValueError("domain and species are required when saving final_tuple")
            final_ssa, final_pde, t_final = final_tuple
            ResultsIO.save(
                path=path,
                final_ssa=final_ssa,
                final_pde=final_pde,
                t_final=t_final,
                domain=domain,
                species=species,
                meta=meta,
            )
            return

        raise ValueError("Provide either results=... or final_tuple=...")