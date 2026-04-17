import sys
import json
import os
import numpy as np
from pathlib import Path

# -----------------------------
# Pretty terminal helpers (no deps)
# -----------------------------
def _supports_color() -> bool:
    return sys.stdout.isatty() and (sys.platform != "win32" or "WT_SESSION" in os.environ or "TERM" in os.environ)


try:
    _COLOR = _supports_color()
except Exception:
    _COLOR = False

RESET = "\033[0m" if _COLOR else ""
BOLD = "\033[1m" if _COLOR else ""
DIM = "\033[2m" if _COLOR else ""

FG = {
    "cyan": "\033[36m" if _COLOR else "",
    "green": "\033[32m" if _COLOR else "",
    "yellow": "\033[33m" if _COLOR else "",
    "red": "\033[31m" if _COLOR else "",
    "blue": "\033[34m" if _COLOR else "",
    "mag": "\033[35m" if _COLOR else "",
    "gray": "\033[90m" if _COLOR else "",
}


def c(text: str, color: str = None, *, bold=False, dim=False) -> str:
    if not _COLOR:
        return text
    prefix = ""
    if bold:
        prefix += BOLD
    if dim:
        prefix += DIM
    if color:
        prefix += FG.get(color, "")
    return f"{prefix}{text}{RESET}"


def hr(width: int = 78, char: str = "─") -> str:
    return char * width


def box_title(title: str, width: int = 78) -> str:
    title = f" {title} "
    inner = width - 2
    if len(title) > inner:
        title = title[: inner - 1] + "…"
    pad_left = (inner - len(title)) // 2
    pad_right = inner - len(title) - pad_left
    return f"┌{('─' * pad_left)}{title}{('─' * pad_right)}┐"


def box_line(text: str, width: int = 78) -> str:
    inner = width - 2
    if len(text) > inner:
        text = text[: inner - 1] + "…"
    return f"│{text.ljust(inner)}│"


def box_bottom(width: int = 78) -> str:
    return f"└{('─' * (width - 2))}┘"


def badge(label: str, color: str) -> str:
    return c(f"[{label}]", color=color, bold=True)


def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def format_stoich(species_data):
    """Turns {'A': 1, 'B': 2} into 'A + 2B'."""
    if not species_data:
        return "∅"
    if isinstance(species_data, dict):
        parts = []
        for k in sorted(species_data.keys()):
            v = species_data[k]
            if v > 0:
                parts.append(f"{v if v > 1 else ''}{k}")
        return " + ".join(parts) if parts else "∅"
    return str(species_data)


def try_decode_meta(meta_raw):
    """
    Returns (decoded_dict, err)
    - decoded_dict: dict | None
    - err: Exception | None
    """
    try:
        if hasattr(meta_raw, "shape") and meta_raw.shape == ():
            meta_raw = meta_raw.item()

        if isinstance(meta_raw, (bytes, bytearray)):
            meta_raw = meta_raw.decode("utf-8", errors="replace")

        decoded = json.loads(str(meta_raw))
        return decoded, None

    except Exception as e:
        return None, e


def print_table(rows, headers, col_widths, *, indent=0):
    """Simple fixed-width table with box-drawing separators."""
    pad = " " * indent
    header_line = " │ ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep = "─" * len(header_line)
    print(pad + c(header_line, "gray", bold=True))
    print(pad + c(sep, "gray", dim=True))
    for r in rows:
        line = " │ ".join(str(cell).ljust(w)[:w] for cell, w in zip(r, col_widths))
        print(pad + line)


def _fmt_threshold_value(x):
    """Pretty-print scalar/list/dict threshold values."""
    if isinstance(x, dict):
        return ", ".join(f"{k}:{x[k]}" for k in sorted(x.keys()))
    if isinstance(x, (list, tuple)):
        return str(list(x))
    return str(x)


def _extract_conversion_meta(meta: dict):
    """
    Returns a normalized conversion dict:
      {
        'DC_threshold': ... | None,
        'CD_threshold': ... | None,
        'rate': ... | None,
      }
    Supports both new and legacy metadata layouts.
    """
    conv = {"DC_threshold": None, "CD_threshold": None, "rate": None}

    if not isinstance(meta, dict):
        return conv

    # New format
    raw_conv = meta.get("conversion")
    if isinstance(raw_conv, dict):
        conv["DC_threshold"] = raw_conv.get("DC_threshold")
        conv["CD_threshold"] = raw_conv.get("CD_threshold")
        conv["rate"] = raw_conv.get("rate")

    # Legacy fallbacks
    if conv["CD_threshold"] is None and "threshold_particles" in meta:
        conv["CD_threshold"] = meta.get("threshold_particles")
    if conv["rate"] is None and "conversion_rate" in meta:
        conv["rate"] = meta.get("conversion_rate")

    return conv


def inspect_npz(path: Path, width: int = 78):
    print()
    print(box_title(c("NPZ INSPECTOR", "cyan", bold=True), width))
    print(box_line(f"{badge('FILE', 'blue')} {c(path.name, bold=True)}", width))
    print(box_line(f"{DIM}{str(path.resolve())}{RESET}", width))
    print(box_bottom(width))

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(c(f"{badge('ERROR', 'red')} Could not read file: {e}", "red", bold=True))
        return

    # -----------------------------
    # 1) Array overview
    # -----------------------------
    print()
    print(c("▌ ARRAY OVERVIEW", "mag", bold=True))
    rows = []
    for k in data.files:
        v = data[k]
        if getattr(v, "shape", None) == ():
            dtype = getattr(v, "dtype", "scalar")
            rows.append((k, "scalar", f"dtype={dtype}", str(v)))
        else:
            shape_str = str(v.shape)
            dtype = getattr(v, "dtype", "?")
            peek = ""
            if k == "species":
                try:
                    peek = f"{v.tolist()}"
                except Exception:
                    peek = "<unprintable>"
            rows.append((k, "array", f"shape={shape_str}, dtype={dtype}", peek))

    print_table(
        rows=rows,
        headers=("KEY", "KIND", "DETAILS", "PEEK"),
        col_widths=(18, 8, 30, 18),
        indent=0,
    )

    # -----------------------------
    # 2) Metadata deep dive
    # -----------------------------
    meta = {}
    if "meta_json" in data.files:
        print()
        print(c("▌ METADATA", "mag", bold=True))
        decoded, err = try_decode_meta(data["meta_json"])
        if decoded is None:
            print(c(f"{badge('WARN', 'yellow')} Could not decode meta_json: {err}", "yellow"))
        else:
            meta = decoded

            conv = _extract_conversion_meta(meta)
            dc_thr = conv["DC_threshold"]
            cd_thr = conv["CD_threshold"]
            rate = conv["rate"]

            if dc_thr is not None or cd_thr is not None:
                print(f"{badge('HYBRID', 'green')} Conversion thresholds:")
                if dc_thr is not None:
                    print(f"  - DC_threshold: {c(_fmt_threshold_value(dc_thr), bold=True)}")
                if cd_thr is not None:
                    print(f"  - CD_threshold: {c(_fmt_threshold_value(cd_thr), bold=True)}")

            if rate is not None:
                print(f"{badge('RATE', 'green')} Conversion rate: {c(str(rate), bold=True)}")

            if "diffusion_rates" in meta and isinstance(meta["diffusion_rates"], dict):
                diffs = ", ".join(f"{k}:{v}" for k, v in meta["diffusion_rates"].items())
                print(f"{badge('DIFF', 'green')} {diffs}")

            # Reactions
            if "reactions" in meta:
                print()
                print(c("▌ REACTION FRAMEWORK", "mag", bold=True))

                r_rows = []
                for i, r in enumerate(meta["reactions"], 1):
                    try:
                        if isinstance(r, dict):
                            reac = format_stoich(r.get("reactants"))
                            prod = format_stoich(r.get("products"))
                            name = r.get("rate_name", "n/a")
                            val = r.get("rate", meta.get("reaction_rates", {}).get(name, "??"))
                        elif isinstance(r, (list, tuple)):
                            reac = format_stoich(r[0]) if len(r) > 0 else "?"
                            prod = format_stoich(r[1]) if len(r) > 1 else "?"
                            name = r[2] if len(r) > 2 else "n/a"
                            val = meta.get("reaction_rates", {}).get(name, "??")
                        else:
                            reac, prod, name, val = "?", "?", "n/a", "??"

                        arrow = c("→", "cyan", bold=True)
                        r_rows.append((f"{i}", f"{reac} {arrow} {prod}", name, val))
                    except Exception:
                        r_rows.append((f"{i}", "<parse error>", "n/a", "??"))

                print_table(
                    rows=r_rows,
                    headers=("#", "REACTION", "RATE NAME", "VALUE"),
                    col_widths=(3, 42, 12, 10),
                    indent=0,
                )

    # -----------------------------
    # 3) Spatial / temporal summary
    # -----------------------------
    print()
    print(c("▌ SPATIAL / TEMPORAL", "mag", bold=True))

    t_key = next((k for k in data.files if "time" in k.lower()), None)
    if t_key:
        try:
            t = data[t_key]
            if len(t) >= 2:
                print(f"{badge('TIME', 'blue')} {len(t)} steps  {DIM}({t[0]:.3f} → {t[-1]:.3f}){RESET}")
            else:
                print(f"{badge('TIME', 'blue')} {len(t)} steps")
        except Exception as e:
            print(c(f"{badge('WARN', 'yellow')} Couldn't read time array: {e}", "yellow"))
    else:
        print(f"{badge('TIME', 'blue')} {DIM}No time array found{RESET}")

    # K and PDE multiple
    K = None
    try:
        if "n_ssa" in data.files:
            K = data["n_ssa"].item() if getattr(data["n_ssa"], "shape", None) == () else data["n_ssa"]
    except Exception:
        K = None

    if K is None:
        dom = meta.get("domain", {}) if isinstance(meta.get("domain", {}), dict) else {}
        K = dom.get("K", dom.get("n_ssa"))

    mult = None
    try:
        if "pde_multiple" in data.files:
            mult = data["pde_multiple"].item() if getattr(data["pde_multiple"], "shape", None) == () else data["pde_multiple"]
    except Exception:
        mult = None

    if mult is None:
        dom = meta.get("domain", {}) if isinstance(meta.get("domain", {}), dict) else {}
        mult = dom.get("pde_multiple", 1)

    Ki = safe_int(K)
    mi = safe_int(mult, 1)
    if Ki is not None:
        pde = Ki * (mi if mi is not None else 1)
        print(f"{badge('DOMAIN', 'blue')} K={c(str(Ki), bold=True)} (SSA)   PDE subgrid={c(str(pde), bold=True)}   {DIM}(multiple={mi}){RESET}")
    else:
        print(f"{badge('DOMAIN', 'blue')} {DIM}No domain K found (n_ssa / meta.domain.K){RESET}")

    print()
    print(c(hr(width), "gray", dim=True))


def main():
    paths = [Path(p) for p in sys.argv[1:] if Path(p).exists()]
    if not paths:
        print("Usage: python inspect_results.py <file1.npz> [file2.npz ...]")
        return
    for p in paths:
        inspect_npz(p)


if __name__ == "__main__":
    main()