import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
STRIDE = 5
INTERVAL_MS = 40
BLIT = False
REPEAT = True

hybrid_path = "output/production_degradation_hybrid_prob.npz"   # or None
ssa_path = "output/production_degradation_ssa.npz"              # or None

if hybrid_path is None and ssa_path is None:
    raise ValueError("At least one of hybrid_path or ssa_path must be provided.")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_npz_with_meta(path):
    data = np.load(path, allow_pickle=True)
    meta = {}
    if "meta_json" in data.files:
        try:
            meta = json.loads(data["meta_json"].item())
        except Exception:
            meta = {}
    return data, meta

def get_thresholds_for_species(threshold_meta, species_name):
    """
    Returns (cd_threshold_particles, dc_threshold_particles)
    from metadata.

    Supported formats:
      {"U": [20, 30]}   -> CD=20, DC=30
      {"U": 20}         -> CD=DC=20
      20                -> CD=DC=20
      [20, 30]          -> global CD/DC for all species
    """
    if threshold_meta is None:
        return None, None

    val = None
    if isinstance(threshold_meta, dict):
        val = threshold_meta.get(species_name, None)
    else:
        val = threshold_meta

    if val is None:
        return None, None

    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 2:
            return float(val[0]), float(val[1])
        if len(val) == 1:
            return float(val[0]), float(val[0])

    if isinstance(val, (int, float, np.integer, np.floating)):
        v = float(val)
        return v, v

    return None, None

# ------------------------------------------------------------
# Load hybrid data (optional)
# ------------------------------------------------------------
hybrid_meta = {}
hybrid_data = None
hybrid_ssa = None
hybrid_pde = None
hybrid_ssa_density = None
threshold_meta = None

# canonical grid/time info
time = None
domain_length = None
n_ssa = None
pde_multiple = None
n_pde = None
n_species = None
T = None
species_names = None

if hybrid_path is not None:
    hybrid_data, hybrid_meta = load_npz_with_meta(hybrid_path)

    time = np.asarray(hybrid_data["time"])
    hybrid_ssa = np.asarray(hybrid_data["ssa"])   # (S, K, T)
    hybrid_pde = np.asarray(hybrid_data["pde"])   # (S, Npde, T)

    domain_length = float(hybrid_data["domain_length"])
    n_ssa = int(hybrid_data["n_ssa"])
    pde_multiple = int(hybrid_data["pde_multiple"])
    n_pde = n_ssa * pde_multiple

    n_species = hybrid_ssa.shape[0]
    T = hybrid_ssa.shape[2]
    species_names = hybrid_meta.get("species", [f"Species {i}" for i in range(n_species)])
    threshold_meta = hybrid_meta.get("conversion_threshold", None)

# ------------------------------------------------------------
# Load SSA data (optional)
# ------------------------------------------------------------
ssa_meta = {}
ssa_data = None
pure_ssa = None
pure_ssa_density = None

if ssa_path is not None:
    ssa_data, ssa_meta = load_npz_with_meta(ssa_path)

    ssa_time = np.asarray(ssa_data["time"])
    pure_ssa = np.asarray(ssa_data["ssa"])   # (S, K, T)

    if time is None:
        time = ssa_time
        domain_length = float(ssa_data["domain_length"])
        n_ssa = int(ssa_data["n_ssa"])
        pde_multiple = int(ssa_data["pde_multiple"])
        n_pde = n_ssa * pde_multiple
        n_species = pure_ssa.shape[0]
        T = pure_ssa.shape[2]
        species_names = ssa_meta.get("species", [f"Species {i}" for i in range(n_species)])

    else:
        # basic compatibility checks
        if pure_ssa.shape[0] != n_species:
            raise ValueError("SSA and hybrid files have different numbers of species")
        if pure_ssa.shape[1] != n_ssa:
            raise ValueError("SSA and hybrid files have different n_ssa")
        if pure_ssa.shape[2] != T:
            raise ValueError("SSA and hybrid files have different number of time points")

# ------------------------------------------------------------
# Grid setup
# ------------------------------------------------------------
h_ssa = domain_length / n_ssa
h_pde = domain_length / n_pde

ssa_x = np.linspace(0, domain_length - h_ssa, n_ssa)
pde_x = np.linspace(h_pde / 2, domain_length - h_pde / 2, n_pde)

if hybrid_ssa is not None:
    hybrid_ssa_density = hybrid_ssa / h_ssa

if pure_ssa is not None:
    pure_ssa_density = pure_ssa / h_ssa

def combined_density_on_ssa(species_idx, frame):
    if hybrid_ssa is None or hybrid_pde is None:
        return None
    ssa_mass = hybrid_ssa[species_idx, :, frame]
    pde_density = hybrid_pde[species_idx, :, frame]
    pde_mass_blocks = (pde_density * h_pde).reshape(n_ssa, pde_multiple).sum(axis=1)
    combined_mass = ssa_mass + pde_mass_blocks
    return combined_mass / h_ssa

# ------------------------------------------------------------
# y-limits
# ------------------------------------------------------------
ymax = []
probe_frames = [0, T // 2, T - 1]

for s in range(n_species):
    vals = []

    if hybrid_ssa_density is not None:
        vals.append(float(hybrid_ssa_density[s].max()))

    if hybrid_pde is not None:
        vals.append(float(hybrid_pde[s].max()))

    if pure_ssa_density is not None:
        vals.append(float(pure_ssa_density[s].max()))

    if hybrid_ssa is not None and hybrid_pde is not None:
        vals.append(max(float(combined_density_on_ssa(s, k).max()) for k in probe_frames))

    if threshold_meta is not None:
        cd_thr, dc_thr = get_thresholds_for_species(threshold_meta, species_names[s])
        if cd_thr is not None:
            vals.append(cd_thr / h_ssa)
        if dc_thr is not None:
            vals.append(dc_thr / h_ssa)

    vmax = max(vals) if vals else 1.0
    ymax.append(1.15 * vmax if vmax > 0 else 1.0)

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig, axes = plt.subplots(1, n_species, figsize=(8 * n_species, 5))
if n_species == 1:
    axes = [axes]

title_bits = []
if hybrid_path is not None:
    title_bits.append("Hybrid")
if ssa_path is not None:
    title_bits.append("SSA")

fig.suptitle(" + ".join(title_bits) + " animation", fontsize=16)

bars_hybrid_ssa_list = []
bars_pure_ssa_list = []
bars_combined_list = []
line_pde_list = []
time_text_list = []
cd_line_list = []
dc_line_list = []

for s in range(n_species):
    ax = axes[s]

    # Hybrid SSA bars
    if hybrid_ssa_density is not None:
        bars_hybrid_ssa = ax.bar(
            ssa_x,
            hybrid_ssa_density[s, :, 0],
            width=h_ssa,
            align="edge",
            alpha=0.65,
            label="Hybrid SSA",
        )
    else:
        bars_hybrid_ssa = None

    # Pure SSA bars
    if pure_ssa_density is not None:
        bars_pure_ssa = ax.bar(
            ssa_x,
            pure_ssa_density[s, :, 0],
            width=h_ssa,
            align="edge",
            alpha=0.35,
            label="Pure SSA",
        )
    else:
        bars_pure_ssa = None

    # Hybrid combined bars
    if hybrid_ssa is not None and hybrid_pde is not None:
        comb0 = combined_density_on_ssa(s, 0)
        bars_combined = ax.bar(
            ssa_x,
            comb0,
            width=h_ssa,
            align="edge",
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
            label="Combined",
        )
    else:
        bars_combined = None

    # Hybrid PDE line
    if hybrid_pde is not None:
        line_pde, = ax.plot(
            pde_x,
            hybrid_pde[s, :, 0],
            linestyle="--",
            linewidth=2,
            label="Hybrid PDE",
        )
    else:
        line_pde = None

    # Threshold lines (hybrid metadata only)
    if threshold_meta is not None:
        sp_name = species_names[s]
        cd_thr, dc_thr = get_thresholds_for_species(threshold_meta, sp_name)

        if cd_thr is not None:
            cd_line = ax.axhline(
                cd_thr / h_ssa,
                color="green",
                linestyle=":",
                linewidth=2,
                label=f"CD threshold ({cd_thr:g})",
            )
        else:
            cd_line = None

        if dc_thr is not None:
            dc_line = ax.axhline(
                dc_thr / h_ssa,
                color="purple",
                linestyle="--",
                linewidth=2,
                label=f"DC threshold ({dc_thr:g})",
            )
        else:
            dc_line = None
    else:
        cd_line = None
        dc_line = None

    ax.set_title(species_names[s])
    ax.set_xlim(0, domain_length)
    ax.set_ylim(0, ymax[s])
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    bars_hybrid_ssa_list.append(bars_hybrid_ssa)
    bars_pure_ssa_list.append(bars_pure_ssa)
    bars_combined_list.append(bars_combined)
    line_pde_list.append(line_pde)
    time_text_list.append(time_text)
    cd_line_list.append(cd_line)
    dc_line_list.append(dc_line)

# ------------------------------------------------------------
# Animation
# ------------------------------------------------------------
frames = list(range(0, T, max(1, int(STRIDE))))

def update(frame):
    artists = []

    for s in range(n_species):
        # Hybrid SSA
        if bars_hybrid_ssa_list[s] is not None:
            for bar, h in zip(bars_hybrid_ssa_list[s], hybrid_ssa_density[s, :, frame]):
                bar.set_height(float(h))
                artists.append(bar)

        # Pure SSA
        if bars_pure_ssa_list[s] is not None:
            for bar, h in zip(bars_pure_ssa_list[s], pure_ssa_density[s, :, frame]):
                bar.set_height(float(h))
                artists.append(bar)

        # Combined
        if bars_combined_list[s] is not None:
            comb = combined_density_on_ssa(s, frame)
            for bar, h in zip(bars_combined_list[s], comb):
                bar.set_height(float(h))
                artists.append(bar)

        # PDE line
        if line_pde_list[s] is not None:
            line_pde_list[s].set_ydata(hybrid_pde[s, :, frame])
            artists.append(line_pde_list[s])

        # time
        time_text_list[s].set_text(f"t = {time[frame]:.3f}")
        artists.append(time_text_list[s])

        if cd_line_list[s] is not None:
            artists.append(cd_line_list[s])
        if dc_line_list[s] is not None:
            artists.append(dc_line_list[s])

    return artists

ani = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=INTERVAL_MS,
    blit=BLIT,
    repeat=REPEAT,
    cache_frame_data=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()