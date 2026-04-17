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

hybrid_path = "output/two_species_toy_prob.npz"

# ------------------------------------------------------------
# Load hybrid data
# ------------------------------------------------------------
data = np.load(hybrid_path, allow_pickle=True)

meta = {}
if "meta_json" in data.files:
    try:
        meta = json.loads(data["meta_json"].item())
    except Exception:
        meta = {}

time = np.asarray(data["time"])
hybrid_ssa = np.asarray(data["ssa"])   # (n_species, K, T)
hybrid_pde = np.asarray(data["pde"])   # (n_species, Npde, T)

domain_length = float(data["domain_length"])
n_ssa = int(data["n_ssa"])
pde_multiple = int(data["pde_multiple"])
n_pde = n_ssa * pde_multiple

n_species = hybrid_ssa.shape[0]
T = hybrid_ssa.shape[2]

species_names = meta.get("species", [f"Species {i}" for i in range(n_species)])

# ------------------------------------------------------------
# Grid setup
# ------------------------------------------------------------
h_ssa = domain_length / n_ssa
h_pde = domain_length / n_pde

ssa_x = np.linspace(0, domain_length - h_ssa, n_ssa)
pde_x = np.linspace(h_pde / 2, domain_length - h_pde / 2, n_pde)

hybrid_ssa_density = hybrid_ssa / h_ssa

def combined_density_on_ssa(species_idx, frame):
    ssa_mass = hybrid_ssa[species_idx, :, frame]
    pde_density = hybrid_pde[species_idx, :, frame]
    pde_mass_blocks = (pde_density * h_pde).reshape(n_ssa, pde_multiple).sum(axis=1)
    combined_mass = ssa_mass + pde_mass_blocks
    return combined_mass / h_ssa

# ------------------------------------------------------------
# Threshold handling
# ------------------------------------------------------------
threshold_meta = meta.get("conversion_threshold", {})

def get_thresholds_for_species(species_name):
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
            return float(val[0]), float(val[1])   # (CD, DC)
        if len(val) == 1:
            return float(val[0]), float(val[0])

    if isinstance(val, (int, float, np.integer, np.floating)):
        v = float(val)
        return v, v

    return None, None

# ------------------------------------------------------------
# y-limits
# ------------------------------------------------------------
ymax = []
probe_frames = [0, T // 2, T - 1]

for s in range(n_species):
    vals = [
        float(hybrid_ssa_density[s].max()),
        float(hybrid_pde[s].max()),
        max(float(combined_density_on_ssa(s, k).max()) for k in probe_frames),
    ]

    cd_thr, dc_thr = get_thresholds_for_species(species_names[s])
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

fig.suptitle("Hybrid animation", fontsize=16)

bars_ssa_list = []
bars_combined_list = []
line_pde_list = []
time_text_list = []
cd_line_list = []
dc_line_list = []

for s in range(n_species):
    ax = axes[s]

    bars_ssa = ax.bar(
        ssa_x,
        hybrid_ssa_density[s, :, 0],
        width=h_ssa,
        align="edge",
        alpha=0.65,
        label="Hybrid SSA",
    )

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

    line_pde, = ax.plot(
        pde_x,
        hybrid_pde[s, :, 0],
        linestyle="--",
        linewidth=2,
        label="Hybrid PDE",
    )

    # Threshold lines
    sp_name = species_names[s]
    cd_thr, dc_thr = get_thresholds_for_species(sp_name)

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

    ax.set_title(sp_name)
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

    bars_ssa_list.append(bars_ssa)
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
        for bar, h in zip(bars_ssa_list[s], hybrid_ssa_density[s, :, frame]):
            bar.set_height(float(h))
            artists.append(bar)

        comb = combined_density_on_ssa(s, frame)
        for bar, h in zip(bars_combined_list[s], comb):
            bar.set_height(float(h))
            artists.append(bar)

        line_pde_list[s].set_ydata(hybrid_pde[s, :, frame])
        artists.append(line_pde_list[s])

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