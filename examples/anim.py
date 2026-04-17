import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
STRIDE = 1
INTERVAL_MS = 500
BLIT = False
REPEAT = True

hybrid_path = "output/diffusion_hybrid_bool.npz"   # or None
ssa_path = "output/diffusion_SSA.npz"                           # or None
pde_path = "output/diffusion_PDE.npz"                           # or None

if hybrid_path is None and ssa_path is None and pde_path is None:
    raise ValueError("At least one of hybrid_path, ssa_path, or pde_path must be provided.")

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

def ensure_TN(arr, N_expected):
    """
    Force PDE array to shape (T, N).
    Accepts either (T, N) or (N, T).
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")

    T, N = arr.shape
    if N == N_expected:
        return arr
    if T == N_expected:
        return arr.T

    raise ValueError(f"Array shape {arr.shape} doesn't match expected spatial size N={N_expected}")

def find_one_species_array(npz):
    """
    Try likely keys for one-species PDE data.
    """
    keys = set(npz.files)
    candidates = [
        "u_record", "u_rec", "u", "U",
        "record", "solution"
    ]
    for k in candidates:
        if k in keys:
            return np.asarray(npz[k])

    raise KeyError(f"Could not find one-species PDE array in keys: {npz.files}")

def find_multi_species_arrays(npz, n_species_expected):
    """
    Try likely key pairs for multi-species PDE data.
    """
    keys = set(npz.files)
    candidate_lists = [
        ["u_record", "v_record"],
        ["U", "V"],
        ["u", "v"],
        ["A", "B"],
    ]

    for pair in candidate_lists:
        if len(pair) >= n_species_expected and all(k in keys for k in pair[:n_species_expected]):
            return [np.asarray(npz[k]) for k in pair[:n_species_expected]]

    raise KeyError(f"Could not find {n_species_expected} PDE species arrays in keys: {npz.files}")

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
        if pure_ssa.shape[0] != n_species:
            raise ValueError("SSA and hybrid files have different numbers of species")
        if pure_ssa.shape[1] != n_ssa:
            raise ValueError("SSA and hybrid files have different n_ssa")
        if pure_ssa.shape[2] != T:
            raise ValueError("SSA and hybrid files have different number of time points")

# ------------------------------------------------------------
# Load pure PDE data (optional)
# ------------------------------------------------------------
pde_meta = {}
pde_data = None
pure_pde = None        # shape: (S, T_pde, N_pde)
time_pde = None

if pde_path is not None:
    pde_data, pde_meta = load_npz_with_meta(pde_path)

    # if no hybrid/ssa file set canonical geometry, try to infer it from PDE file
    if domain_length is None:
        if "L" not in pde_data.files:
            raise ValueError("Pure PDE file must contain 'L' if hybrid/ssa file is absent")
        domain_length = float(np.asarray(pde_data["L"]).item())

    if n_pde is None:
        if "x" in pde_data.files:
            x_pde_tmp = np.asarray(pde_data["x"])
            n_pde = len(x_pde_tmp)
        elif "n" in pde_data.files:
            n_pde = int(np.asarray(pde_data["n"]).item())
        else:
            raise ValueError("Could not infer n_pde from PDE file")

    if n_species is None:
        # infer from available arrays
        pde_keys = set(pde_data.files)
        if any(k in pde_keys for k in ["v_record", "V", "v"]):
            n_species = 2
            species_names = ["U", "V"] if species_names is None else species_names
        else:
            n_species = 1
            species_names = ["U"] if species_names is None else species_names

    if n_ssa is None and pde_multiple is None:
        # no SSA/hybrid reference grid, so fake an SSA grid equal to PDE grid for plotting compatibility
        n_ssa = n_pde
        pde_multiple = 1

    if len(species_names) != n_species:
        species_names = [f"Species {i}" for i in range(n_species)]

    if n_species == 1:
        U_raw = find_one_species_array(pde_data)
        U_TN = ensure_TN(U_raw, n_pde)   # (T, N)
        pure_pde = U_TN[None, :, :]      # (1, T, N)
    else:
        raw_list = find_multi_species_arrays(pde_data, n_species)
        pure_pde = np.stack([ensure_TN(arr, n_pde) for arr in raw_list], axis=0)  # (S, T, N)

    if "timevector" in pde_data.files:
        time_pde = np.asarray(pde_data["timevector"])
    elif "time" in pde_data.files:
        time_pde = np.asarray(pde_data["time"])
    elif "t" in pde_data.files:
        time_pde = np.asarray(pde_data["t"])
    elif "dt" in pde_data.files:
        dtp = float(np.asarray(pde_data["dt"]).item())
        time_pde = np.arange(pure_pde.shape[1]) * dtp

    if time is None:
        if time_pde is None:
            raise ValueError("Could not infer time vector from PDE file")
        time = time_pde.copy()
        T = len(time)

# ------------------------------------------------------------
# Compatibility checks for pure PDE
# ------------------------------------------------------------
if pure_pde is not None:
    if pure_pde.shape[0] != n_species:
        raise ValueError("Pure PDE and other data have different numbers of species")
    if pure_pde.shape[2] != n_pde:
        raise ValueError("Pure PDE and hybrid grid have different n_pde")

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

def map_time_to_pde_frame(frame):
    if pure_pde is None:
        return None
    if time_pde is None:
        return int(np.clip(frame, 0, pure_pde.shape[1] - 1))
    t = float(time[frame])
    return int(np.argmin(np.abs(time_pde - t)))

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

    if pure_pde is not None:
        vals.append(float(pure_pde[s].max()))

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
if pde_path is not None:
    title_bits.append("PDE")

fig.suptitle(" + ".join(title_bits) + " animation", fontsize=16)

bars_hybrid_ssa_list = []
bars_pure_ssa_list = []
bars_combined_list = []
line_hybrid_pde_list = []
line_pure_pde_list = []
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
        line_hybrid_pde, = ax.plot(
            pde_x,
            hybrid_pde[s, :, 0],
            linestyle="--",
            linewidth=2,
            label="Hybrid PDE",
        )
    else:
        line_hybrid_pde = None

    # Pure PDE line
    if pure_pde is not None:
        fp0 = map_time_to_pde_frame(0)
        line_pure_pde, = ax.plot(
            pde_x,
            pure_pde[s, fp0, :],
            linestyle="-",
            linewidth=2,
            label="Pure PDE",
        )
    else:
        line_pure_pde = None

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
    line_hybrid_pde_list.append(line_hybrid_pde)
    line_pure_pde_list.append(line_pure_pde)
    time_text_list.append(time_text)
    cd_line_list.append(cd_line)
    dc_line_list.append(dc_line)

# ------------------------------------------------------------
# Animation
# ------------------------------------------------------------
frames = list(range(0, T, max(1, int(STRIDE))))

def update(frame):
    artists = []
    frame_p = map_time_to_pde_frame(frame)

    for s in range(n_species):
        # Hybrid SSA
        if bars_hybrid_ssa_list[s] is not None:
            for bar, h in zip(bars_hybrid_ssa_list[s], hybrid_ssa_density[s, :, frame]):
                bar.set_height(float(h))
                artists.append(bar)

        # Pure SSA
        if bars_pure_ssa_list[s] is not None:
            frame_s = min(frame, pure_ssa_density.shape[2] - 1)
            for bar, h in zip(bars_pure_ssa_list[s], pure_ssa_density[s, :, frame_s]):
                bar.set_height(float(h))
                artists.append(bar)

        # Combined
        if bars_combined_list[s] is not None:
            comb = combined_density_on_ssa(s, frame)
            for bar, h in zip(bars_combined_list[s], comb):
                bar.set_height(float(h))
                artists.append(bar)

        # Hybrid PDE
        if line_hybrid_pde_list[s] is not None:
            line_hybrid_pde_list[s].set_ydata(hybrid_pde[s, :, frame])
            artists.append(line_hybrid_pde_list[s])

        # Pure PDE
        if line_pure_pde_list[s] is not None and frame_p is not None:
            line_pure_pde_list[s].set_ydata(pure_pde[s, frame_p, :])
            artists.append(line_pure_pde_list[s])

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