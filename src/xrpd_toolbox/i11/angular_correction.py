import itertools
import os
from typing import Any

# import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peakutils
from matplotlib.axes import Axes
from matplotlib.widgets import Slider
from pyFAI.calibrant import get_calibrant
from scipy.interpolate import griddata
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from xrpd_toolbox.gui.stereographic_projection import (
    spherical_score_plot_2d,
    spherical_score_plot_3d,
)
from xrpd_toolbox.i11.mythen import (
    # CENTRE,
    # DEFAULT_ANG_CAL,
    # DEFAULT_BAD_CHANS,
    # MYTHEN_PIXEL_SIZE,
    # PIXELS_PER_MODULE,
    # PSD_RADIUS,
    # AngularCalibration,
    AngularCalibration2D,
    # BadChannels,
    # ModuleConversion,
    # ModuleConversion2D,
    MythenDetector,
    MythenSettings,
)

# from xrpd_toolbox.i11.mythen3_reduction_legacy import I11Reduction
from xrpd_toolbox.utils.mythen_utils import find_pair, paired_modules
from xrpd_toolbox.utils.utils import h5_to_array

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)

DEFAULT_COUNTER = 0
STRIPS_PER_MODULE = 1280
ACTIVE_MODULES = list(range(0, 28))

FloatOrArray = float | np.ndarray


def base_to_euler(
    jack_y1: FloatOrArray,
    jack_y2: FloatOrArray,
    jack_y3: FloatOrArray,
    plate_x1: FloatOrArray,
    plate_x2: FloatOrArray,
    jack_positions: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ] = (
        (-160.0, 150.0),  # jack_y1: (x, z) [mm]
        (160.0, 150.0),  # jack_y2
        (0.0, -260.0),  # jack_y3
    ),
    plate_arm_z_offsets: tuple[float, float] = (
        80.0,
        -80.0,
    ),  # z of plate_x1, plate_x2 arms [mm]
    reference_point: tuple[float, float] = (
        0.0,
        0.0,
    ),  # (x, z) origin for angles/translation
    euler_sequence: str = "XYZ",
) -> dict[str, np.ndarray]:
    """Convert base actuator readings to a pose (translation + pitch/roll/yaw).
    Axes: X = horizontal transverse, Y = vertical, Z = beam direction.
    Stack (bottom->top): plate_x1/plate_x2 (X + yaw) -> jack_y1/y2/y3 (Y + pitch/roll).
    Accepts scalars or numpy arrays (elementwise) for the actuator readings.
    """
    jack_y1 = np.asarray(jack_y1, dtype=float)
    jack_y2 = np.asarray(jack_y2, dtype=float)
    jack_y3 = np.asarray(jack_y3, dtype=float)
    plate_x1 = np.asarray(plate_x1, dtype=float)
    plate_x2 = np.asarray(plate_x2, dtype=float)

    reference_x, reference_z = reference_point

    # Plane fit through jack_y1, jack_y2, jack_y3 -> vertical_translation, pitch, roll
    jack_xz_coords = np.array(jack_positions, dtype=float)
    jack_heights = np.array([jack_y1, jack_y2, jack_y3], dtype=float)
    plane_fit_matrix = np.column_stack(
        [
            np.ones(3),
            jack_xz_coords[:, 0] - reference_x,
            jack_xz_coords[:, 1] - reference_z,
        ]
    )
    vertical_translation, height_slope_vs_x, height_slope_vs_z = np.linalg.solve(
        plane_fit_matrix, jack_heights
    )
    pitch_rad = np.arctan(height_slope_vs_z)
    roll_rad = np.arctan(height_slope_vs_x)

    # Line fit through plate_x1, plate_x2 -> lateral_translation, yaw
    plate_x1_z_offset, plate_x2_z_offset = plate_arm_z_offsets
    lateral_slope_vs_z = (plate_x2 - plate_x1) / (plate_x2_z_offset - plate_x1_z_offset)
    yaw_rad = np.arctan(lateral_slope_vs_z)
    lateral_translation = plate_x1 - lateral_slope_vs_z * (
        plate_x1_z_offset - reference_z
    )
    beam_axis_translation = (
        vertical_translation * 0.0
    )  # not observable; matches shape/type

    def _rotation_about(axis: str, angle_rad: FloatOrArray) -> Rotation:
        angle_rad = np.asarray(angle_rad, dtype=float)
        return Rotation.from_euler(
            axis,
            angle_rad.reshape(angle_rad.shape + (1,)) if angle_rad.ndim else angle_rad,
        )

    yaw_rotation = _rotation_about("y", yaw_rad)
    tilt_rotation = _rotation_about("z", roll_rad) * _rotation_about("x", pitch_rad)
    combined_rotation = yaw_rotation * tilt_rotation

    euler_angles_deg = combined_rotation.as_euler(euler_sequence, degrees=True)

    return {
        "lateral_translation": lateral_translation,
        "vertical_translation": vertical_translation,
        "beam_axis_translation": beam_axis_translation,
        "pitch_deg": np.degrees(pitch_rad),
        "roll_deg": np.degrees(roll_rad),
        "yaw_deg": np.degrees(yaw_rad),
        "euler_angles_deg": euler_angles_deg,
        "euler_sequence": euler_sequence,  # type: ignore
        "rotation_matrix": combined_rotation.as_matrix(),
    }


def make_steps(log_data: pd.DataFrame, param_cols: list[str], n_steps=10):

    min_max_val = {}

    for position in param_cols:
        min_val, max_val = np.amin(log_data[position]), np.amax(log_data[position])
        min_max_val[position] = (min_val, max_val)

    min_max_val["Y3"] = (-0.7, 0.7)

    grids = [np.linspace(lo, hi, n_steps) for lo, hi in min_max_val.values()]
    combos = list(itertools.product(*grids))
    df = pd.DataFrame(combos, columns=param_cols)

    print(df)

    return min_max_val


# def plot_static_pca(log_data: pd.DataFrame):

#     plt.figure(figsize=(8, 6))

#     plt.contourf(XX, YY, ZZ, levels=100, cmap="RdYlGn_r")

#     plt.scatter(
#         X_pca[:, 0],
#         X_pca[:, 1],
#         c=scores,
#         cmap="RdYlGn_r",
#         edgecolor="k",
#         s=20,
#     )

#     plt.colorbar(label="distance score")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.title("Interpolated score landscape")
#     plt.savefig(f"{comissioning_directory}/pca.png")
#     plt.show()


def plot_pca(log_data: pd.DataFrame, param_cols: list[str]):

    # PCA
    position_values = log_data[param_cols].values
    pca = PCA(n_components=3)
    position_values_pca = pca.fit_transform(position_values)
    scores = log_data["score"].to_numpy()

    # Grid over PC1/PC2
    n = 200
    xg = np.linspace(
        position_values_pca[:, 0].min(), position_values_pca[:, 0].max(), n
    )
    yg = np.linspace(
        position_values_pca[:, 1].min(), position_values_pca[:, 1].max(), n
    )
    x_mesh, y_mesh = np.meshgrid(xg, yg)

    pc3_min = position_values_pca[:, 2].min()
    pc3_max = position_values_pca[:, 2].max()
    pc3_value = np.median(position_values_pca[:, 2])

    # Thickness of displayed slice
    slice_width = 0.1 * (pc3_max - pc3_min)

    def interpolate_slice(pc3):
        query = np.column_stack(
            [x_mesh.ravel(), y_mesh.ravel(), np.full(x_mesh.size, pc3)]
        )

        z_mesh = griddata(
            position_values_pca,
            scores,
            query,
            method="linear",
        )

        return z_mesh.reshape(x_mesh.shape)

    # Initial slice
    z_mesh = interpolate_slice(pc3_value)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)

    _ = np.isfinite(z_mesh)
    vmin = np.nanmin(z_mesh)
    vmax = np.nanmax(z_mesh)

    im = ax.imshow(
        z_mesh,
        origin="lower",
        extent=(xg.min(), xg.max(), yg.min(), yg.max()),
        cmap="RdYlGn_r",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    mask = np.abs(position_values_pca[:, 2] - pc3_value) < slice_width

    scatter = ax.scatter(
        position_values_pca[mask, 0],
        position_values_pca[mask, 1],
        c=scores[mask],
        cmap="RdYlGn_r",
        edgecolor="k",
        s=30,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax, label="distance score")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PC3 = {pc3_value:.2f}")

    # Slider
    slider_ax = plt.axes((0.2, 0.05, 0.6, 0.03))
    slider = Slider(
        slider_ax,
        "PC3",
        pc3_min,
        pc3_max,
        valinit=pc3_value,
    )

    def update(val):
        pc3 = slider.val

        z_mesh = interpolate_slice(pc3)

        valid = np.isfinite(z_mesh)
        if not np.any(valid):
            return

        vmin = np.nanmin(z_mesh)
        vmax = np.nanmax(z_mesh)

        # Update image
        im.set_data(z_mesh)
        im.set_clim(vmin, vmax)
        cbar.update_normal(im)

        # Update scatter
        mask = np.abs(position_values_pca[:, 2] - pc3) < slice_width
        scatter.set_offsets(position_values_pca[mask, :2])
        scatter.set_array(scores[mask])
        scatter.set_clim(vmin, vmax)

        ax.set_title(f"PC3 = {pc3:.2f}")

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


def signal_quality(y):
    """
    Quantify how 'peak-like' (vs noise-like) a signal is, using lag-1
    autocorrelation. Real peaks are smooth -> neighboring points are
    highly correlated. Pure noise -> near-zero correlation.

    Returns
    -------
    quality : float in [0, 1]. 1 = smooth/clean peak, 0 = pure noise.
    """
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    if np.std(y) == 0 or len(y) < 2:
        return 0.0
    ac = np.corrcoef(y[:-1], y[1:])[0, 1]
    if np.isnan(ac):
        return 0.0
    return float(np.clip(ac, 0, 1))  # negative autocorr also treated as "not a peak"


def peak_overlap_score(
    x1, y1, x2, y2, method="overlap", n_points=1000, as_distance=True, noise_weight=10
):
    """
    method: "overlap" | "bhattacharyya" | "cosine"
    as_distance: if True, returns 0 = perfect match, 1 = no overlap (higher = worse)
                if False, returns 1 = perfect match, 0 = no overlap (higher = better)
    """
    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    x_min = min(x1.min(), x2.min())
    x_max = max(x1.max(), x2.max())
    x_common = np.linspace(x_min, x_max, n_points)

    y1_i = np.interp(x_common, x1, y1, left=0, right=0)
    y2_i = np.interp(x_common, x2, y2, left=0, right=0)

    y1_i = np.clip(y1_i, 0, None)
    y2_i = np.clip(y2_i, 0, None)

    a1 = np.trapezoid(y1_i, x_common)
    a2 = np.trapezoid(y2_i, x_common)
    if a1 == 0 or a2 == 0:
        return 1.0 if as_distance else 0.0
    p1 = y1_i / a1
    p2 = y2_i / a2

    if method == "overlap":
        similarity = np.trapezoid(np.minimum(p1, p2), x_common)
    elif method == "bhattacharyya":
        similarity = np.trapezoid(np.sqrt(p1 * p2), x_common)
    elif method == "cosine":
        similarity = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    else:
        raise ValueError("method must be 'overlap', 'bhattacharyya', or 'cosine'")

    distance = 1.0 - similarity
    # assess noise on the ORIGINAL (non-interpolated, non-normalized) signals
    q1 = signal_quality(y1)
    q2 = signal_quality(y2)
    combined_quality = min(q1, q2)  # worst-offender drives the penalty
    # inflate distance towards 1 as noise increases
    similarity = similarity * combined_quality

    distance = 1.0 - similarity
    return distance if as_distance else similarity


def add_info_box(
    ax: Axes,
    info: dict[str, Any],
    loc: str = "upper right",
    fontsize: float = 9,
    title: str | None = None,
):
    """
    Add a text box with key/value info to a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to annotate.
    info : Mapping[str, Any]
        Key/value pairs to display, one per line, formatted "key: value".
    loc : str
        One of "upper right", "upper left", "lower right", "lower left".
    fontsize : float
        Font size for the text.
    title : str, optional
        Optional bold title line at the top of the box.

    Returns
    -------
    matplotlib.text.Text
        The created text artist (in case you want to tweak it further).
    """
    # Build the text, right-padding keys so values line up
    key_width = max((len(str(k)) for k in info), default=0)
    lines = [f"{str(k):<{key_width}} : {v}" for k, v in info.items()]
    if title:
        lines.insert(0, title)
    text_str = "\n".join(lines)

    positions = {
        "upper right": {"x": 0.98, "y": 0.97, "ha": "right", "va": "top"},
        "upper left": {"x": 0.02, "y": 0.97, "ha": "left", "va": "top"},
        "lower right": {"x": 0.98, "y": 0.03, "ha": "right", "va": "bottom"},
        "lower left": {"x": 0.02, "y": 0.03, "ha": "left", "va": "bottom"},
    }
    if loc not in positions:
        raise ValueError(f"loc must be one of {list(positions)}, got {loc!r}")
    pos = positions[loc]

    return ax.text(
        pos["x"],
        pos["y"],
        text_str,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontfamily="monospace",
        horizontalalignment=pos["ha"],
        verticalalignment=pos["va"],
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.9,
        },
    )


def read_log_files(log_dir: str) -> pd.DataFrame:
    columns = [
        "FILE NAME",
        "SCAN",
        "START TIME",
        "XTRANS1",
        "XTRANS2",
        "Y1",
        "Y2",
        "Y3",
    ]

    log_data = pd.DataFrame(columns=columns)

    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            data = pd.read_csv(
                os.path.join(log_dir, filename),
                names=columns,
                skiprows=1,
                sep=r"\s+",
            )
            log_data = pd.concat([log_data, data], ignore_index=True)

    return log_data


def build_filepaths(comissioning_directory: str, log_data: pd.DataFrame) -> list[str]:

    filepaths = []

    for data_file in log_data["FILE NAME"].unique():
        data_file_path = os.path.join(comissioning_directory, str(data_file) + ".nxs")
        filepaths.append(data_file_path)

    return filepaths


# def split_into_modules_new(
#     filespaths: list[str],
#     modules: Collection[int] = tuple(range(28)),
#     bad_channels: Collection[int] = (),
#     out_dir: str = "/host-home/projects/outputs/mythen_calibration/processed",
# ):
#     """Read multiple nexus files and create one HDF5 file per module containing
#     all frames (concatenated across files).

#     Each output HDF5 will contain two datasets:
#       - "data": shape (N_total_frames, STRIPS_PER_MODULE)
#       - "delta": shape (N_total_frames,)

#     Returns list of output filepaths (one per module).
#     """
#     h5_files = {}
#     h5_datasets = {}

#     os.makedirs(out_dir, exist_ok=True)

#     def _create_module_file(mod, sample_dtype):
#         out_filepath = os.path.join(out_dir, f"module_{mod}_all.h5")
#         f = h5py.File(out_filepath, "w", libver="latest")
#         dset = f.create_dataset(
#             "data",
#             shape=(0, STRIPS_PER_MODULE),
#             maxshape=(None, STRIPS_PER_MODULE),
#             dtype=sample_dtype,
#             chunks=(1, STRIPS_PER_MODULE),
#             compression="gzip",
#         )
#         ddelta = f.create_dataset(
#             "delta",
#             shape=(0,),
#             maxshape=(None,),
#             dtype="f8",
#             chunks=(1,),
#             compression="gzip",
#         )
#         h5_files[mod] = f
#         h5_datasets[mod] = (dset, ddelta)
#         return out_filepath

#     out_filepaths = []

#     try:
#         for filepath in filespaths:
#             print(f"Reading file: {filepath}")
#             with h5py.File(filepath, "r") as file:
#                 entry = file["entry"]
#                 # read delta and data for this file in bulk
#                 delta = entry["mythen_nx"]["delta"][()]
#                 data_all = entry["mythen_nx"]["data"][:, :, DEFAULT_COUNTER][()]

#                 n_delta_points = int(delta.shape[0])

#                 # zero-out bad channels across all frames
#                 if len(bad_channels) > 0:
#                     data_all[:, bad_channels] = 0

#                 total_strips = data_all.shape[1]
#                 n_modules = len(modules)

#                 if total_strips == n_modules * STRIPS_PER_MODULE:
#                     # reshape to (n_delta, n_modules, STRIPS_PER_MODULE)
#                     data_by_module = data_all.reshape(
#                         (n_delta_points, n_modules, STRIPS_PER_MODULE)
#                     )
#                 else:
#                     # fallback: split along axis=1
#                     data_by_module = np.stack(
#                         [np.hsplit(data_all, n_modules)[m] for m in range(n_modules)],
#                         axis=1,
#                     )

#                 # for each module, append this file's frames in one operation
#                 for idx, n_mod in enumerate(modules):
#                     m = int(n_mod)
#                     mod_frames = data_by_module[:, idx, :]
#                     mod_deltas = np.array(delta, dtype="f8")

#                     if m not in h5_datasets:
#                         outp = _create_module_file(m, mod_frames.dtype)
#                         out_filepaths.append(outp)

#                     dset, ddelta = h5_datasets[m]
#                     old_n = dset.shape[0]
#                     add_n = mod_frames.shape[0]
#                     new_n = old_n + add_n
#                     dset.resize((new_n, STRIPS_PER_MODULE))
#                     dset[old_n:new_n, :] = mod_frames

#                     ddelta.resize((new_n,))
#                     ddelta[old_n:new_n] = mod_deltas

#     finally:
#         for f in h5_files.values():
#             try:
#                 f.close()
#             except Exception:
#                 pass

#     for path in out_filepaths:
#         print(f"Wrote: {path}")

#     return out_filepaths


# def read_module_file_and_compute_tth(
#     module_h5_path: str,
#     centre: float,
#     conv: float,
#     offset: float,
#     beamline_offset: float,
# ):
#     """Read a per-module HDF5 file created by split_into_modules_new and
#     compute raw_tth and per-frame two-theta arrays.

#     Returns a dict with keys:
#       - data: ndarray (N_frames, STRIPS_PER_MODULE)
#       - delta: ndarray (N_frames,)
#       - raw_tth: ndarray (STRIPS_PER_MODULE,) computed from module_pixel_number
#       - tth: ndarray (N_frames, STRIPS_PER_MODULE) == raw_tth + delta[:, None]
#     """
#     if not os.path.exists(module_h5_path):
#         raise FileNotFoundError(module_h5_path)

#     with h5py.File(module_h5_path, "r") as f:
#         if "data" not in f or "delta" not in f:
#             raise KeyError("Expected datasets 'data' and 'delta' in module file")

#         data = f["data"][()]
#         delta = f["delta"][()]

#     # pixel indices
#     module_pixel_number = np.arange(STRIPS_PER_MODULE, dtype=np.int64)

#     # compute raw tth once
#     raw_tth = I11Reduction.channel_to_angle(
#         module_pixel_number, centre, conv, offset, beamline_offset
#     )

#     # broadcast add delta to get per-frame tth
#     # ensure delta is shape (N,)
#     delta = np.asarray(delta, dtype=float)
#     tth = raw_tth[None, :] + delta[:, None]

#     return {"data": data, "delta": delta, "raw_tth": raw_tth, "tth": tth}


def get_peak(x, y):

    indexes = peakutils.indexes(y, thres=0.10, min_dist=100)

    peak_x = x[indexes]
    peak_y = y[indexes]
    return peak_x, peak_y


def trim(x, y, upper=11.36, lower=11.5):

    index = np.argwhere((lower < x) & (x < upper)).flatten()
    return x[index], y[index]


def get_best_params_from_pca(log_data: pd.DataFrame, param_cols: list[str]):

    param_vals = log_data[param_cols].values
    pca = PCA(n_components=3)
    param_vals_pca = pca.fit_transform(param_vals)

    scores = log_data["score"].to_numpy()

    n = 50

    xg = np.linspace(param_vals_pca[:, 0].min(), param_vals_pca[:, 0].max(), n)
    yg = np.linspace(param_vals_pca[:, 1].min(), param_vals_pca[:, 1].max(), n)
    zg = np.linspace(param_vals_pca[:, 2].min(), param_vals_pca[:, 2].max(), n)

    x_mesh, y_mesh, z_mesh = np.meshgrid(xg, yg, zg, indexing="ij")

    values = griddata(
        param_vals_pca,  # (N,3)
        scores,
        (x_mesh, y_mesh, z_mesh),
        method="linear",  # cubic is not supported in 3D
    )

    flat = values.ravel()

    valid = ~np.isnan(flat)

    best_flat = np.argsort(flat[valid])[:5]

    valid_indices = np.flatnonzero(valid)[best_flat]

    i, j, k = np.unravel_index(valid_indices, values.shape)

    best_pca = np.column_stack(
        (
            xg[i],
            yg[j],
            zg[k],
        )
    )

    print(best_pca)

    loadings = pd.DataFrame(
        pca.components_.T, index=param_cols, columns=["PC1", "PC2", "PC3"]
    )
    print(loadings)

    # reconstruct in standardized space, assuming other PCs = 0
    reconstructed_scaled = pca.inverse_transform(best_pca)

    # undo the StandardScaler to get back to real parameter units
    scaler = StandardScaler().fit(param_vals)  # refit or reuse the one from before
    reconstructed_params = scaler.inverse_transform(reconstructed_scaled)

    result = pd.DataFrame(reconstructed_params, columns=param_cols)
    print(result)

    print(log_data)


def other_plot():
    pass
    # print(log_data)

    # fig = px.parallel_coordinates(
    #     log_data,
    #     dimensions=["XTRANS1", "XTRANS2", "Y1", "Y2", "Y3"],
    #     color="score",
    #     color_continuous_scale=px.colors.sequential.Viridis,
    #     labels={"score": "distance"},
    # )
    # fig.write_html("output.html")

    # import seaborn as sns

    # sns.pairplot(
    #     log_data,
    #     vars=param_cols,
    #     hue="score",
    #     palette="viridis",
    #     diag_kind="kde",
    # )
    # plt.show()
    # plt.close()


def plot_peak_overlap_results(
    log_data: pd.DataFrame, filenumber: int, modules: list[int], good_score: float = 7
):

    title_data = log_data[log_data["FILE NAME"] == filenumber]

    assert title_data["FILE NAME"].to_numpy()[0] == filenumber

    info = {
        "XTRANS1": round(title_data["XTRANS1"].to_numpy()[0], 3),
        "XTRANS2": round(title_data["XTRANS2"].to_numpy()[0], 3),
        "Y1": round(title_data["Y1"].to_numpy()[0], 3),
        "Y2": round(title_data["Y2"].to_numpy()[0], 3),
        "Y3": round(title_data["Y3"].to_numpy()[0], 3),
        "score": f"{title_data['score'].to_numpy()[0]:.2f}",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    plt.title(f"File {filenumber}")
    add_info_box(ax, info, loc="upper right", fontsize=10, title="Scan Info")

    # pixel_range = np.arange(0, len(counts1_trim), 1)

    plt.plot(tth1_trim, counts1_trim, label=f"Module {modules[0]}")
    plt.plot(tth2_trim, counts2_trim, label=f"Module {modules[1]}")

    plt.vlines(
        calibrant_peak,
        np.amin(counts1_trim),
        np.max(counts1_trim),
        color="red",
        linestyle="--",
        label="Calibrant Peaks",
    )
    plt.legend()

    save_path = (
        (
            f"{comissioning_directory}/{modules}/output_{filenumber}_modules={modules}.png"
        )
        .replace(" ", "")
        .replace(",", "_")
    )

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)

    if score < good_score:
        plt.savefig(
            f"{comissioning_directory}/best/output_{filenumber}_modules={modules}.png"
        )

    # plt.show()
    plt.close()


def find_frame_with_those_modules(
    angular_calibration: AngularCalibration2D,
    module: int,
    deltas: np.ndarray,
    observed_reflections_in_tth: np.ndarray,
    tol=0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    deltas_positions = []
    frames = []
    peaks = []

    for frame, delta in enumerate(deltas):
        delta = float(delta)

        module_tth = angular_calibration.return_module_tth(module=module, delta=delta)
        paired_module_tth = angular_calibration.return_module_tth(
            module=find_pair(module), delta=delta
        )

        modisbetween = (module_tth.min() - tol <= observed_reflections_in_tth) & (
            observed_reflections_in_tth <= module_tth.max() + tol
        )

        pairedisbetween = (
            paired_module_tth.min() - tol <= observed_reflections_in_tth
        ) & (observed_reflections_in_tth <= paired_module_tth.max() + tol)

        good = modisbetween * pairedisbetween

        peaks_in_both_modules = observed_reflections_in_tth[good]

        if good.any() and (len(peaks_in_both_modules) == 1):
            print(
                module,
                find_pair(module),
                f"{delta=} {frame=} {peaks_in_both_modules}",
            )

            peaks.append(peaks_in_both_modules[0])
            frames.append(frame)
            deltas_positions.append(delta)

    peaks = np.array(peaks)
    frames = np.array(frames)
    deltas_positions = np.array(deltas_positions)

    return peaks, frames, deltas_positions


def find_best_frame_and_modules_for_peak(
    angular_calibration: AngularCalibration2D,
    deltas: np.ndarray,
    observed_reflections_in_tth: np.ndarray,
    peak: float = 16.21,
):

    print(deltas)

    for module_pair in paired_modules():
        for frame, delta in enumerate(deltas):
            module1_tth = angular_calibration.return_module_tth(
                module=module_pair[0], delta=delta
            )

            mod1isbetween = module1_tth.min() <= peak <= module1_tth.max()

            module2_tth = angular_calibration.return_module_tth(
                module=module_pair[1], delta=delta
            )

            mod2isbetween = module2_tth.min() <= peak <= module2_tth.max()

            good_frame = mod1isbetween * mod2isbetween

            if good_frame:
                print(
                    module_pair,
                    frame,
                    peak,
                    good_frame,
                )

    return [None]


def generate_score():

    wavelength_in_ang = 0.828828
    calibrant_name = "LaB6"

    calibrant = get_calibrant(calibrant_name)
    calibrant.wavelength = wavelength_in_ang / 1e10
    observed_reflections_in_tth = np.array(calibrant.get_peaks("2th_deg"), dtype=float)

    print(f"{observed_reflections_in_tth=}")
    # quit()

    # bad_channels = load_int_array_from_file(
    #     "/workspaces/XRPD-Toolbox/config/i11/bad_channels.txt"
    # )

    # comissioning_directory = "/dls/i11/data/2026/cm44155-2"
    comissioning_directory = "/scratch/translate_mythen/cm44155-2"

    log_dir = "/workspaces/PSD_alignment_Logs"
    log_data = read_log_files(log_dir)
    filepaths = build_filepaths(comissioning_directory, log_data)

    # print(make_steps(log_data, param_cols))

    # modules_list = [11, 13]
    modules_list = list(range(14))

    all_modules = np.array([[m, find_pair(m)] for m in modules_list])

    if len(np.unique(all_modules.flatten())) != (len(modules_list) * 2):
        raise ValueError(
            f"Modules list contains duplicates or invalid modules: {all_modules}"
        )

    # add nans and create score columns
    for m in modules_list:
        log_data[f"score_{m}_{find_pair(m)}"] = np.nan

    print(log_data)

    x1 = log_data["XTRANS1"].to_numpy()
    x2 = log_data["XTRANS2"].to_numpy()
    y1 = log_data["Y1"].to_numpy()
    y2 = log_data["Y2"].to_numpy()
    y3 = log_data["Y3"].to_numpy()

    pose = base_to_euler(jack_y1=y1, jack_y2=y2, jack_y3=y3, plate_x1=x1, plate_x2=x2)

    log_data["euler_x"] = pose["euler_angles_deg"][:, 0]
    log_data["euler_y"] = pose["euler_angles_deg"][:, 1]
    log_data["euler_z"] = pose["euler_angles_deg"][:, 2]

    for filepath in filepaths:
        print(f"Processing file: {filepath}")

        basename = os.path.basename(filepath)
        filenumber = int(basename.replace(".nxs", ""))

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            quit()

        # cal = AngularCalibrateMythen(
        #     filepath=filepath,
        #     wavelength_in_ang=wavelength_in_ang,
        #     calibrant_name=calibrant_name,
        #     bad_modules=[17, 27],
        # )

        # cal.get_selected_peaks(mask_type=None, use_pickle=False, plot_fit=True)

        settings = MythenSettings()
        angular_calibration = AngularCalibration2D.load_from_json(
            "/scratch/translate_mythen/ang_cal.json"
        )

        # # for data_file in check_files:

        # slide_hash = {(13, 14): slice(4, 5), (11, 16): slice(1, 2)}

        for _, module in enumerate(modules_list):
            deltas = h5_to_array(filepath, "/entry/mythen_nx/delta")

            # frames = find_best_frame_and_modules_for_peak(
            #     angular_calibration, deltas, observed_reflections_in_tth
            # )

            peaks, frames, deltas_positions = find_frame_with_those_modules(
                angular_calibration=angular_calibration,
                module=module,
                deltas=deltas,
                observed_reflections_in_tth=observed_reflections_in_tth,
            )

            analysis = MythenDetector(
                filepath=filepath,
                settings=settings,
                angular_calibration=angular_calibration,
                frames=[
                    frames[0]
                ],  # DO NOT FORGET THIS - I have only selected first peak
            )

            print("See above")

            module_score_col = f"score_{module}_{find_pair(module)}"

            module_data = analysis.get_diffraction_for_seperate_modules(
                modules=[module, find_pair(module)]
            )

            tth1, counts1 = module_data[module]
            tth2, counts2 = module_data[find_pair(module)]

            # if module in [0, 1, 3]:
            #     plt.plot(tth1, counts1, label=f"Module {module}")
            #     plt.plot(tth2, counts2, label=f"Module {find_pair(module)}")
            #     plt.legend()
            #     plt.show()

            calibrant_peaks_index = np.isclose(
                observed_reflections_in_tth,
                np.mean(tth1),
                atol=1,  # type: ignore
            )

            print(observed_reflections_in_tth[calibrant_peaks_index])

            try:
                calibrant_peak = float(
                    (observed_reflections_in_tth[calibrant_peaks_index])[0]
                )

                maxtth_for_both = min(tth1.max(), tth2.max())
                mmintth_for_both = max(tth1.min(), tth2.min())

                calibrant_upper = calibrant_peak + 0.2
                calibrant_lower = calibrant_peak - 0.2

                upper = min(calibrant_upper, maxtth_for_both)
                lower = max(calibrant_lower, mmintth_for_both)

                # print(upper, lower)

                tth1_trim, counts1_trim = trim(tth1, counts1, upper=upper, lower=lower)
                tth2_trim, counts2_trim = trim(tth2, counts2, upper=upper, lower=lower)

                if (len(tth1_trim) < 20) or (len(tth2_trim) < 20):
                    print(f"NO DATA for {module}")
                    continue

            except Exception as _:
                print("No calibrant peak in data!")
                continue

            # plt.plot(tth1_trim, counts1_trim, label=f"Module {module}")
            # plt.plot(tth2_trim, counts2_trim, label=f"Module {find_pair(module)}")
            # plt.legend()
            # plt.show()

            module1_peak_x, module1_peak_y = get_peak(
                tth1_trim, np.array(counts1_trim, dtype=float)
            )
            module2_peak_x, module2_peak_y = get_peak(
                tth2_trim, np.array(counts2_trim, dtype=float)
            )

            peak_diff1 = np.abs(calibrant_peak - module1_peak_x[0])
            peak_diff2 = np.abs(calibrant_peak - module2_peak_x[0])

            peak_error = (peak_diff1 * peak_diff2) + 1

            try:
                score = (
                    peak_overlap_score(
                        tth1_trim,
                        counts1_trim,
                        tth2_trim,
                        counts2_trim,
                        method="overlap",
                    )
                    * 100
                    # * peak_error
                )

            except Exception as e:
                print(f"Error processing file {filepath}: {e}")
                score = np.nan

            # add score to row of filenumber
            log_data.loc[log_data["FILE NAME"] == filenumber, module_score_col] = score
            score_cols = [f"score_{m}_{find_pair(m)}" for m in modules_list]
            log_data["score"] = log_data[score_cols].sum(axis=1)  # or .sum/.prod

            plot_peak_overlap_results(
                log_data=log_data,
                filenumber=filenumber,
                good_score=7,
                modules=[module, find_pair(module)],
            )
    log_data = log_data.astype(float, errors="ignore")
    log_data.dropna(axis=1, how="all")
    log_data.to_csv(f"{comissioning_directory}/scores.csv", index=False)

    return log_data


def get_data(score_filepath: str, reload: bool = True) -> pd.DataFrame:

    if os.path.exists(score_filepath) and not reload:
        score_dataframe = pd.read_csv(score_filepath)

    else:
        score_dataframe = generate_score()

    return score_dataframe


if __name__ == "__main__":
    # score_cols = [f"score_{m}_{find_pair(m)}" for m in modules_list]
    # log_data["score"] = log_data[score_cols].sum(axis=1)  # or .sum/.prod

    comissioning_directory = "/scratch/translate_mythen/cm44155-2"
    score_filepath = f"{comissioning_directory}/scores.csv"
    param_cols = ["XTRANS1", "XTRANS2", "Y1", "Y2", "Y3"]

    log_data = get_data(score_filepath, reload=False)

    print(log_data)

    euler_x = log_data["euler_x"].to_numpy()
    euler_y = log_data["euler_y"].to_numpy()
    euler_z = log_data["euler_z"].to_numpy()
    score = log_data["score"].to_numpy()

    fig = spherical_score_plot_2d(euler_x, euler_x, score, best="min")
    fig.savefig(f"/{comissioning_directory}/stereographic_projection.png")
    print("saved demo plot")

    # plot_static_pca()
    get_best_params_from_pca(log_data=log_data, param_cols=param_cols)
    plot_pca(log_data=log_data, param_cols=param_cols)
