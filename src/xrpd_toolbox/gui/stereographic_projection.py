from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d


def spherical_score_plot_3d(
    euler_x: np.ndarray,
    euler_y: np.ndarray,
    score: np.ndarray,
    best: Literal["max", "min"] = "min",
    grid_bins: int = 60,
    interpolation: Literal["linear", "cubic", "nearest"] = "linear",
    angle_unit: Literal["deg", "rad"] = "deg",
    colormap: str = "Inferno",
) -> go.Figure:
    """Interactive 3D sphere heatmap of best score per tilt direction, with X/Y/Z axes.

    euler_x, euler_y are treated as a small tilt (like pitch/roll) away from the
    pole (north pole = untilted state). Duplicate/nearby samples at the same
    direction are collapsed to the best score before interpolation onto the sphere.
    """
    euler_x = np.asarray(euler_x, dtype=float)
    euler_y = np.asarray(euler_y, dtype=float)
    score = np.asarray(score, dtype=float)

    euler_x_rad = np.radians(euler_x) if angle_unit == "deg" else euler_x
    euler_y_rad = np.radians(euler_y) if angle_unit == "deg" else euler_y

    # Tilt magnitude (polar angle from north pole) and direction (azimuth)
    polar_angle_rad = np.hypot(euler_x_rad, euler_y_rad)
    azimuth_rad = np.arctan2(euler_y_rad, euler_x_rad)

    # Collapse duplicate/nearby points in (polar, azimuth) space, keeping best score
    polar_edges = np.linspace(0, polar_angle_rad.max() * 1.02, grid_bins + 1)
    azimuth_edges = np.linspace(-np.pi, np.pi, grid_bins + 1)
    binned_best_score, polar_e, azimuth_e, _ = binned_statistic_2d(
        polar_angle_rad,
        azimuth_rad,
        score,
        statistic=best,
        bins=[polar_edges, azimuth_edges],  # type: ignore
    )
    polar_centers = (polar_e[:-1] + polar_e[1:]) / 2.0
    azimuth_centers = (azimuth_e[:-1] + azimuth_e[1:]) / 2.0
    polar_grid, azimuth_grid = np.meshgrid(
        polar_centers, azimuth_centers, indexing="ij"
    )
    occupied = ~np.isnan(binned_best_score)

    # Interpolate onto a fine, regular (polar, azimuth) grid for a smooth surface
    fine_polar = np.linspace(0, polar_angle_rad.max() * 1.02, 150)
    fine_azimuth = np.linspace(-np.pi, np.pi, 150)
    fine_polar_grid, fine_azimuth_grid = np.meshgrid(
        fine_polar, fine_azimuth, indexing="ij"
    )
    smooth_score = griddata(
        points=(polar_grid[occupied], azimuth_grid[occupied]),
        values=binned_best_score[occupied],
        xi=(fine_polar_grid, fine_azimuth_grid),
        method=interpolation,
    )

    # Spherical -> Cartesian (north pole = untilted state, on +Z)
    sphere_x = np.sin(fine_polar_grid) * np.cos(fine_azimuth_grid)
    sphere_y = np.sin(fine_polar_grid) * np.sin(fine_azimuth_grid)
    sphere_z = np.cos(fine_polar_grid)

    surface = go.Surface(
        x=sphere_x,
        y=sphere_y,
        z=sphere_z,
        surfacecolor=smooth_score,
        colorscale=colormap,
        colorbar={"title": "score"},
        showscale=True,
    )

    # Full translucent reference sphere, so the (often small) data patch reads
    # as sitting on a globe rather than floating as a flat disc.
    ref_polar = np.linspace(0, np.pi, 60)
    ref_azimuth = np.linspace(-np.pi, np.pi, 60)
    ref_polar_grid, ref_azimuth_grid = np.meshgrid(
        ref_polar, ref_azimuth, indexing="ij"
    )
    reference_sphere = go.Surface(
        x=np.sin(ref_polar_grid) * np.cos(ref_azimuth_grid),
        y=np.sin(ref_polar_grid) * np.sin(ref_azimuth_grid),
        z=np.cos(ref_polar_grid),
        surfacecolor=np.zeros_like(ref_polar_grid),
        colorscale=[[0, "lightgray"], [1, "lightgray"]],
        opacity=0.25,
        showscale=False,
        hoverinfo="skip",
    )

    # Latitude/longitude graticule lines for visual scale/orientation
    graticule_traces = []
    for lat_deg in np.arange(0, 181, 20):
        lat = np.radians(lat_deg)
        az = np.linspace(-np.pi, np.pi, 100)
        graticule_traces.append(
            go.Scatter3d(
                x=np.sin(lat) * np.cos(az),
                y=np.sin(lat) * np.sin(az),
                z=np.full_like(az, np.cos(lat)),
                mode="lines",
                line={"color": "gray", "width": 1},
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for lon_deg in np.arange(-180, 180, 20):
        lon = np.radians(lon_deg)
        lat = np.linspace(0, np.pi, 100)
        graticule_traces.append(
            go.Scatter3d(
                x=np.sin(lat) * np.cos(lon),
                y=np.sin(lat) * np.sin(lon),
                z=np.cos(lat),
                mode="lines",
                line={"color": "gray", "width": 1},
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Raw sample points for reference, drawn on the sphere surface
    raw_x = np.sin(polar_angle_rad) * np.cos(azimuth_rad)
    raw_y = np.sin(polar_angle_rad) * np.sin(azimuth_rad)
    raw_z = np.cos(polar_angle_rad)
    raw_points = go.Scatter3d(
        x=raw_x,
        y=raw_y,
        z=raw_z,
        mode="markers",
        marker={"size": 1.5, "color": "white", "opacity": 0.3},
        showlegend=False,
    )

    # Cartesian X/Y/Z axes through the sphere, with labels
    axis_length = 1.4
    axis_traces = []
    axis_specs = [
        ("X", (1, 0, 0), "red"),
        ("Y", (0, 1, 0), "green"),
        ("Z", (0, 0, 1), "blue"),
    ]
    for axis_label, (ax, ay, az), axis_color in axis_specs:
        axis_traces.append(
            go.Scatter3d(
                x=[-axis_length * ax, axis_length * ax],
                y=[-axis_length * ay, axis_length * ay],
                z=[-axis_length * az, axis_length * az],
                mode="lines+text",
                line={"color": axis_color, "width": 4},
                text=["", axis_label],
                textposition="top center",
                showlegend=False,
            )
        )

    figure = go.Figure(
        data=[reference_sphere, *graticule_traces, surface, raw_points, *axis_traces]
    )
    figure.update_layout(
        title=f"Spherical projection - {best} score per tilt direction",
        scene={
            "xaxis": {"range": [-axis_length, axis_length], "visible": False},
            "yaxis": {"range": [-axis_length, axis_length], "visible": False},
            "zaxis": {"range": [-axis_length, axis_length], "visible": False},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return figure


def spherical_score_plot_2d(
    euler_x: np.ndarray,
    euler_y: np.ndarray,
    score: np.ndarray,
    best: Literal["max", "min"] = "max",
    grid_bins: int = 60,
    interpolation: Literal["linear", "cubic", "nearest"] = "linear",
    angle_unit: Literal["deg", "rad"] = "deg",
    colormap: str = "inferno",
    color_range: tuple[float, float] | None = None,
    log_scale: bool = False,
) -> Figure:
    """Stereographic (equal-angle) projection heatmap of best score per tilt direction.

    euler_x, euler_y are treated as a small tilt (like pitch/roll) away from the
    pole; euler_x/euler_y magnitude sets the polar angle, their ratio sets azimuth.
    Duplicate/nearby samples at the same direction are collapsed to the best score.
    Binning and interpolation happen in (polar angle, azimuth) space - the same
    approach as spherical_score_plot - so the two stay directly comparable; the
    result is only projected to a flat disc for display at the very end.
    """
    euler_x = np.asarray(euler_x, dtype=float)
    euler_y = np.asarray(euler_y, dtype=float)
    score = np.asarray(score, dtype=float)

    valid = np.isfinite(euler_x) & np.isfinite(euler_y) & np.isfinite(score)
    n_dropped = len(score) - valid.sum()
    if n_dropped:
        print(
            f"spherical_score_plot_2d: dropping {n_dropped} row(s) with NaN/inf values"
        )
    euler_x, euler_y, score = euler_x[valid], euler_y[valid], score[valid]

    if euler_x.size < 4:
        raise ValueError(
            f"spherical_score_plot_2d needs at least 4 valid (non-NaN) samples to "
            f"build a surface, got {euler_x.size}. Check that euler_x/euler_y/score "
            f"were pulled from the right columns and aren't empty or all-NaN."
        )
    if np.allclose(euler_x, euler_y):
        raise ValueError(
            "euler_x and euler_y are identical (or near-identical) - every point falls "
            "on a single diagonal line, so a 2D surface can't be interpolated from them"
            "This is usually a copy-paste bug (e.g. passing the same column/array twice"
            "instead of euler_x and euler_y)."
        )
    if log_scale and score.min() <= 0:
        raise ValueError(
            f"log_scale=True requires all scores to be strictly positive, but the "
            f"minimum score is {score.min()}. Log scale is undefined for zero/negative "
            f"values."
        )

    euler_x_rad = np.radians(euler_x) if angle_unit == "deg" else euler_x
    euler_y_rad = np.radians(euler_y) if angle_unit == "deg" else euler_y

    # Tilt magnitude (polar angle from north pole) and direction (azimuth)
    polar_angle_rad = np.hypot(euler_x_rad, euler_y_rad)
    azimuth_rad = np.arctan2(euler_y_rad, euler_x_rad)

    # Collapse duplicate/nearby points in (polar, azimuth) space, keeping best score
    polar_edges = np.linspace(0, polar_angle_rad.max() * 1.02, grid_bins + 1)
    azimuth_edges = np.linspace(-np.pi, np.pi, grid_bins + 1)
    binned_best_score, polar_e, azimuth_e, _ = binned_statistic_2d(
        polar_angle_rad,
        azimuth_rad,
        score,
        statistic=best,
        bins=[polar_edges, azimuth_edges],  # type: ignore[arg-type]  # scipy stub only declares int
    )
    polar_centers = (polar_e[:-1] + polar_e[1:]) / 2.0
    azimuth_centers = (azimuth_e[:-1] + azimuth_e[1:]) / 2.0
    polar_grid, azimuth_grid = np.meshgrid(
        polar_centers, azimuth_centers, indexing="ij"
    )
    occupied = ~np.isnan(binned_best_score)

    # Interpolate onto a fine, regular (polar, azimuth) grid for a smooth surface
    fine_polar = np.linspace(0, polar_angle_rad.max() * 1.02, 300)
    fine_azimuth = np.linspace(-np.pi, np.pi, 300)
    fine_polar_grid, fine_azimuth_grid = np.meshgrid(
        fine_polar, fine_azimuth, indexing="ij"
    )
    occupied_polar = polar_grid[occupied]
    occupied_azimuth = azimuth_grid[occupied]
    occupied_score = binned_best_score[occupied]

    # Azimuth is periodic (-pi and +pi are the same direction), but griddata treats
    # it as a flat, non-periodic domain and leaves a seam/gap near +-pi. Pad with
    # ghost copies shifted by +-2pi so interpolation sees continuity across the seam.
    wrapped_polar = np.concatenate([occupied_polar] * 3)
    wrapped_azimuth = np.concatenate(
        [occupied_azimuth - 2 * np.pi, occupied_azimuth, occupied_azimuth + 2 * np.pi]
    )
    wrapped_score = np.concatenate([occupied_score] * 3)

    try:
        smooth_score = griddata(
            points=(wrapped_polar, wrapped_azimuth),
            values=wrapped_score,
            xi=(fine_polar_grid, fine_azimuth_grid),
            method=interpolation,
        )
    except Exception as error:
        raise ValueError(
            "Could not build a surface from euler_x/euler_y - the points are likely "
            "collinear or otherwise degenerate (e.g. one axis has near-zero spread, or "
            "the same data was passed for both axes)."
        ) from error

    # Project the smoothed (polar, azimuth) grid to the flat disc for display only.
    # Plotted on a native polar axis (not pre-converted to Cartesian x/y) because
    # a ring of constant polar angle traces a full circle - non-monotonic in x or y -
    # which breaks pcolormesh's Cartesian cell-edge inference and causes seam artifacts.
    fine_radius_grid = np.tan(fine_polar_grid / 2.0)
    smooth_score = np.ma.masked_where(np.isnan(smooth_score), smooth_score)

    disc_extent = fine_radius_grid.max()
    # Default the color axis to the real score range rather than auto-scaling from
    # the interpolated surface, whose cubic spline can overshoot past the real data
    # near sparse/noisy edges. This only affects display (out-of-range pixels are
    # clamped to the colormap's end colors) - the returned smooth_score array is
    # left untouched.
    color_vmin, color_vmax = (
        color_range
        if color_range is not None
        else (float(score.min()), float(score.max()))
    )
    if log_scale and color_vmin <= 0:
        raise ValueError(
            f"log_scale=True requires a positive color_range, got vmin={color_vmin}."
        )
    color_norm = (
        LogNorm(vmin=color_vmin, vmax=color_vmax)
        if log_scale
        else Normalize(vmin=color_vmin, vmax=color_vmax)
    )

    fig = plt.figure(figsize=(7, 6))
    ax = cast(PolarAxes, fig.add_subplot(111, projection="polar"))
    heatmap = ax.pcolormesh(
        fine_azimuth_grid,
        fine_radius_grid,
        smooth_score,
        cmap=colormap,
        shading="auto",
        norm=color_norm,
    )
    projected_radius = np.tan(polar_angle_rad / 2.0)
    ax.scatter(azimuth_rad, projected_radius, s=4, c="white", alpha=0.25, linewidths=0)

    # Radial ticks labelled by polar angle in degrees, spokes every 30 degrees
    max_polar_deg = np.degrees(2 * np.arctan(disc_extent))
    tick_polar_deg = np.arange(15, max_polar_deg, 15)
    ax.set_rticks(np.tan(np.radians(tick_polar_deg) / 2.0))
    ax.set_yticklabels([f"{d:.0f}°" for d in tick_polar_deg])
    ax.set_thetagrids(np.arange(0, 360, 30))
    ax.set_rlim(0, disc_extent)
    ax.grid(color="gray", alpha=0.4, linewidth=0.5)
    ax.set_title(f"Stereographic projection - {best} score per tilt direction", pad=20)

    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("score (log scale)" if log_scale else "score")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # --- synthetic demo data: a noisy score peak, now over a much wider tilt range ---
    rng = np.random.default_rng(0)
    n_samples = 6000
    demo_euler_x = rng.uniform(-70, 70, n_samples)  # degrees
    demo_euler_y = rng.uniform(-70, 70, n_samples)  # degrees
    true_peak_x, true_peak_y = 25.0, -15.0
    demo_score = 100 * np.exp(
        -((demo_euler_x - true_peak_x) ** 2 + (demo_euler_y - true_peak_y) ** 2) / 800.0
    ) + rng.normal(0, 5, n_samples)

    fig = spherical_score_plot_3d(demo_euler_x, demo_euler_y, demo_score, best="max")
    fig.write_html("test.html")
    print("saved demo plot")

    figure = spherical_score_plot_2d(demo_euler_x, demo_euler_y, demo_score, best="max")
