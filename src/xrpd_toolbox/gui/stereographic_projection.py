from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d


def spherical_score_plot_3d(
    euler_x: np.ndarray,
    euler_y: np.ndarray,
    score: np.ndarray,
    best: Literal["max", "min"] = "min",
    grid_bins: int = 60,
    interpolation: Literal["linear", "cubic", "nearest"] = "cubic",
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
    best: Literal["max", "min"] = "min",
    grid_bins: int = 60,
    interpolation: Literal["linear", "cubic", "nearest"] = "cubic",
    angle_unit: Literal["deg", "rad"] = "deg",
    colormap: str = "inferno",
) -> Figure:
    """Stereographic (equal-angle) projection heatmap of best score per tilt direction.

    euler_x, euler_y are treated as a small tilt (like pitch/roll) away from the
    pole; euler_x/euler_y magnitude sets the polar angle, their ratio sets azimuth.
    Duplicate/nearby samples at the same direction are collapsed to the best score.
    """
    euler_x = np.asarray(euler_x, dtype=float)
    euler_y = np.asarray(euler_y, dtype=float)
    score = np.asarray(score, dtype=float)

    valid = np.isfinite(euler_x) & np.isfinite(euler_y) & np.isfinite(score)
    n_dropped = len(score) - valid.sum()
    if n_dropped:
        print(
            f"stereographic_score_heatmap: dropping {n_dropped} row with NaN/inf values"
        )
    euler_x, euler_y, score = euler_x[valid], euler_y[valid], score[valid]

    if euler_x.size < 4:
        raise ValueError(
            f"stereographic_score_heatmap needs at least 4 valid (non-NaN) samples to "
            f"build a surface, got {euler_x.size}. Check that euler_x/euler_y/score "
            f"were pulled from the right columns and aren't empty or all-NaN."
        )

    if angle_unit == "deg":
        euler_x_rad = np.radians(euler_x)
        euler_y_rad = np.radians(euler_y)
    else:
        euler_x_rad = euler_x
        euler_y_rad = euler_y

    # Tilt magnitude (polar angle) and direction (azimuth) from the pole
    polar_angle_rad = np.hypot(euler_x_rad, euler_y_rad)
    azimuth_rad = np.arctan2(euler_y_rad, euler_x_rad)

    # Equal-angle (Wulff) stereographic projection onto the unit disc
    projected_radius = np.tan(polar_angle_rad / 2.0)
    projected_x = projected_radius * np.cos(azimuth_rad)
    projected_y = projected_radius * np.sin(azimuth_rad)

    # Collapse duplicate/nearby points onto a fine grid, keeping the best score per cell
    disc_extent = np.max(projected_radius) * 1.05
    grid_edges = np.linspace(-disc_extent, disc_extent, grid_bins + 1)
    binned_best_score, x_edges, y_edges, _ = binned_statistic_2d(
        projected_x,
        projected_y,
        score,
        statistic=best,
        bins=[grid_edges, grid_edges],  # type: ignore
    )
    bin_centers_x = (x_edges[:-1] + x_edges[1:]) / 2.0
    bin_centers_y = (y_edges[:-1] + y_edges[1:]) / 2.0
    grid_x, grid_y = np.meshgrid(bin_centers_x, bin_centers_y, indexing="ij")
    occupied = ~np.isnan(binned_best_score)

    # Interpolate the sparse best-score points onto a smooth fine grid for the heatmap
    fine_axis = np.linspace(-disc_extent, disc_extent, 300)
    fine_grid_x, fine_grid_y = np.meshgrid(fine_axis, fine_axis)
    smooth_score = griddata(
        points=(grid_x[occupied], grid_y[occupied]),
        values=binned_best_score[occupied],
        xi=(fine_grid_x, fine_grid_y),
        method=interpolation,
    )

    # Mask outside the projection disc and outside the convex hull of real data
    outside_disc = fine_grid_x**2 + fine_grid_y**2 > disc_extent**2
    smooth_score = np.ma.masked_where(
        outside_disc | np.isnan(smooth_score), smooth_score
    )

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"aspect": "equal"})
    heatmap = ax.pcolormesh(
        fine_grid_x, fine_grid_y, smooth_score, cmap=colormap, shading="auto"
    )
    ax.scatter(projected_x, projected_y, s=4, c="white", alpha=0.25, linewidths=0)

    # Stereonet-style reference circles/spokes, labelled in the polar angle unit
    max_polar_deg = np.degrees(2 * np.arctan(disc_extent))
    for polar_deg in np.arange(15, max_polar_deg, 15):
        radius = np.tan(np.radians(polar_deg) / 2.0)
        ax.add_patch(
            Circle((0, 0), radius, fill=False, color="gray", linewidth=0.5, alpha=0.6)
        )
    for azimuth_deg in np.arange(0, 360, 30):
        azimuth = np.radians(azimuth_deg)
        ax.plot(
            [0, disc_extent * np.cos(azimuth)],
            [0, disc_extent * np.sin(azimuth)],
            color="gray",
            linewidth=0.5,
            alpha=0.6,
        )
    ax.add_patch(Circle((0, 0), disc_extent, fill=False, color="black", linewidth=1.2))

    ax.set_xlim(-disc_extent, disc_extent)
    ax.set_ylim(-disc_extent, disc_extent)
    ax.set_xlabel("euler_x tilt direction")
    ax.set_ylabel("euler_y tilt direction")
    ax.set_title(f"Stereographic projection - {best} score per tilt direction")
    ax.set_xticks([])
    ax.set_yticks([])

    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("score")

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
