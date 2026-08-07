"""Interactive Debye-Scherrer cone / curved dual-strip detector
alignment simulator. Shows, live, how pitch/yaw/roll/translation
misalignment distorts each strip's diffraction pattern relative to
the correct pyFAI-calibrant peak positions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.widgets import Button, Slider, TextBox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from pyFAI.calibrant import CALIBRANT_FACTORY, get_calibrant  # noqa: E402

# Geometry notes
# --------------
# - Sample sits at the world origin; the beam travels along +x.
# - Each strip is an arc of radius `radius_mm`, sweeping polar angle
#   `alpha_deg` from the beam axis at a fixed azimuth (a "2-theta
#   arm"), like a curved position-sensitive detector on a Rowland
#   circle centered on the sample.
# - The two strips are parallel: strip two is strip one translated
#   by a fixed vector of length `gap_mm`, not offset in azimuth
#   (which would make them converge/diverge like meridians).
# - At the nominal pose the central design arc sits exactly on the
#   radius_mm-sphere around the sample, so alpha equals the true
#   scattering angle and each Debye-Scherrer cone crosses the arc at
#   exactly one point. Pitch/yaw/roll/translation carry the rigid
#   strip pair away from that sphere, which is what breaks the
#   one-point-per-ring property and shows up as misalignment.
# - pitch rotates about the detector's local z axis, yaw about its
#   local y axis, roll about its local x (beam) axis. Rotations are
#   applied about the detector's own mechanical vertex, then the
#   `distance`/ty/tz translation is applied.


def calibrant_reflections(
    calibrant_name: str,
    wavelength_a: float,
    two_theta_max: float = 175.0,
    min_rel_intensity: float = 0.5,
) -> list[dict[str, float]]:
    """Look up a pyFAI calibrant and compute its line positions.

    Positions come from pyFAI's tabulated d-spacings; pyFAI has no
    intensities, so relative intensity here is a Lorentz-polarization
    estimate for display only, not physically exact."""
    if calibrant_name not in CALIBRANT_FACTORY:
        names = ", ".join(sorted(CALIBRANT_FACTORY.keys()))
        raise KeyError(f"Unknown calibrant '{calibrant_name}'. Available: {names}")

    calibrant = get_calibrant(calibrant_name)
    calibrant.wavelength = wavelength_a * 1e-10
    two_theta_rad = np.asarray(calibrant.get_2th(), dtype=float)

    reflections = []
    for two_theta_r in two_theta_rad:
        two_theta_deg = np.degrees(two_theta_r)
        if two_theta_deg < 1.0 or two_theta_deg > two_theta_max:
            continue
        theta_r = two_theta_r / 2.0
        d_spacing = wavelength_a / (2.0 * np.sin(theta_r))
        # Lorentz-polarization factor as an intensity proxy (unpolarized
        # lab source, no monochromator); pyFAI gives positions only.
        lp = (1.0 + np.cos(two_theta_r) ** 2) / (np.sin(theta_r) ** 2 * np.cos(theta_r))
        reflections.append(
            {"d": d_spacing, "two_theta": two_theta_deg, "intensity_raw": lp}
        )

    if not reflections:
        return []

    max_intensity = max(r["intensity_raw"] for r in reflections)
    out = []
    for r in reflections:
        rel_intensity = 100.0 * r["intensity_raw"] / max_intensity
        if rel_intensity >= min_rel_intensity:
            out.append(
                {"d": r["d"], "two_theta": r["two_theta"], "intensity": rel_intensity}
            )
    out.sort(key=lambda r: r["two_theta"])
    return out


def rot_x(deg: float) -> np.ndarray:
    """Rotation matrix about the x axis, angle in degrees."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    """Rotation matrix about the y axis, angle in degrees."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg: float) -> np.ndarray:
    """Rotation matrix about the z axis, angle in degrees."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def detector_rotation(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """Combined rotation matrix: pitch about z, yaw about y, roll about x.

    Note: pitch/yaw are swapped relative to the usual aircraft
    convention, to match this project's reference frame.
    """
    return rot_y(yaw_deg) @ rot_z(pitch_deg) @ rot_x(roll_deg)


def sphere_point(
    radius_mm: float, alpha_deg: np.ndarray | float, phi_deg: float
) -> np.ndarray:
    """Point(s) at radius, polar angle alpha (from +x), azimuth phi.

    Azimuth is measured in the y-z plane from +y toward +z. Returned
    shape is (..., 3), broadcasting with alpha_deg.
    """
    alpha = np.radians(alpha_deg)
    phi = np.radians(phi_deg)
    x = radius_mm * np.cos(alpha)
    y = radius_mm * np.sin(alpha) * np.cos(phi)
    z = radius_mm * np.sin(alpha) * np.sin(phi)
    return np.stack([x, y, z], axis=-1)


class DetectorAlignmentSimulator:
    """Interactive 3D simulator for curved dual-strip detector alignment."""

    def __init__(
        self,
        calibrant_name: str = "LaB6",
        wavelength_a: float = 0.82,
        sample_to_detector_mm: float = 762.0,
        arc_span_deg: float = 80.0,
        arc_center_deg: float = 45.0,
        central_azimuth_deg: float = 0.0,
        n_pixels: int = 1024,
        default_gap_mm: float = 5.0,
        distance_slider_half_range_mm: float = 50.0,
        angular_sigma_deg: float = 0.08,
        n_cones_drawn: int = 9,
    ) -> None:
        self.calibrant_name = calibrant_name
        self.wavelength_a = wavelength_a
        self.radius_mm = sample_to_detector_mm
        self.arc_span = arc_span_deg
        self.arc_center = arc_center_deg
        self.phi_center = central_azimuth_deg
        self.n_pixels = n_pixels
        self.sigma = angular_sigma_deg
        self.default_gap = default_gap_mm
        self.distance_half_range = distance_slider_half_range_mm

        self.reflections = calibrant_reflections(calibrant_name, wavelength_a)
        if not self.reflections:
            raise RuntimeError(
                f"No reflections in range for calibrant '{calibrant_name}' "
                f"at wavelength {wavelength_a} A - check wavelength."
            )

        self.alpha_min = arc_center_deg - arc_span_deg / 2.0
        self.alpha_max = arc_center_deg + arc_span_deg / 2.0
        self.alpha_deg = np.linspace(self.alpha_min, self.alpha_max, n_pixels)

        self.n_cones_drawn = n_cones_drawn
        self._cone_artists: list[Any] = []
        self._ref_line_artists: list[Line2D] = []
        self.cones_to_draw: list[dict[str, float]] = []
        self._select_cones_to_draw()

        self.sliders: dict[str, Slider] = {}
        self._r_max = 0.0
        self._half_extent = 0.0

        self._build_figure()
        self._draw_static_scene()
        self._update(None)
        plt.show()

    def _select_cones_to_draw(self) -> None:
        strongest = sorted(self.reflections, key=lambda r: -r["intensity"])
        top = strongest[: self.n_cones_drawn]
        self.cones_to_draw = sorted(top, key=lambda r: r["two_theta"])

    # -- geometry -------------------------------------------------------

    def _perp_direction(self) -> np.ndarray:
        """Unit vector perpendicular to the central arc's own plane.

        This is the direction the two strips are offset along, so
        they stay exactly parallel rather than diverging.
        """
        phi = np.radians(self.phi_center)
        return np.array([0.0, -np.sin(phi), np.cos(phi)])

    def _detector_pixel_world_coords(
        self,
        pitch_deg: float,
        yaw_deg: float,
        roll_deg: float,
        distance_mm: float,
        ty_mm: float,
        tz_mm: float,
        gap_mm: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """World-space (n, 3) pixel coordinates for strip 1 and strip 2.

        Both strips share the central design arc, offset sideways by
        +-gap/2 to stay parallel, then rotated about the detector's
        vertex and translated by (distance_mm, ty_mm, tz_mm)."""
        perp = self._perp_direction()

        center_nominal = sphere_point(self.radius_mm, self.alpha_deg, self.phi_center)
        strip1_nominal = center_nominal - (gap_mm / 2.0) * perp
        strip2_nominal = center_nominal + (gap_mm / 2.0) * perp

        vertex_nominal = sphere_point(self.radius_mm, self.arc_center, self.phi_center)

        rot_mat = detector_rotation(pitch_deg, yaw_deg, roll_deg)
        offset_x = distance_mm - self.radius_mm
        translation = vertex_nominal + np.array([offset_x, ty_mm, tz_mm])

        world1 = (strip1_nominal - vertex_nominal) @ rot_mat.T + translation
        world2 = (strip2_nominal - vertex_nominal) @ rot_mat.T + translation
        return world1, world2

    def _scattering_angles(self, world_pts: np.ndarray) -> np.ndarray:
        """True 2-theta (deg) between the beam axis and sample->pixel."""
        norms = np.linalg.norm(world_pts, axis=1)
        cos_a = np.clip(world_pts[:, 0] / norms, -1.0, 1.0)
        return np.degrees(np.arccos(cos_a))

    def _pattern_from_angles(self, two_theta_obs: np.ndarray) -> np.ndarray:
        """Sum of Gaussians at each calibrant peak, sampled per pixel."""
        intensity = np.zeros_like(two_theta_obs)
        for r in self.reflections:
            delta = (two_theta_obs - r["two_theta"]) / self.sigma
            intensity += r["intensity"] * np.exp(-0.5 * delta**2)
        return intensity

    # -- figure / widgets -------------------------------------------------

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(16, 10.2))
        assert self.fig.canvas.manager is not None
        self.fig.canvas.manager.set_window_title(
            "Calibrant Debye-Scherrer / Curved Dual-Strip Detector Simulator"
        )

        grid = self.fig.add_gridspec(
            1,
            2,
            left=0.04,
            right=0.98,
            top=0.95,
            bottom=0.36,
            width_ratios=[1.25, 1.0],
            wspace=0.28,
        )

        self.ax_3d: Axes3D = self.fig.add_subplot(grid[0, 0], projection="3d")
        grid_right = grid[0, 1].subgridspec(2, 1, hspace=0.35)
        self.ax_p1: Axes = self.fig.add_subplot(grid_right[0, 0])
        self.ax_p2: Axes = self.fig.add_subplot(grid_right[1, 0])

        self.ax_3d.set_xlabel("x (beam) [mm]")
        self.ax_3d.set_ylabel("y [mm]")
        self.ax_3d.set_zlabel("z [mm]")
        self.ax_3d.set_title(f"Sample, {self.calibrant_name} cones, detector pose")

        self.ax_p1.set_title("Strip 1 (offset side -gap/2) - observed pattern")
        self.ax_p2.set_title("Strip 2 (offset side +gap/2) - observed pattern")
        for ax in (self.ax_p1, self.ax_p2):
            ax.set_xlabel("Pixel design angle (nominal 2\u03b8) [deg]")
            ax.set_ylabel("Intensity [a.u.]")
            ax.set_ylim(0, 115)

        self._build_calibrant_controls()
        self._build_sliders()

    def _build_calibrant_controls(self) -> None:
        """Text boxes to pick the calibrant and wavelength live."""
        y0 = 0.305
        height = 0.032

        ax_cal = self.fig.add_axes((0.06, y0, 0.16, height))
        self.tb_calibrant = TextBox(ax_cal, "Calibrant  ", initial=self.calibrant_name)
        self.tb_calibrant.on_submit(self._on_calibrant_submit)

        ax_wave = self.fig.add_axes((0.34, y0, 0.10, height))
        wavelength_str = f"{self.wavelength_a:g}"
        self.tb_wavelength = TextBox(
            ax_wave, "\u03bb [\u00c5]  ", initial=wavelength_str
        )
        self.tb_wavelength.on_submit(self._on_wavelength_submit)

        ax_status = self.fig.add_axes((0.06, y0 - 0.040, 0.90, 0.030))
        ax_status.axis("off")
        self.status_text: Text = ax_status.text(
            0, 0.5, "", fontsize=7.8, va="center", color="0.35"
        )
        self._refresh_status_text()

    def _refresh_status_text(self, message: str | None = None) -> None:
        available = ", ".join(sorted(CALIBRANT_FACTORY.keys()))
        base = (
            f"Loaded {self.calibrant_name}: {len(self.reflections)} lines shown "
            f"(\u03bb={self.wavelength_a:g} \u00c5).  Available calibrants: {available}"
        )
        self.status_text.set_text(message if message else base)

    def _on_calibrant_submit(self, text: str) -> None:
        name = text.strip()
        if name not in CALIBRANT_FACTORY:
            available = ", ".join(sorted(CALIBRANT_FACTORY.keys()))
            self._refresh_status_text(
                f"'{name}' is not a known pyFAI calibrant - keeping "
                f"'{self.calibrant_name}'. Available: {available}"
            )
            self.tb_calibrant.set_val(self.calibrant_name)
            return
        self.calibrant_name = name
        self._reload_calibrant()

    def _on_wavelength_submit(self, text: str) -> None:
        try:
            value = float(text.strip())
            if not (0.01 < value < 20.0):
                raise ValueError
        except ValueError:
            self._refresh_status_text(
                f"'{text}' is not a valid wavelength in \u00c5 - keeping "
                f"{self.wavelength_a:g} \u00c5."
            )
            self.tb_wavelength.set_val(f"{self.wavelength_a:g}")
            return
        self.wavelength_a = value
        self._reload_calibrant()

    def _reload_calibrant(self) -> None:
        try:
            new_reflections = calibrant_reflections(
                self.calibrant_name, self.wavelength_a
            )
        except Exception as exc:  # noqa: BLE001 - surface any load error to the user
            self._refresh_status_text(f"Error loading '{self.calibrant_name}': {exc}")
            return
        if not new_reflections:
            self._refresh_status_text(
                f"'{self.calibrant_name}' has no reflections below 2\u03b8=175deg "
                f"at {self.wavelength_a:g} \u00c5 - try a shorter wavelength."
            )
            return

        self.reflections = new_reflections
        self._select_cones_to_draw()
        self._redraw_cones_and_references()
        self.ax_3d.set_title(f"Sample, {self.calibrant_name} cones, detector pose")
        self._refresh_status_text()
        self._update(None)
        self.fig.canvas.draw_idle()

    def _build_sliders(self) -> None:
        dist = self.radius_mm
        half = self.distance_half_range
        specs: list[tuple[str, float, float, float, str]] = [
            ("pitch", -30.0, 30.0, 0.0, "deg"),
            ("yaw", -30.0, 30.0, 0.0, "deg"),
            ("roll", -90.0, 90.0, 0.0, "deg"),
            ("distance", dist - half, dist + half, dist, "mm"),
            ("ty", -50.0, 50.0, 0.0, "mm"),
            ("tz", -50.0, 50.0, 0.0, "mm"),
            ("gap", 0.0, 60.0, self.default_gap, "mm"),
        ]
        col_w = 0.36
        left_col_x = 0.11
        right_col_x = 0.61
        top_y = 0.215
        row_h = 0.030
        row_gap = 0.009

        for i, (name, v_min, v_max, v_init, unit) in enumerate(specs):
            col = 0 if i < 4 else 1
            row = i if i < 4 else i - 4
            x0 = left_col_x if col == 0 else right_col_x
            y0 = top_y - row * (row_h + row_gap)
            ax = self.fig.add_axes((x0, y0, col_w, row_h))
            slider = Slider(ax, f"{name} [{unit}]", v_min, v_max, valinit=v_init)
            slider.on_changed(self._update)
            self.sliders[name] = slider

        button_row_y = top_y - 3.6 * (row_h + row_gap)
        reset_x = right_col_x + col_w - 0.10
        ax_reset = self.fig.add_axes((reset_x, button_row_y, 0.10, 0.045))
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_reset.on_clicked(self._reset)

        self._build_view_buttons(right_col_x, button_row_y)

        self.ax_info = self.fig.add_axes((0.04, 0.005, 0.94, 0.050))
        self.ax_info.axis("off")
        self.info_text: Text = self.ax_info.text(
            0, 0.5, "", fontsize=8.5, va="center", family="monospace"
        )

    def _build_view_buttons(self, x0: float, y0: float) -> None:
        """Buttons to snap the 3D camera to x, y, z, or an isometric view."""
        width = 0.048
        gap = 0.010
        view_specs: list[tuple[str, Callable[[Any], None]]] = [
            ("X", self._view_x),
            ("Y", self._view_y),
            ("Z", self._view_z),
            ("Iso", self._view_iso),
        ]
        self.view_buttons: dict[str, Button] = {}
        for i, (label, callback) in enumerate(view_specs):
            ax_view = self.fig.add_axes((x0 + i * (width + gap), y0, width, 0.045))
            button = Button(ax_view, label)
            button.on_clicked(callback)
            self.view_buttons[label] = button

    def _reset(self, event: Any) -> None:
        for slider in self.sliders.values():
            slider.reset()

    def _view_x(self, event: Any) -> None:
        """Look down the beam (+x) axis: detector face-on (y-z plane)."""

        self.ax_3d.view_init(elev=0, azim=0)
        self.fig.canvas.draw_idle()

    def _view_y(self, event: Any) -> None:
        """Look down the y axis: beam vs. strip-offset (x-z) plane."""
        self.ax_3d.view_init(elev=0, azim=-90)
        self.fig.canvas.draw_idle()

    def _view_z(self, event: Any) -> None:
        """Look down the z axis: top-down beam/arc-sweep (x-y) plane."""
        self.ax_3d.view_init(elev=90, azim=-90)
        self.fig.canvas.draw_idle()

    def _view_iso(self, event: Any) -> None:
        """Restore the default isometric-ish 3D perspective."""
        self.ax_3d.view_init(elev=20, azim=-60)
        self.fig.canvas.draw_idle()

    # -- static scene ------------------------------------------------------

    def _draw_static_scene(self) -> None:
        ax = self.ax_3d
        ax.view_init(elev=20, azim=-60)

        # equal-range, equal-aspect box: a true (undistorted) square/cube
        # view, so circles stay circular and angles are not skewed.
        self._r_max = self.radius_mm * 1.35
        self._half_extent = self._r_max / 2.0
        r_max, half_extent = self._r_max, self._half_extent

        ax.scatter([0], [0], [0], color="black", s=60, marker="o", label="Sample")  # type: ignore
        ax.plot(
            [0, r_max],
            [0, 0],
            [0, 0],
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Beam axis",
        )

        perp = self._perp_direction()
        center_nominal = sphere_point(self.radius_mm, self.alpha_deg, self.phi_center)
        ghost1 = center_nominal - (self.default_gap / 2.0) * perp
        ghost2 = center_nominal + (self.default_gap / 2.0) * perp
        for ghost in (ghost1, ghost2):
            ax.plot(
                ghost[:, 0],
                ghost[:, 1],
                ghost[:, 2],
                color="0.6",
                linestyle=":",
                linewidth=1.3,
            )

        ax.set_xlim(0, r_max)
        ax.set_ylim(-half_extent, half_extent)
        ax.set_zlim(-half_extent, half_extent)
        ax.set_box_aspect((1, 1, 1))

        self.line_strip1 = Line3D(
            [], [], [], color="crimson", linewidth=3, label="Strip 1"
        )
        self.line_strip2 = Line3D(
            [], [], [], color="royalblue", linewidth=3, label="Strip 2"
        )
        ax.add_line(self.line_strip1)
        ax.add_line(self.line_strip2)
        ax.legend(loc="upper left", fontsize=8)

        (self.pat_line1,) = self.ax_p1.plot(
            [], [], color="crimson", linewidth=1.2, zorder=3
        )
        (self.pat_line2,) = self.ax_p2.plot(
            [], [], color="royalblue", linewidth=1.2, zorder=3
        )

        self._redraw_cones_and_references()

    def _redraw_cones_and_references(self) -> None:
        """Redraw the cone surfaces and "correct position" reference vlines.

        Clears any previously drawn ones first, so this is safe to
        call again after a calibrant/wavelength change.
        """
        ax = self.ax_3d
        r_max = self._r_max

        for artist in self._cone_artists:
            artist.remove()
        self._cone_artists = []
        for artist in self._ref_line_artists:
            artist.remove()
        self._ref_line_artists = []

        cmap = plt.get_cmap("plasma")
        phi_lin = np.linspace(0, 2 * np.pi, 60)
        r_lin = np.linspace(0, r_max, 2)
        phi_grid, r_grid = np.meshgrid(phi_lin, r_lin)
        n_cones = max(1, len(self.cones_to_draw) - 1)
        for i, refl in enumerate(self.cones_to_draw):
            two_theta_rad = np.radians(refl["two_theta"])
            cone_radius = r_grid * np.tan(two_theta_rad)
            x_grid = r_grid
            y_grid = cone_radius * np.cos(phi_grid)
            z_grid = cone_radius * np.sin(phi_grid)
            color = cmap(i / n_cones)
            surface = ax.plot_surface(
                x_grid,
                y_grid,
                z_grid,
                color=color,
                alpha=0.10,
                linewidth=0,
                antialiased=True,
                shade=False,
            )
            self._cone_artists.append(surface)

        # "correct" (calibrant-defined) reference positions: each strip's
        # pixel angle is the nominal 2-theta by construction (to within
        # the small parallel-offset residual noted in the module notes).
        for ax_pattern in (self.ax_p1, self.ax_p2):
            first = True
            for refl in self.reflections:
                if self.alpha_min <= refl["two_theta"] <= self.alpha_max:
                    line = ax_pattern.axvline(
                        refl["two_theta"],
                        color="forestgreen",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.65,
                        zorder=1,
                        label="Correct calibrant position" if first else None,
                    )
                    self._ref_line_artists.append(line)
                    first = False
            ax_pattern.set_xlim(self.alpha_min, self.alpha_max)
            ax_pattern.legend(loc="upper right", fontsize=7)

    # -- update on slider change --------------------------------------------

    def _update(self, val: Any) -> None:
        pitch = self.sliders["pitch"].val
        yaw = self.sliders["yaw"].val
        roll = self.sliders["roll"].val
        distance = self.sliders["distance"].val
        ty = self.sliders["ty"].val
        tz = self.sliders["tz"].val
        gap = self.sliders["gap"].val

        world1, world2 = self._detector_pixel_world_coords(
            pitch, yaw, roll, distance, ty, tz, gap
        )

        step = max(1, self.n_pixels // 300)
        w1x, w1y, w1z = world1[::step, 0], world1[::step, 1], world1[::step, 2]
        w2x, w2y, w2z = world2[::step, 0], world2[::step, 1], world2[::step, 2]
        self.line_strip1.set_data_3d(w1x, w1y, w1z)
        self.line_strip2.set_data_3d(w2x, w2y, w2z)

        two_theta1 = self._scattering_angles(world1)
        two_theta2 = self._scattering_angles(world2)
        intensity1 = self._pattern_from_angles(two_theta1)
        intensity2 = self._pattern_from_angles(two_theta2)

        self.pat_line1.set_data(self.alpha_deg, intensity1)
        self.pat_line2.set_data(self.alpha_deg, intensity2)

        self._update_info_text(
            pitch,
            yaw,
            roll,
            distance,
            ty,
            tz,
            gap,
            two_theta1,
            two_theta2,
            intensity1,
            intensity2,
        )
        self.fig.canvas.draw_idle()

    def _update_info_text(
        self,
        pitch: float,
        yaw: float,
        roll: float,
        distance: float,
        ty: float,
        tz: float,
        gap: float,
        two_theta1: np.ndarray,
        two_theta2: np.ndarray,
        intensity1: np.ndarray,
        intensity2: np.ndarray,
    ) -> None:
        """Compute and display the inter-strip peak-mismatch diagnostic."""
        strongest = max(self.reflections, key=lambda r: r["intensity"])
        tol = 6 * self.sigma
        mask1 = np.abs(two_theta1 - strongest["two_theta"]) < tol
        mask2 = np.abs(two_theta2 - strongest["two_theta"]) < tol

        peak1 = np.nan
        if np.any(mask1) and np.sum(intensity1[mask1]) > 0:
            weights = intensity1[mask1]
            peak1 = np.sum(self.alpha_deg[mask1] * weights) / np.sum(weights)

        peak2 = np.nan
        if np.any(mask2) and np.sum(intensity2[mask2]) > 0:
            weights = intensity2[mask2]
            peak2 = np.sum(self.alpha_deg[mask2] * weights) / np.sum(weights)

        both_finite = np.isfinite(peak1) and np.isfinite(peak2)
        mismatch = peak1 - peak2 if both_finite else np.nan

        info = (
            f"pose:  pitch={pitch:+6.2f} deg   yaw={yaw:+6.2f} deg   "
            f"roll={roll:+6.2f} deg   distance={distance:7.2f} mm   "
            f"ty={ty:+6.2f} mm   tz={tz:+6.2f} mm   gap={gap:5.2f} mm\n"
            f"strongest line ({strongest['two_theta']:.2f} deg 2\u03b8) observed "
            f"centroid:  strip1={peak1:+7.3f} deg   strip2={peak2:+7.3f} deg   "
            f"inter-strip mismatch={mismatch:+7.3f} deg"
        )
        self.info_text.set_text(info)


if __name__ == "__main__":
    DetectorAlignmentSimulator(
        calibrant_name="LaB6",
        wavelength_a=0.82,
        sample_to_detector_mm=762.0,
        arc_span_deg=80.0,
        arc_center_deg=45.0,
        central_azimuth_deg=0.0,
        n_pixels=1024,
        default_gap_mm=5.0,
        angular_sigma_deg=0.08,
        n_cones_drawn=9,
    )
