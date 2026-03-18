import math
from collections.abc import Collection
from typing import Literal

import numpy as np
import peakutils
from pydantic import BaseModel, model_validator
from scipy.optimize import curve_fit


def gaussian_sigma_to_fwhm(sigma: float | int) -> float:
    return float(sigma) * 2 * np.sqrt(2 * np.log(2))


def gaussian_fwhm_to_sigma(fwhm: float | int) -> float:
    return float(fwhm) / (2 * np.sqrt(2 * np.log(2)))


def lorentzian_gamma_to_fwhm(gamma: float | int) -> float:
    return 2 * float(gamma)


def lorentzian_fwhm_to_gamma(fwhm: float | int) -> float:
    return float(fwhm) / 2


def gaussian(
    x: np.ndarray,
    amplitude: float | int,
    centre: float | int,
    fwhm: float | int,
    background: float | int | np.ndarray = 0,
    normalised: bool = True,
) -> np.ndarray:
    """
    Gaussian peak function.

    Parameters
    ----------
    x : array-like
        Input coordinate(s).
    amplitude : float | int
        Amplitude parameter:
        - If normalised=True: total area under the curve.
        - If normalised=False: peak height.
    centre : float | int
        Peak center.
    fwhm : float | int
        Full-Wdth at half maximum of the peak - the peak width (must be > 0).
    background : float | int | array-like, optional
        Additive background (constant or array matching `x`). Default is 0.
    normalised : bool, optional
        If True (default), returns an area-normalised Gaussian.
        If False, returns a Gaussian with peak height A.

    Returns
    -------
    NDArray[np.float64]
        Evaluated Gaussian function.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)

    sigma = gaussian_fwhm_to_sigma(fwhm)

    if normalised:
        prefactor = amplitude / (sigma * np.sqrt(2 * np.pi))
    else:
        prefactor = amplitude

    return prefactor * np.exp(-((x_arr - centre) ** 2) / (2 * sigma**2)) + bg


def lorentzian(
    x: np.ndarray,
    amplitude: float | int,
    centre: float | int,
    fwhm: float | int,
    background: float | int | np.ndarray = 0,
    normalised: bool = True,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    bg = np.asarray(background, dtype=float)

    gamma = float(fwhm) / 2

    if normalised:
        prefactor = amplitude / np.pi
        core = gamma / ((x - centre) ** 2 + gamma**2)
    else:
        prefactor = amplitude
        core = gamma**2 / ((x - centre) ** 2 + gamma**2)

    return prefactor * core + bg


def pseudo_voigt(
    x: np.ndarray,
    amplitude: float | int,
    centre: float | int,
    fwhm: float | int,
    eta: float | int,
    background: float | int | np.ndarray = 0,
    normalised: bool = True,
) -> np.ndarray:
    """
    Pseudo-Voigt peak function with a single FWHM parameter.

    Parameters
    ----------
    x : array-like
        Input coordinate(s).
    amplitude : float | int
        Amplitude parameter:
        - If normalised=True: total area under the curve.
        - If normalised=False: approximate peak height.
    centre : float | int
        Peak center.
    fwhm : float | int
        Full width at half maximum (must be > 0).
    eta : float | int
        Mixing parameter:
        - 0 → pure Gaussian
        - 1 → pure Lorentzian
    background : float | int | array-like, optional
        Additive background. Default is 0.
    normalised : bool, optional
        If True (default), area-normalised.
        If False, amplitude ≈ peak height.

    Returns
    -------
    NDArray[np.float64]
        Evaluated pseudo-Voigt function.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)

    fwhm = float(fwhm)
    eta = float(eta)

    # Convert FWHM to internal parameters
    sigma = gaussian_fwhm_to_sigma(fwhm)
    gamma = lorentzian_fwhm_to_gamma(fwhm)

    if normalised:
        gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((x_arr - centre) ** 2) / (2 * sigma**2)
        )
        lorentz = (1 / np.pi) * (gamma / ((x_arr - centre) ** 2 + gamma**2))
    else:
        gauss = np.exp(-((x_arr - centre) ** 2) / (2 * sigma**2))
        lorentz = gamma**2 / ((x_arr - centre) ** 2 + gamma**2)

    return amplitude * (eta * lorentz + (1 - eta) * gauss) + bg


def smooth_tophat(
    x: np.ndarray,
    amplitude: float | int,
    centre: float | int,
    fwhm: float | int,
    sharpness: float | int,
    background: float | int | np.ndarray = 0,
    normalised: bool = True,
) -> np.ndarray:
    """
    Smoothed top-hat using FWHM parameterisation.

    FWHM is defined as the distance between half-maximum points.
    """

    x_arr = np.asarray(x, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)

    fwhm = float(fwhm)
    sharpness = float(sharpness)

    # Convert FWHM → effective plateau width
    edge_correction = 2 * np.log(3) / sharpness
    width = max(fwhm - edge_correction, 1e-12)  # avoid negative width

    left = 1 / (1 + np.exp(-sharpness * (x_arr - (centre - width / 2))))
    right = 1 / (1 + np.exp(-sharpness * (x_arr - (centre + width / 2))))

    core = left - right

    if normalised:
        scale = amplitude / fwhm  # area ~ fwhm
    else:
        scale = amplitude

    return scale * core + bg


def closest_indices(arr1, arr2):
    """
    For each value in arr1, find the index of the closest value in arr2.
    Returns an array of indices with the same shape as arr1.
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    # Broadcast arr1 and arr2 to compute pairwise differences
    diffs = np.abs(arr1[..., np.newaxis] - arr2)
    # Find the index of the minimum difference along the last axis (arr2)
    idx = np.argmin(diffs, axis=-1)
    return idx


class Peak(BaseModel):
    amplitude: float | int
    centre: float | int
    fwhm: float | int

    peak_type: Literal["gaussian", "lorentzian", "pseudo-voigt"] = "gaussian"

    eta: float | int | None = None  # only used for pseudo-voigt - mixing parameter
    sharpness: float | int | None = None  # only used for tophat

    background: float | int = 0
    normalised: bool = True  # if normalised the
    # integral under peak is equal to amplitude. ie number of counts in peak

    @model_validator(mode="after")
    def validate_parameters(self):
        # Allow NaNs to bypass strict validation
        if any(
            isinstance(v, float) and math.isnan(v)
            for v in [self.amplitude, self.centre, self.fwhm]
        ):
            return self

        # Manual constraints (NaN-safe)
        if self.amplitude <= 0:
            raise ValueError("amplitude must be > 0")

        if self.fwhm <= 0:
            raise ValueError("fwhm must be > 0")

        if self.peak_type == "pseudo-voigt":
            if self.eta is None:
                raise ValueError("eta must be provided for pseudo-voigt")
            if not (0 <= self.eta <= 1):
                raise ValueError("eta must be between 0 and 1")
        else:
            if self.eta is not None:
                raise ValueError("eta should only be set for pseudo-voigt")

        if self.peak_type == "top-hat":
            if self.sharpness is None:
                raise ValueError("eta must be provided for top-hat")
            if self.sharpness <= 0:
                raise ValueError("sharpness must be gretaer than 0")
        else:
            if self.sharpness is not None:
                raise ValueError("eta should only be set for top-hat")

        return self

    def calculate(self, x: np.ndarray) -> np.ndarray:
        if self.peak_type == "gaussian":
            return gaussian(
                x=x,
                amplitude=self.amplitude,
                centre=self.centre,
                fwhm=self.fwhm,
                background=self.background,
                normalised=self.normalised,
            )
        elif self.peak_type == "lorentzian":
            return lorentzian(
                x=x,
                amplitude=self.amplitude,
                centre=self.centre,
                fwhm=self.fwhm,
                background=self.background,
                normalised=self.normalised,
            )
        elif self.peak_type == "pseudo-voigt":
            assert self.eta is not None

            return pseudo_voigt(
                x=x,
                amplitude=self.amplitude,
                centre=self.centre,
                fwhm=self.fwhm,
                eta=self.eta,
                background=self.background,
                normalised=self.normalised,
            )
        elif self.peak_type == "top-hat":
            assert self.sharpness is not None

            return smooth_tophat(
                x=x,
                amplitude=self.amplitude,
                centre=self.centre,
                fwhm=self.fwhm,
                sharpness=self.sharpness,
                background=self.background,
                normalised=self.normalised,
            )

        else:
            raise ValueError(f"{self.peak_type} is not an allowed peak type")


def fit_peaks(
    x: np.ndarray, y: np.ndarray, initial_x_pos: Collection[int | float]
) -> list[Peak]:
    fitted_peaks = []

    for x_guess in initial_x_pos:
        try:
            width_guess = 2
            # Estimate amplitude from nearest data point
            idx = np.argmin(np.abs(x - x_guess))
            amp_guess = y[idx] * np.sqrt(2 * np.pi) * width_guess

            p0 = [
                amp_guess,
                x_guess,
                width_guess,
            ]

            start_idx = np.searchsorted(x, x_guess - 1)
            end_idx = np.searchsorted(x, x_guess + 1, side="right")

            x_fit = x[start_idx:end_idx]
            y_fit = y[start_idx:end_idx]

            if len(y_fit) == 0:
                fitted_peaks.append(Peak(amplitude=np.nan, centre=np.nan, fwhm=np.nan))
                continue

            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=10000)  # type: ignore

            fitted_peaks.append(Peak(amplitude=popt[0], centre=popt[1], fwhm=popt[2]))

        except RuntimeError:
            fitted_peaks.append(Peak(amplitude=np.nan, centre=np.nan, fwhm=np.nan))

    return fitted_peaks


def calculate_profile(
    x: np.ndarray,
    peaks: Collection[Peak],
    background: int | float | np.ndarray,
    phase_scale: int | float = 1,
    wdt: int | float = 5,
):
    """wdt (range) of calculated profile of a single Bragg reflection in units of FWHM
    (typically 4 for Gaussian and 20-30 for Lorentzian, 4-5 for TOF).

    peaks: list of class: Peak which contain (cen, amp, fwhm)

    background: scalar or array, if array must be same shape as x
    """

    if isinstance(background, np.ndarray):
        assert len(x) == len(background)

    intensity = np.zeros_like(x) + background

    for peak in peaks:
        assert peak.background == 0

        start_idx = np.searchsorted(x, peak.centre - (wdt * peak.fwhm))
        end_idx = np.searchsorted(x, peak.centre + (wdt * peak.fwhm), side="right")

        xi = x[start_idx:end_idx]
        peak_intensity = peak.calculate(xi) * phase_scale
        intensity[start_idx:end_idx] += peak_intensity

    return intensity


def find_and_fit_peaks(x: np.ndarray, y: np.ndarray) -> list[Peak]:
    """function to get the centre peaks given without guessing"""

    y_smoothed = np.convolve(
        y, np.ones(5), mode="same"
    )  # smooth the data to reduce noise

    threshold = np.amax(y_smoothed) / 20
    indexes = peakutils.indexes(y_smoothed, thres=threshold, min_dist=30)  # type: ignore

    initial_x_pos = x[indexes]
    fitted_peaks = fit_peaks(x, y_smoothed, initial_x_pos=initial_x_pos)

    return fitted_peaks
