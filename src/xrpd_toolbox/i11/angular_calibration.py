import json
import os
import pickle
from collections.abc import Collection, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peakutils
from h5py import File as h5pyFile
from lmfit import Parameters, minimize, report_fit
from lmfit.minimizer import MinimizerResult
from pyFAI.calibrant import get_calibrant
from scipy.interpolate import interp1d

from xrpd_toolbox.fit_engine.peaks import closest_indices
from xrpd_toolbox.i11.mythen import (
    CENTRE,
    MYTHEN_PIXEL_SIZE,
    PIXELS_PER_MODULE,
    PSD_RADIUS,
    AngularCalibration,
    ModuleConversion,
    ModuleConversion2D,
    MythenDetector,
    MythenSettings,
)
from xrpd_toolbox.utils.mythen_utils import (
    calc_intial_module_conv,
    calc_starting_module_offset,
    channel_to_angle,
    # paired_modules,
    read_singular_angcal_files,
)
from xrpd_toolbox.utils.utils import (
    h5_to_array,
    load_int_array_from_file,
    rebin_together,
)

PEAK_MASK_LITERAL = Literal["select_peaks", "below", "below2", "max", "between"]


def module_distance(module: int):
    max_dist = 13.5

    distance = np.abs(module - 14)
    # Normalize so:
    # distance = 0   → red
    # distance = max → blue
    normalised = 1 - (distance / max_dist)

    return normalised


def plot_convs(conv: pd.DataFrame, steps):
    plt.figure(figsize=(16, 10))

    import matplotlib.cm as cm

    cmap = cm.get_cmap("bwr")

    for mod in range(28):
        if mod in [11]:
            continue

        normalised = module_distance(mod)

        color = cmap(normalised)

        if mod > 13:
            line = "--"
        else:
            line = "-"

        plt.plot(
            steps,
            conv[str(mod)],
            label=str(mod),
            linestyle=line,
            color=color,
        )

    plt.legend()
    plt.xlabel("Delta Range of Calibration")
    plt.ylabel("Calibrated Distance of Module (mm)")
    plt.savefig("/host-home/projects/outputs/step_cals.png")
    plt.show()
    plt.close()


def top_n_recurring(arr, n):
    arr = np.array(arr)
    # Get unique values and their counts
    unique_vals, counts = np.unique(arr, return_counts=True)

    # Sort by counts descending
    sorted_indices = np.argsort(counts)[::-1]

    # Select top n
    top_n = unique_vals[sorted_indices][:n]

    return top_n


def index_of_closest(arr, value):
    """
    Return the index of the closest value in arr to the given value.
    """
    arr = np.asarray(arr)
    idx = np.abs(arr - value).argmin()
    return idx


def gaussian(x: np.ndarray, cen: float, amp: float, fwhm: float):
    # "1-d gaussian: gaussian(x, amp, cen, fwhm)"

    return (amp / (np.sqrt(2 * np.pi) * fwhm)) * np.exp(
        -((x - cen) ** 2) / (2 * fwhm**2)
    )


def multi_gaussian(x: np.ndarray, peaks, background=0, phase_scale=1, wdt: int = 4):
    """wdt (range) of calculated profile of a single Bragg reflection in units of FWHM
    (typically 4
    for Gaussian and 20-30 for Lorentzian, 4-5 for TOF).

    peaks: list of (cen, amp, fwhm)

    background: scalar or array
    """

    y = np.zeros_like(x) + background

    for peak in peaks:
        cen, amp, fwhm = peak
        start_idx = np.searchsorted(x, cen - wdt)
        end_idx = np.searchsorted(x, cen + wdt, side="right")

        xi = x[start_idx:end_idx]
        peak = gaussian(xi, cen, amp, fwhm) * phase_scale

        y[start_idx:end_idx] += peak

    return y


def generate_filepaths(data_dir, nexus_file_numbers):
    filepaths = []

    for data_file_number in nexus_file_numbers:
        filepath = os.path.join(data_dir, f"{data_file_number}.nxs")
        filepaths.append(filepath)

    return filepaths


class AngularCalibrateMythen:
    def __init__(
        self,
        filepaths,
        wavelength_in_ang,
        module_centre=CENTRE,
        active_modules=tuple(range(28)),  # noqa
        bad_modules: list[int] = [17],  # noqa
        lower_delta=0,
        upper_delta=90,
    ):

        self.DEFAULT_COUNTER = 0
        self.STRIPS_PER_MODULE = PIXELS_PER_MODULE
        self.p = MYTHEN_PIXEL_SIZE  # pixel size in mm
        self.psd_radius = PSD_RADIUS

        self.filepaths = filepaths
        self.wavelength_in_ang = wavelength_in_ang

        self.lower_delta = lower_delta
        self.upper_delta = upper_delta

        self.active_modules = list(active_modules)
        self.bad_modules = bad_modules
        self.good_modules = [
            f for f in self.active_modules if f not in self.bad_modules
        ]

        calibrant = get_calibrant("Si")
        calibrant.wavelength = self.wavelength_in_ang / 1e10
        self.observed_reflections_in_tth = calibrant.get_peaks("2th_deg")

        print(self.observed_reflections_in_tth)

        self.peaks_to_fit = [
            69.6350914079859,
            71.75570444785892,
            75.23501956270378,
            77.29530751474618,
            80.69374073239572,
            82.71626960778555,
            86.06823060567002,
            88.07221803978473,
        ]

        self.module_centre = module_centre  # 639.5p = 31.975 mm from the center of det
        self.module_pixel_number = np.arange(self.STRIPS_PER_MODULE, dtype=np.int64)

        self.init_conv = self.p / self.psd_radius  # 6.56e-5

        self.single_peak = True  # True is much better

        ang_cal = "/host-home/projects/outputs/mythen_calibration/processed/ang_cal_171125.off"  # noqa
        self.module_angular_cal, self.beamline_offset = read_singular_angcal_files(
            ang_cal
        )  # ["offset"], module_cal["conv"], module_cal["centre"]

        self.bad_channels = load_int_array_from_file(
            "/workspaces/XRPD-Toolbox/config/i11/badchannels.txt"
        )

        ########################################

    def get_selected_peaks(
        self,
        use_pickle: bool = True,
        mask_type: PEAK_MASK_LITERAL = "below",
        plot_fit: bool = True,
    ):

        self.use_pickle = use_pickle

        fitted_peaks_for_modules_file = (
            f"/host-home/projects/outputs/{si_nexus_file_numbers[0]}_fitted_peaks.obj"
        )

        if not use_pickle:
            new_split_si_filepaths = [
                f"/host-home/projects/outputs/mythen_calibration/processed/{f}_modules.h5"
                for f in si_nexus_file_numbers
            ]

            self.module_dataset_filepath = new_split_si_filepaths[0]

            if not os.path.exists(self.module_dataset_filepath):
                new_split_si_filepaths = self.split_into_modules(
                    self.filepaths, bad_channels=list(self.bad_channels)
                )  # (delta, n_modules, PIXELS_PER_MODULE)

            delta_points = self.get_delta_points(self.filepaths[0])
            fitted_peaks_for_modules = self.fit_peaks_across_delta(
                delta_points=delta_points,
                module_angular_cal=self.module_angular_cal,
                modules=self.active_modules,
                observed_reflections_in_tth=self.observed_reflections_in_tth,
                beamline_offset=self.beamline_offset,
            )

            with open(fitted_peaks_for_modules_file, "wb") as fp:
                pickle.dump(fitted_peaks_for_modules, fp)

        else:
            with open(fitted_peaks_for_modules_file, "rb") as pickle_file:
                fitted_peaks_for_modules = pickle.load(pickle_file)

        fitted_peaks_for_modules = self.remove_bad_modules(fitted_peaks_for_modules)

        if plot_fit:
            self.plot_fit_stats(fitted_peaks_for_modules)

        self.fitted_peaks_for_modules = self.select_peaks(
            fitted_peaks_for_modules, mask_type=mask_type
        )

    def add_bad_modules_to_results(self, results_dict: dict):

        for bad_mod in self.bad_modules:
            results_dict[f"conv_{bad_mod}"] = self.module_angular_cal[bad_mod]["conv"]
            results_dict[f"offset_{bad_mod}"] = self.module_angular_cal[bad_mod][
                "offset"
            ]
            results_dict[f"centre_{bad_mod}"] = self.module_centre

        return results_dict

    def fit(
        self,
        fit_method: str,
        module_conversion: type[
            ModuleConversion | ModuleConversion2D
        ] = ModuleConversion2D,
        starting_params="guess",
        plot_fit: bool = True,
    ):

        self.module_conversion = module_conversion

        self.plot_iter = 0

        self.resid_per_module = {}
        for module in self.good_modules:
            self.resid_per_module[module] = []

        # if starting_params == "guess":
        params = self.create_starting_params(self.beamline_offset)
        # else:
        #     params = self.create_starting_params_from_original(
        #         self.module_angular_cal, self.beamline_offset
        #     )

        results = minimize(
            self.return_residual_for_modules,
            params,
            args=(self.good_modules, self.fitted_peaks_for_modules),
            nan_policy="omit",
            method=fit_method,
        )

        results: MinimizerResult

        report_fit(results)

        self.residual = results.residual

        angcal_filepath = f"/host-home/projects/outputs/mythen_calibration/processed/ang_cal_020426_cen_{self.module_centre}_{fit_method}_{self.bad_modules}.off"  # noqa

        self.results_dict: dict = results.params.valuesdict()  # type: ignore
        print(self.results_dict)

        self.results_dict = self.add_bad_modules_to_results(self.results_dict)

        self.save_results(
            results_dict=self.results_dict,
            filepath=angcal_filepath,
            modules=self.active_modules,
            bad_modules=self.bad_modules,
            original_ang_cal=self.module_angular_cal,
        )

        # print(AngularCalibration.model_fields)

        pydantic_dict = self.results_dict_to_pydantic(self.results_dict)
        angular_calibration = AngularCalibration.model_validate(pydantic_dict)
        angular_calibration.save_to_json(angcal_filepath.replace(".off", ".json"))

        # check_files = "/dls/i11/data/2025/cm40625-5/1399181.nxs"
        # check_files = [
        #     "/host-home/projects/outputs/step_scan/1410696.nxs",
        #     "/host-home/projects/outputs/step_scan/1414223.nxs",
        # ]

        settings = MythenSettings(bad_modules=self.bad_modules)

        # for data_file in check_files:
        basename = os.path.basename(self.filepaths[0])

        analysis = MythenDetector(
            filepath=self.filepaths[0],
            settings=settings,
            angular_calibration=angular_calibration,
        )

        if plot_fit:
            for peak in self.observed_reflections_in_tth[[3, 5, 11]]:
                analysis.plot_by_region_of_interest(
                    [peak],
                    tol=0.04,
                    filepath=f"/host-home/projects/outputs/roi_{basename}.png",
                )

            analysis.plot_diffraction_by_mod()

    def split_into_modules(
        self,
        filespaths: list[str],
        modules: Collection[int] = tuple(range(28)),
        bad_channels: Sequence[int] = (),
    ):
        n_modules = len(modules)

        out_filepaths = []

        for filepath in filespaths:
            delta = h5_to_array(filepath, "/entry/mythen_nx/delta")

            filename = os.path.basename(filepath)
            filenumber = filename.replace(".nxs", "")

            n_delta_points = delta.shape[0]

            module_array = np.zeros((n_delta_points, n_modules, self.STRIPS_PER_MODULE))

            for i in range(delta.shape[0]):
                print(f"File: {filepath}, Frame: {i}, Delta: {delta[i]}")

                data = h5_to_array(filepath, "/entry/mythen_nx/data")[
                    i, :, self.DEFAULT_COUNTER
                ]

                data[bad_channels] = 0

                split_module_data = np.split(data, n_modules)

                for n_mod in modules:
                    module_data = split_module_data[n_mod]

                    # if n_mod > 13:
                    #     module_data = np.flip(module_data)

                    module_array[i, n_mod, :] = module_data

            out_filepath = os.path.join(
                "/host-home/projects/outputs",
                filenumber + "_modules.h5",
            )
            out_filepaths.append(out_filepath)

            with h5pyFile(out_filepath, "w", libver="latest") as h5f:
                h5f["data1"] = module_array

            return out_filepaths

    def extract_module_dataset(self, module_to_analyse: int, delta_points):
        module_datasets = []

        for n_delta, _ in enumerate(delta_points):
            nxs_data = h5_to_array(
                self.module_dataset_filepath, "data1"
            )  # (delta, n_modules, PIXELS_PER_MODULE)

            module_data = nxs_data[n_delta, module_to_analyse, :]
            module_datasets.append(module_data)

        module_datasets = np.array(module_datasets)

        return module_datasets

    def average_within_tolerance(self, arr, tol):
        """
        For a 1D numpy array, if two adjacent values are within 'tol',
        replace them with their average
        and remove one of them, so the returned array is shorter.
        No explicit Python loops.
        """
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return arr

        arr = np.sort(arr)  # Ensure the array is sorted at the beginning

        # Find adjacent pairs within tolerance
        close = np.abs(arr[1:] - arr[:-1]) <= tol

        # Indices to keep: start with all True
        keep = np.ones(arr.shape, dtype=bool)
        # Where close, we'll keep only the first of the pair (set the second to False)
        keep[1:][close] = False

        # Compute averages for close pairs
        avgs = (arr[:-1][close] + arr[1:][close]) / 2

        # Output array: fill with arr, then replace the kept
        # indices that start a close pair with the average
        out = arr[keep]
        out_indices = np.where(close)[0][keep[:-1][close]]
        out[out_indices] = avgs

        return out

    def fit_peaks_across_delta(
        self,
        delta_points,
        module_angular_cal,
        modules,
        observed_reflections_in_tth,
        beamline_offset,
    ):
        fitted_peaks_for_modules = {}
        big_df = pd.DataFrame()

        for module_to_analyse in modules:
            module_dataset = self.extract_module_dataset(
                module_to_analyse=module_to_analyse, delta_points=delta_points
            )
            params = module_angular_cal[module_to_analyse]

            centre = params["centre"]
            conv = params["conv"]
            offset = params["offset"]

            module_pixel_number = np.arange(self.STRIPS_PER_MODULE, dtype=np.int64)

            raw_tth = channel_to_angle(
                module_pixel_number, centre, conv, offset, beamline_offset
            )

            tol = 0.01
            trim = 10

            calc_peak_tth = np.array([])
            detected_peak_pixel = np.array([])
            delta_of_point = np.array([])

            for n, (delta, dataset) in enumerate(
                zip(delta_points, module_dataset, strict=True)
            ):
                print(module_to_analyse, n, delta)

                dataset[0:trim] = np.nan
                dataset[len(dataset) - trim : :] = np.nan
                # dataset = dataset[trim:-trim]
                mask = dataset == 0

                real_tth = raw_tth + delta
                real_tth[0:trim] = np.nan
                real_tth[len(dataset) - trim : :] = np.nan

                dataset[mask] = np.nan
                real_tth[mask] = np.nan

                data_tth_mean = np.nanmean(real_tth)

                mintth, maxtth = np.nanmin(real_tth), np.nanmax(real_tth)

                tth_calculated_peak_centres = observed_reflections_in_tth[
                    (maxtth + tol > observed_reflections_in_tth)
                    & (observed_reflections_in_tth > mintth - tol)
                ]
                tth_calculated_peak_centres = np.sort(tth_calculated_peak_centres)

                if (
                    len(tth_calculated_peak_centres) > 0
                ):  # if there are peaks as detected by calculation from cif
                    if (len(tth_calculated_peak_centres) > 1) and self.single_peak:
                        middle_index = closest_indices(
                            data_tth_mean, tth_calculated_peak_centres
                        )
                        tth_calculated_peak_centres = np.array(
                            [tth_calculated_peak_centres[middle_index]]
                        )

                    non_nan_dataset = np.nan_to_num(dataset)

                    indexes = peakutils.indexes(
                        non_nan_dataset, thres=0.10, min_dist=100
                    )
                    tth_peaks_centres_in_data = real_tth[indexes]
                    pixel_peak_in_data = module_pixel_number[indexes]

                    n_calc, n_data = (
                        len(tth_calculated_peak_centres),
                        len(tth_peaks_centres_in_data),
                    )

                    if n_data > n_calc:
                        # if extra peaks are detected then clean it up
                        # by taking the closest ones
                        index = closest_indices(
                            tth_calculated_peak_centres, tth_peaks_centres_in_data
                        )
                        tth_peaks_centres_in_data = tth_peaks_centres_in_data[index]
                        pixel_peak_in_data = pixel_peak_in_data[index]
                        n_calc, n_data = (
                            len(tth_calculated_peak_centres),
                            len(tth_peaks_centres_in_data),
                        )

                        real_tth_no_nan = raw_tth + delta
                        tth_peaks_centres_in_data_refined = peakutils.interpolate(
                            real_tth_no_nan, non_nan_dataset, ind=pixel_peak_in_data
                        )

                        interp_func = interp1d(real_tth_no_nan, module_pixel_number)

                        try:
                            pixel_peak_in_data_refined = interp_func(
                                tth_peaks_centres_in_data_refined
                            )

                            if (
                                abs(
                                    tth_peaks_centres_in_data_refined
                                    - tth_peaks_centres_in_data
                                )
                                < 0.4
                            ):
                                pixel_peak_in_data = (
                                    pixel_peak_in_data_refined.flatten()
                                )
                                n_calc, n_data = (
                                    len(tth_calculated_peak_centres),
                                    len(tth_peaks_centres_in_data),
                                )
                            else:
                                continue

                        except Exception as e:
                            pixel_peak_in_data_refined = tth_peaks_centres_in_data
                            print(e)
                    try:
                        if (
                            abs(
                                tth_peaks_centres_in_data_refined  # type: ignore
                                - tth_peaks_centres_in_data
                            )
                            > 0.4
                        ):
                            continue
                    except Exception as e:
                        print(e)
                        continue

                    # if (
                    #     (n % 1 == 0)
                    #     and (n > 29)
                    #     and (n < 41)
                    #     and (module_to_analyse in [5])
                    # ):
                    #     plt.plot(real_tth, non_nan_dataset)
                    #     plt.scatter(
                    #         tth_peaks_centres_in_data,
                    #         non_nan_dataset[pixel_peak_in_data.astype(int)],
                    #         color="red",
                    #     )
                    #     plt.show()

                    if (
                        n_calc != n_data
                    ):  # if still not equal -  why? (peak probably on edge of data)
                        continue

                    calc_peak_tth = np.append(
                        calc_peak_tth, tth_calculated_peak_centres
                    )
                    detected_peak_pixel = np.append(
                        detected_peak_pixel, pixel_peak_in_data
                    )
                    delta_of_point = np.append(
                        delta_of_point, [delta] * len(tth_calculated_peak_centres)
                    )

                    continue

                else:  # if there are no peaks in this range skip
                    continue

            # if module_to_analyse in [5]:
            #     plt.ylabel("Intensity (A.U)")
            #     plt.xlabel("tth")

            module_data = pd.DataFrame()
            module_data["calc_peak_tth"] = calc_peak_tth
            module_data["pixel"] = detected_peak_pixel
            module_data["delta"] = delta_of_point
            module_data["module"] = module_to_analyse

            big_df = pd.concat((big_df, module_data))

            # print(len(module_data))

            # print(len(module_data))

            # mask = np.isclose(
            #     module_data["calc_peak_tth"].to_numpy()[:, None],  # shape (rows, 1)
            #     self.peaks_to_fit,  # shape (n,)
            #     rtol=1e-5,
            #     atol=1e-8,
            # ).any(axis=1)

            # mask = module_data["delta"] < 25

            # module_data = module_data[mask]

            # print(len(module_data))

            # module_data.to_csv(f"/workspaces/{module_to_analyse}.csv")

            # median_tth = np.median(calc_peak_tth)

            # module_data = module_data[
            #     (module_data["calc_peak_tth"] > median_tth - 0.2)
            #     & (module_data["calc_peak_tth"] < median_tth + 0.2)
            # ]

            # it's a dict
            fitted_peaks_for_modules[module_to_analyse] = module_data

        for peak in np.unique(big_df["calc_peak_tth"]):
            peak_data = big_df[big_df["calc_peak_tth"] == peak]

            print(peak, "(tth)")
            print(len(np.unique(peak_data["module"])), "\n")

        return fitted_peaks_for_modules

    def return_residual_for_modules(
        self,
        params: Parameters,
        modules: list[int],
        fitted_peaks_for_modules: dict[int, pd.DataFrame],
        ring_compare: bool = False,
        plot: bool = False,
    ):
        params_dict: dict = params.valuesdict()

        resid_for_all_modules = np.array([])

        for _, (module_to_analyse) in enumerate(modules):
            module_dataframe = fitted_peaks_for_modules[module_to_analyse]

            centre = params_dict[f"module_{module_to_analyse}_centre"]
            beamline_offset = params_dict["beamline_offset"]
            conv = params_dict[f"module_{module_to_analyse}_conv"]
            offset = params_dict[f"module_{module_to_analyse}_offset"]
            delta = module_dataframe["delta"].to_numpy()
            # delta_offset = params_dict[f"delta_offset_{module_to_analyse}"]
            module_pixel = module_dataframe["pixel"].to_numpy()

            self.module_conversion: type[ModuleConversion | ModuleConversion2D]

            if self.module_conversion is ModuleConversion:
                real_tth = self.module_conversion(
                    conv=conv,
                    offset=offset,
                    centre=centre,
                ).calculate_tth_for_pixel(
                    pixel_number=module_pixel, zero_offset=beamline_offset, delta=delta
                )
            elif self.module_conversion is ModuleConversion2D:
                real_tth = self.module_conversion(
                    radius=radius,
                    offset=offset,
                    tilt=tilt,
                ).calculate_tth_for_pixel(
                    pixel_number=module_pixel, zero_offset=beamline_offset, delta=delta
                )
            else:
                raise Exception("Must be ModuleConversion")

            diff = np.abs(real_tth - module_dataframe["calc_peak_tth"])
            # mmultiplying by mean weights the lower agnles greater than higher
            # excess = np.clip(diff - 0.002, 0, None)
            # resid_for_module = excess**2 * 1000

            # max_dist = 13.5

            # distance = abs(module_to_analyse - 14)
            # normalised = 1 - (distance / max_dist)

            # resid_for_module = diff * ((normalised) * 100)
            resid_for_module = diff

            resid_for_all_modules = np.append(resid_for_all_modules, resid_for_module)
            resid_for_module_iter = float(np.nansum(resid_for_module))
            # print(resid_for_module_iter)
            # print(module_to_analyse)
            self.resid_per_module[module_to_analyse].append(resid_for_module_iter)

            if plot and (self.plot_iter % 200 == 0):
                print(module_to_analyse)
                plt.scatter(real_tth, [1] * len(real_tth), label="det")
                plt.scatter(
                    module_dataframe["calc_peak_tth"],
                    [2] * len(module_dataframe),
                    label="calc",
                )
                plt.legend()
                plt.show()

        if ring_compare:
            for bad_mod in self.bad_modules:
                params[f"conv_{bad_mod}"] = self.module_angular_cal[bad_mod]["conv"]
                params[f"offset_{bad_mod}"] = self.module_angular_cal[bad_mod]["offset"]
                params[f"centre_{bad_mod}"] = self.module_centre

            pydantic_dict = self.results_dict_to_pydantic(params)
            angular_calibration = AngularCalibration(**pydantic_dict)

            config_file = "/host-home/projects/outputs/mythen_calibration/mythen3_reduction_config.toml"  # noqa
            settings1 = MythenSettings.load_from_toml(config_file)
            settings2 = MythenSettings.load_from_toml(config_file)

            bad_chan_file = "/workspaces/XRPD-Toolbox/config/i11/badchannels.txt"

            data_file = "/host-home/projects/outputs/angular_calibration/1410289.nxs"
            settings1.bad_channels_filepath = bad_chan_file
            settings2.bad_channels_filepath = bad_chan_file

            settings1.bad_modules = [
                11,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
            ]
            settings2.bad_modules = [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                17,
                27,
            ]  # type: ignore

            mythen3_ring_1 = MythenDetector(
                filepath=data_file,
                settings=settings1,
                angular_calibration=angular_calibration,
            )

            tth1, count1, error1 = mythen3_ring_1.generate_binned_xye(normalise=False)

            mythen3_ring_2 = MythenDetector(
                filepath=data_file,
                settings=settings2,
                angular_calibration=angular_calibration,
            )

            tth2, count2, error2 = mythen3_ring_2.generate_binned_xye(normalise=False)

            x_common, y1_interp, y2_interp = rebin_together(tth1, count1, tth2, count2)

            ring_compare = np.abs(y1_interp - y2_interp)
            ring_compare_resid = np.sum(ring_compare) / (1e9)

            resid_for_all_modules = resid_for_all_modules + ring_compare_resid

        print(np.sum(resid_for_all_modules), np.sum(ring_compare))
        self.plot_iter = self.plot_iter + 1

        return resid_for_all_modules

    def get_delta_points(self, filepath):

        return h5_to_array(filepath, "/entry/mythen_nx/delta")

    def save_results(
        self, results_dict, filepath, modules, bad_modules, original_ang_cal, p=0.05
    ):
        for key in results_dict.keys():
            if "conv" in key:
                print(key, results_dict[key])
            else:
                print(key, results_dict[key])

        with open(filepath, "w") as f:
            for module in modules:
                if module in bad_modules:
                    og_off = original_ang_cal[module]["offset"]
                    og_conv = original_ang_cal[module]["conv"]
                    og_centre = self.module_centre

                    f.write(
                        f"module {module} offset {og_off} conv {og_conv} center {og_centre} #not refined\n"  # noqa
                    )

                else:
                    off = results_dict[f"offset_{module}"]
                    conv = results_dict[f"conv_{module}"]
                    center = results_dict[f"centre_{module}"]

                    f.write(
                        f"module {module} offset {off} conv {conv} center {center} \n"
                    )

            beamline_offset = results_dict["beamline_offset"]

            f.write(f"beamline_offset {beamline_offset}")

        with open(filepath.replace(".off", ".json"), "w") as fp:
            json.dump(results_dict, fp, indent=4)

        print(f"Saved to: {filepath}")

    def plot_resids(self):
        for module in self.resid_per_module.keys():
            mod_resids = self.resid_per_module[module]
            plt.title(module)
            plt.plot(np.log10(mod_resids))
            plt.show()

    # def create_2d_starting_params(self):

    #     convs = calc_intial_module_conv(0.05 / 762)
    #     offsets = calc_starting_module_offset()

    #     params = Parameters()

    def create_starting_params(
        self, zero: float = -0.5, conv_tol: float = 0.2, offset_tol: float = 4
    ) -> Parameters:
        # conv_tol # fractional percent 0.1 = 10%
        # offset_tol = 4  # in degrees

        convs = calc_intial_module_conv(0.05 / 762)
        offsets = calc_starting_module_offset()

        params = Parameters()

        for mod in self.good_modules:
            init_conv = convs[mod]
            init_offset = offsets[mod]

            conv_lower = init_conv - ((abs(init_conv)) * conv_tol)
            conv_upper = init_conv + ((abs(init_conv)) * conv_tol)

            print(mod, init_offset, init_conv, conv_lower, conv_upper)

            params.add(
                f"module_{mod}_conv",
                vary=True,
                value=init_conv,
                min=conv_lower,
                max=conv_upper,
            )
            params.add(
                f"module_{mod}_offset",
                vary=True,
                value=init_offset,
                min=init_offset - offset_tol,
                max=init_offset + offset_tol,
            )

            params.add(
                f"module_{mod}_centre",
                value=self.module_centre,
                vary=False,
                min=self.module_centre - 2,
                max=self.module_centre + 2,
            )  # maybe 640 or 639.5?

        params.add("beamline_offset", value=zero, vary=True, min=-2, max=2)

        return params

    def plot_fit_stats(self, fitted_peaks_for_modules: dict):
        peak_fits = pd.DataFrame(
            columns=["peak"].extend(list(fitted_peaks_for_modules.keys()))
        )

        peak_fits["peak"] = self.peaks_to_fit

        for module in fitted_peaks_for_modules.keys():
            module_data = fitted_peaks_for_modules[module]
            plt.title(str(module))
            # plt.scatter(module_data["delta"], module_data["pixel"])

            mask = np.isclose(
                module_data["calc_peak_tth"].to_numpy()[:, None],  # shape (rows, 1)
                self.peaks_to_fit,  # shape (n,)
                rtol=1e-5,
                atol=1e-8,
            ).any(axis=1)

            module_data = module_data[mask]

            peak_data_gradient = []

            for peak in np.unique(module_data["calc_peak_tth"]):
                peak_in_module = module_data[module_data["calc_peak_tth"] == peak]
                plt.scatter(peak_in_module["delta"], peak_in_module["pixel"])

                m, b = np.polyfit(
                    peak_in_module["delta"].to_numpy(),
                    peak_in_module["pixel"].to_numpy(),
                    1,
                )
                print(m, b)
                peak_data_gradient.append(m)

                plt.plot(
                    peak_in_module["delta"],
                    (m * peak_in_module["delta"]) + b,
                    color="red",
                )
            peak_fits[str(module)] = peak_data_gradient
            plt.savefig(f"/host-home/projects/outputs/peak_fits_{module}.png")
            plt.close()

        mean_grads = []

        plt.figure(figsize=(16, 10))
        plt.title("Absolute Gradient of Fitted Peaks")
        for module in fitted_peaks_for_modules.keys():
            mean_gradient = np.mean(peak_fits[str(module)])
            print(module, mean_gradient)
            mean_grads.append(mean_gradient)

        # for module in fitted_peaks_for_modules.keys():
        #     gradients = peak_fits[str(module)]
        #     plt.title(module)
        #     plt.plot(self.peaks_to_fit, gradients)
        #     plt.show()

        plt.errorbar(
            list(fitted_peaks_for_modules.keys()),
            np.abs(mean_grads),
            np.std(np.abs(mean_grads)),
            fmt="-o",
        )
        plt.ylabel("Mean Gradient Of Peak Fit pixel/delta")
        plt.xlabel("Module number")
        plt.grid(True)
        plt.savefig("/host-home/projects/outputs/gradient.png")
        plt.show()
        plt.close()

        peak_fits.to_csv("/host-home/projects/outputs/peak_gradients.csv")

    def remove_bad_modules(self, fitted_peaks_for_modules: dict):
        for bad_module in self.bad_modules:
            fitted_peaks_for_modules.pop(bad_module)

        return fitted_peaks_for_modules

    def select_peaks(
        self,
        fitted_peaks_for_modules,
        mask_type: PEAK_MASK_LITERAL | None = "select_peaks",
    ):
        for module in fitted_peaks_for_modules.keys():
            module_data = fitted_peaks_for_modules[module]

            print(module, "unique peaks", np.unique(module_data["calc_peak_tth"]))
            print(len(module_data))

            if mask_type == "select_peaks":
                mask = np.isclose(
                    module_data["calc_peak_tth"].to_numpy()[:, None],  # shape (rows, 1)
                    self.peaks_to_fit,  # shape (n,)
                    rtol=1e-5,
                    atol=1e-8,
                ).any(axis=1)

            elif mask_type == "below":
                if module not in [12, 13, 14, 15, 16]:
                    mask = module_data["delta"] < 25
                else:
                    mask = module_data["delta"] < 50

            elif mask_type == "below2":
                distance = abs(module_distance(module))

                if distance > 0.7:
                    distance = distance
                else:
                    distance = 0

                mask = module_data["delta"] < 25 + (25 * distance)

            elif mask_type == "max":
                max_n = 5
                most_present_peaks = top_n_recurring(
                    module_data["calc_peak_tth"], max_n
                )
                mask = np.isclose(
                    module_data["calc_peak_tth"].to_numpy()[:, None],  # shape (rows, 1)
                    most_present_peaks,  # shape (n,)
                    rtol=1e-5,
                    atol=1e-8,
                ).any(axis=1)
            elif mask_type == "between":
                mask = (module_data["delta"] < self.upper_delta) & (
                    module_data["delta"] > self.lower_delta
                )
            else:
                mask = np.ones_like(module_data)  # if none don't mask any

            module_data = module_data[mask]
            print(len(module_data))
            # it's a dict
            fitted_peaks_for_modules[module] = module_data

        return fitted_peaks_for_modules

    def results_dict_to_pydantic(self, results_dict: dict):
        pydantic_dict = {}
        pydantic_dict["beamline_offset"] = results_dict["beamline_offset"]

        for module in self.active_modules:
            pydantic_dict[f"module_{str(module)}"] = {
                "centre": results_dict[f"module_{module}_centre"],
                "conv": results_dict[f"module_{module}_conv"],
                "offset": results_dict[f"module_{module}_offset"],
                # "delta_offset": results_dict.get(f"delta_offset_{module}") or 0,
            }

        return pydantic_dict

    def create_starting_params_from_original(self, starting_params, beamline_offset):
        conv_tol = 0.2  # fractional percent 0.1 = 10%
        offset_tol = 4  # in degrees

        params = Parameters()

        for mod in self.good_modules:
            init_conv = starting_params[mod]["conv"]
            init_offset = starting_params[mod]["offset"]

            conv_lower = init_conv - ((abs(init_conv)) * conv_tol)
            conv_upper = init_conv + ((abs(init_conv)) * conv_tol)

            print(mod, init_offset, init_conv, conv_lower, conv_upper)

            params.add(
                f"conv_{mod}",
                vary=True,
                value=init_conv,
                min=conv_lower,
                max=conv_upper,
            )
            params.add(
                f"offset_{mod}",
                vary=True,
                value=init_offset,
                min=init_offset - offset_tol,
                max=init_offset + offset_tol,
            )

            params.add(
                f"centre_{mod}",
                value=self.module_centre,
                vary=True,
                min=self.module_centre - 5,
                max=self.module_centre + 5,
            )  # maybe 640 or 639.5?

        params.add("beamline_offset", value=beamline_offset, vary=True, min=-2, max=2)

        return params


if __name__ == "__main__":
    # leastsq: Levenberg-Marquardt (default)
    # ’least_squares’: Least-Squares minimization, using Trust Region Reflective method
    # ’differential_evolution’: differential evolution
    # ’brute’: brute force method
    # ’basinhopping’: basinhopping
    # ’ampgo’: Adaptive Memory Programming for Global Optimization
    # ’nelder’: Nelder-Mead
    # ’lbfgsb’: L-BFGS-B
    # ’powell’: Powell
    # ’cg’: Conjugate-Gradient
    # ’newton’: Newton-CG
    # ’cobyla’: Cobyla
    # ’bfgs’: BFGS
    # ’tnc’: Truncated Newton
    # ’trust-ncg’: Newton-CG trust-region
    # ’trust-krylov’: Newton GLTR trust-region
    # ’trust-constr’: trust-region for constrained optimization
    # ’slsqp’: Sequential Linear Squares Programming
    # ’emcee’: Maximum likelihood via Monte-Carlo Markov Chain
    # ’shgo’: Simplicial Homology Global Optimization
    # ’dual_annealing’: Dual Annealing optimization

    # methods = ["leastsq", "least_squares", "differential_evolution", "brute",
    # "basinhopping", "ampgo", "nelder", "lbfgsb", "powell", "cg", "newton", "cobyla",
    # "bfgs", "tnc", "trust-ncg", "trust-exact", "trust-krylov","trust-constr",
    # "slsqp", "shgo", "dual_annealing"]

    fit_method = "leastsq"

    wavelength_in_ang = (
        0.828783  # 0.828773  # Angstrom - as refined by Eamonn on the MAC
    )

    # convs = []

    # for step in steps:
    lower_delta = 0
    upper_delta = 35

    data_dir = "/host-home/projects/outputs/angular_calibration/"
    si_nexus_file_numbers = [1410289]

    filepaths = generate_filepaths(data_dir, si_nexus_file_numbers)

    cal = AngularCalibrateMythen(
        filepaths=filepaths,
        wavelength_in_ang=wavelength_in_ang,
        module_centre=CENTRE,
        bad_modules=[17],
        lower_delta=lower_delta,
        upper_delta=upper_delta,
    )

    cal.get_selected_peaks(mask_type="below", plot_fit=False)

    cal.fit(fit_method=fit_method)
