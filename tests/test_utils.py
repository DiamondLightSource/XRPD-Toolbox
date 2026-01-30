import os

import numpy as np
import scipy.integrate as integrate

from xrpd_toolbox.utils.energy import beam_energy_to_wavelength, tth_to_q
from xrpd_toolbox.utils.utils import (
    gaussian,
    get_filenumber_from_nxs,
    get_folder_paths,
    load_int_array_from_file,
    nexus_file_match,
    normalise,
    normalise_to,
)


def test_get_filenumber_from_nxs():
    filedir = "/dls/i11/test/cm12345-1/i11-99999.nxs"
    assert get_filenumber_from_nxs(filedir) == 99999


def test_nexus_file_match():
    filedir = "/dls/i15-1/test/cm12345-1/i15-1-99999.nxs"
    filename = os.path.basename(filedir)
    assert nexus_file_match(filename, beamline="i15-1") is not None


def test_normalise_to():
    normalised_array = normalise_to([1, 2, 4], minval=0)
    assert np.array_equal(normalised_array, [0.25, 0.5, 1.0])


def test_normalise():
    normalised_array = normalise([1, 2, 4])
    assert np.amax(normalised_array) == 1.0
    assert np.amin(normalised_array) == 0.0


def test_gaussian():
    x = np.linspace(0, 10, 100)
    y = gaussian(x, amp=22.0, cen=5.0, fwhm=1.0, background=0.0)
    assert len(y) == len(x)
    integral = integrate.simpson(y, x)
    assert np.isclose(integral, 22.0, atol=0.5)


def test_tth_to_q():
    tth = 30
    q_in_angstrom = tth_to_q(tth, 1)
    assert np.round(q_in_angstrom, 2) == 3.25


def test_beam_energy_to_wavelength():
    wavelength_in_angstrom = beam_energy_to_wavelength(12.34, unit="kev")
    assert round(wavelength_in_angstrom, 2) == 1.0


def test_load_int_array_from_file_returns_array_when_contains_ints():
    test_file = "int_array.txt"

    # Create a temporary file
    with open(test_file, "w") as f:
        for i in range(1, 6):
            f.write(f"{i}\n")

    # Test loading the array
    result = load_int_array_from_file(test_file)
    expected = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(result, expected)

    # Clean up
    os.remove(test_file)


def test_load_int_array_from_file_returns_none_when_file_doesnt_exist():
    test_file = "nob_existent.txt"

    # Test loading the array
    result = load_int_array_from_file(test_file)
    expected = np.array([])
    assert np.array_equal(result, expected)


def test_load_int_array_from_file_returns_none_when_file_empty():
    test_file = "int_array.txt"

    # Create a temporary file
    with open(test_file, "w") as f:
        f.write("")

    # Test loading the array
    result = load_int_array_from_file(test_file)
    expected = np.array([])
    assert np.array_equal(result, expected)

    # Clean up
    os.remove(test_file)


def test_get_folder_paths():
    list_of_paths = get_folder_paths("/")

    assert "/home" in list_of_paths
    assert isinstance(list_of_paths, list)


# def test_peak_bin_and_propagate_errors():
#     x1 = np.arange(0, 10, 0.1)
#     y1 = gaussian(x1, amp=10.0, cen=5.0, fwhm=1.0, background=0.1)
#     e1 = np.sqrt(y1)

#     x2 = np.arange(0.01, 10.01, 0.1)
#     y2 = gaussian(x2, amp=20.0, cen=5.0, fwhm=1.0, background=0.1)
#     e2 = np.sqrt(y2)

#     x_combined = np.concatenate((x1, x2))
#     y_combined = np.concatenate((y1, y2))
#     e_combined = np.concatenate((e1, e2))

#     binned_x, binned_y, binned_e = bin_and_propagate_errors(
#         x_combined, y_combined, e_combined, rebin_step=0.1, error_calc="internal"
#     )

#     assert len(binned_x) == len(binned_y) == len(binned_e)
#     assert np.amax(binned_y) > np.amax(y1)
#     assert np.amax(binned_y) > np.amax(y2)
