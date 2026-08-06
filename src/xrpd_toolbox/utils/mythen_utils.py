from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt

# def channel_to_angle(ich, off, r, c, dir=1, p=0.05):
#     """
#     ich: channel number, 0-1280
#     off: module offset, degrees
#     r: radius, mm
#     c: center (in pixel or mm?)
#     dir: direction, 1
#     p: pixel size, mm
#     """
#     # print(off)
#     if r < 0:
#         ich = 1279 - ich
#     return off + np.degrees(
#         c * p / np.abs(r) - dir * np.arctan(p * (ich - c) / np.abs(r))
#     )

# def angle_to_channel(ang, off, r, c, dir=1, p=0.05):
#     ich = (
#         np.tan(dir * (np.radians(ang - off) - c * p / np.abs(r))) * np.abs(r) / p
#         + c
#     )
#     if r > 0:
#         return ich
#     else:
#         return 1279 - ich


def channel_to_angle(
    pixel_number: npt.NDArray[np.int_],
    centre: int | float,
    conv: int | float,
    module_angle: int | float,
    zero_offset: int | float,
) -> np.ndarray:
    module_conversions = pixel_number - centre
    module_conversions = module_conversions * conv
    module_conversions = np.arctan(module_conversions)
    raw_tth = module_angle + np.rad2deg(module_conversions) + zero_offset

    return raw_tth


def channel_to_angle_2d(
    pixel_number: npt.NDArray[np.integer],
    centre: float,
    pixel_size: float,
    radius: float,
    module_angle: float,
    tilt: float,
    zero_offset: float,
    pixel_direction: int,
) -> npt.NDArray[np.float64]:
    """
    Convert pixel number to two-theta using full 2D module geometry.

    radius, module_angle define where the module's centre pixel sits
    in real space (polar coords about the sample). tilt is the module's
    own rotation away from perfectly tangential, so its effect on tth
    grows non-linearly across the module (unlike a flat offset).
    pixel_direction is +1/-1 for whether increasing pixel number walks
    tangentially clockwise or counter-clockwise (the sign that used to
    live in the legacy `conv` term).
    """
    displacement_mm: npt.NDArray[np.float64] = (
        pixel_number.astype(np.float64) - centre
    ) * pixel_size

    module_angle_rad: float = np.deg2rad(module_angle)
    module_x: float = radius * np.cos(module_angle_rad)
    module_y: float = radius * np.sin(module_angle_rad)

    direction_rad: float = (
        module_angle_rad + pixel_direction * np.pi / 2 + np.deg2rad(tilt)
    )

    x: npt.NDArray[np.float64] = module_x + displacement_mm * np.cos(direction_rad)
    y: npt.NDArray[np.float64] = module_y + displacement_mm * np.sin(direction_rad)

    raw_tth: npt.NDArray[np.float64] = np.rad2deg(np.arctan2(y, x)) + zero_offset
    return raw_tth


def channel_to_angle_2d_offset_rotation_centre(
    pixel_number: npt.NDArray[np.integer],
    centre: float,
    pixel_size: float,
    radius: float,
    module_angle: float,
    tilt: float,
    zero_offset: float,
    pixel_direction: int,
    delta: float | npt.NDArray[np.float64],
    rotation_centre_x: float = 0.0,
    rotation_centre_y: float = 0.0,
) -> npt.NDArray[np.float64]:
    """
    rotation_centre_x/y locate the arc's mechanical pivot in mm,
    relative to the sample position at the origin. If the pivot
    coincides with the sample (0, 0), rotating by delta is a pure
    angular shift and this reduces to the old `raw_tth + delta`
    behaviour. If it doesn't, the module's distance from the sample
    changes with delta too, which this rotation captures correctly.
    """
    displacement_mm = (pixel_number.astype(np.float64) - centre) * pixel_size

    module_angle_rad = np.deg2rad(module_angle)
    module_x = radius * np.cos(module_angle_rad)
    module_y = radius * np.sin(module_angle_rad)

    direction_rad = module_angle_rad + pixel_direction * np.pi / 2 + np.deg2rad(tilt)

    x = module_x + displacement_mm * np.cos(direction_rad)
    y = module_y + displacement_mm * np.sin(direction_rad)

    # rotate each pixel's position about the arc's true pivot, not the origin
    delta_rad = np.deg2rad(delta)
    cos_d = np.cos(delta_rad)
    sin_d = np.sin(delta_rad)

    dx = x - rotation_centre_x
    dy = y - rotation_centre_y

    x_rot = rotation_centre_x + dx * cos_d - dy * sin_d
    y_rot = rotation_centre_y + dx * sin_d + dy * cos_d

    raw_tth = np.rad2deg(np.arctan2(y_rot, x_rot)) + zero_offset
    return raw_tth


def channel_to_angle_in_real_units(
    pixel_number: npt.NDArray[np.int_],
    centre: int | float,
    offset: int | float,
    beamline_offset: int | float,
    radius: int | float = 762,
    p: float = 0.05,
) -> np.ndarray:
    """
    pixel_number: channel number, usually 0-1280
    centre: centre (in pixel number - ie 1280/2)
    offset: module offset, degrees
    radius: radius, mm - approx 760
    direction: 1 or -1 depending if module is flipped or not
    p: pixel size, mm = 0.05
    """

    raw_tth = channel_to_angle(
        pixel_number, centre, (p / radius), offset, beamline_offset
    )

    return raw_tth


def calc_intial_module_conv(conv=6.5e-05) -> dict[int, float]:
    module_conv_dict = {}

    for mod in range(28):
        if mod > 13:
            module_conv_dict[mod] = -conv
        else:
            module_conv_dict[mod] = conv

    return module_conv_dict


def paired_modules():
    """
    Given a list of module numbers, return a list of (a, b) pairs such that
    a and b are paired as described: 0-27, 1-26, 2-25, ..., 13-14.
    Only pairs where both a and b are in the input list are returned.
    """

    modules = list(range(28))

    modules = np.array(modules)
    n = modules.max()
    pairs = []
    for m in modules:
        pair = n - m
        if pair in modules and m <= pair:
            pairs.append((int(m), int(pair)))

    pairs = np.array(pairs)

    return pairs


def find_pair(mod: int) -> int:
    modules_array = paired_modules()

    row, col = np.where(modules_array == mod)
    if len(row) == 0:
        raise ValueError(f"Module {mod} is not in the paired modules array.")
    return int(modules_array[row[0], 1 - col[0]])


def calc_starting_module_offset(initial_module=0.45, offset=2.5) -> dict[int, float]:
    """Used for calculatign the intial centres of each of the modules"""

    module_pairs = paired_modules()
    module_offsets_dict = {}

    for n, module_pair in enumerate(module_pairs[::-1]):
        print(module_pair)

        ring_2_cen = (n * 5) + initial_module
        ring_1_cen = ring_2_cen + offset

        module_offsets_dict[int(module_pair[1])] = ring_2_cen
        module_offsets_dict[int(module_pair[0])] = ring_1_cen

    print(module_offsets_dict)

    return module_offsets_dict


def calc_starting_module_centre(initial_module=0.45, offset=2.5):
    """Used for calculatign the intial centres of each of the modules"""

    module_pairs = paired_modules()
    module_centres_dict = {}

    for n, module_pair in enumerate(module_pairs[::-1]):
        print(module_pair)

        ring_2_cen = (n * 5) + initial_module
        ring_1_cen = ring_2_cen + offset

        module_centres_dict[int(module_pair[1])] = ring_2_cen
        module_centres_dict[int(module_pair[0])] = ring_1_cen

    print(module_centres_dict)

    return module_centres_dict


def read_config(mythen3_config_filepath: str | Path) -> list[int]:
    """
    reads the config file used by SLSDet and works out what modules are currently active
    """

    enabled_modules_hostnames = []

    with open(mythen3_config_filepath) as file:
        lines = [line.rstrip() for line in file]

    for _, line in enumerate(lines):
        if line.startswith("hostname"):
            enabled_modules_hostnames = line.split()[1::]

    enabled_modules = [
        int(n_mod.rstrip()[-3::]) - 100 for n_mod in enabled_modules_hostnames
    ]

    return enabled_modules


def modules_to_pixels(modules: int | Iterable[int]):
    if isinstance(modules, int):
        pixels = slice(modules * 1280, (modules + 1) * 1280, None)
    elif isinstance(modules, Iterable):
        pixels = np.concatenate([np.arange(i * 1280, (i + 1) * 1280) for i in modules])
    else:
        raise TypeError("Must be int or iterable of ints")

    return pixels


def read_singular_angcal_files(angcal_filepath: str) -> tuple[dict, float]:
    """

    Reads a single of ang.off files and returns a dict with the
    each modules anngular calibrations contains within a dict

    each module dict contains "offset", "conv" and "centre"

    eg. self.module_angular_cal[module]["offset"]

    """

    module_angular_cal = {}
    beamline_offset = 0

    with open(angcal_filepath) as f:
        for line in f:
            if "beamline_offset" in line:
                elements = line.split()
                beamline_offset = float(elements[1])

            elif line := line.strip():
                elements = line.split()
                module_cal = {}

                (
                    module_in_file,
                    module_cal["offset"],
                    module_cal["conv"],
                    module_cal["centre"],
                ) = (
                    int(elements[1]),
                    float(elements[3]),
                    float(elements[5]),
                    float(elements[7]),
                )

                module_angular_cal[module_in_file] = module_cal

    return module_angular_cal, beamline_offset
