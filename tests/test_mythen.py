import math
import os
from pathlib import Path

import numpy as np
import pytest

from xrpd_toolbox import BASE_PATH
from xrpd_toolbox.core import XYEData
from xrpd_toolbox.i11.mythen import (
    CENTRE,
    DEFAULT_BAD_CHANS,
    MODULES_IN_DETECTOR,
    MYTHEN_PIXEL_SIZE,
    PIXEL_NUMBER,
    PSD_RADIUS,
    BadChannels,
    ModuleConversion,
    ModuleConversion2D,
    MythenDetector,
    MythenSettings,
)
from xrpd_toolbox.utils.mythen_utils import (
    calc_intial_module_conv,
    calc_starting_module_offset,
    channel_to_angle,
    channel_to_angle_2d_offset_rotation_centre,
)

CONFIG_FILE = (
    Path(__file__).parent.parent / "config" / "i11" / "mythen3_reduction_config.toml"
)

LEGANCY_CONIG = Path(__file__).parent.parent / "config" / "i11" / "legacy_config.toml"


SI_DATA_FILE = BASE_PATH.parent.parent / "tests" / "data" / "1410696.nxs"


@pytest.fixture
def mythen_settings():
    mythen_settings = MythenSettings(
        active_modules=[1, 2, 3],
        bad_modules=[4, 5],
        bad_channel_masking=True,
        flatfield_filepath="flatfield.h5",
        apply_flatfield=False,
        modules_in_flatfield=[1, 2],
        send_to_ispyb=False,
        rebin_step=0.004,
        default_counter=0,
        edge_bad_channels=10,
        error_calc="poisson",
        data_reduction_mode="step_scan",
        bad_channels_filepath="bad_channels.txt",
        angcal_filepath="angcal.txt",
    )

    return mythen_settings


def test_mythen_settings(mythen_settings: MythenSettings):
    assert mythen_settings.data_reduction_mode == "step_scan"
    assert mythen_settings.rebin_step == 0.004


def test_mythen_settings_load_from_toml():
    settings = MythenSettings.load_from_toml(CONFIG_FILE)

    assert isinstance(settings, MythenSettings)


def test_mythen_toml_save_load(mythen_settings: MythenSettings):
    file_path = "file.toml"

    mythen_settings.save_to_toml(file_path)
    loaded_mythen_settings = MythenSettings.load_from_toml(file_path)

    assert mythen_settings == loaded_mythen_settings

    os.remove(file_path)


def test_mythen_yaml_save_load(mythen_settings: MythenSettings):
    file_path = "file.yaml"

    mythen_settings.save_to_yaml(file_path)
    loaded_mythen_settings = MythenSettings.load_from_yaml(file_path)

    assert mythen_settings == loaded_mythen_settings

    os.remove(file_path)


def test_mythen_load_fails_when_incorrect_file_extension(
    mythen_settings: MythenSettings,
):
    file_path = "file.txt"

    with pytest.raises(ValueError):
        mythen_settings.save_to_yaml(file_path)

    with pytest.raises(ValueError):
        mythen_settings.save_to_toml(file_path)


def test_mythen_data_reduction():

    detector = MythenDetector(filepath=SI_DATA_FILE, filename_suffix="_test")

    detector.process_step_scan(control=False)

    assert str(detector.output_directory) == str(SI_DATA_FILE.parent / "processed")
    assert Path(detector.xye_filepath_out).exists()
    assert str(Path(detector.xye_filepath_out).parent) == str(
        SI_DATA_FILE.parent / "processed"
    )

    _ = XYEData.from_csv(detector.xye_filepath_out)

    os.remove(detector.xye_filepath_out)


def test_data_reduction_mode_validation():
    mythen_settings = MythenSettings(
        active_modules=[1, 2, 3],
        bad_modules=[4, 5],
        bad_channel_masking=True,
        flatfield_filepath="flatfield.h5",
        apply_flatfield=False,
        modules_in_flatfield=[1, 2],
        send_to_ispyb=False,
        rebin_step=0.004,
        default_counter=0,
        edge_bad_channels=10,
        error_calc="internal",  # type: ignore
        data_reduction_mode=0,  # type: ignore
        bad_channels_filepath="bad_channels.txt",
        angcal_filepath="angcal.off",
    )

    assert mythen_settings.data_reduction_mode == "step_scan"
    assert str(mythen_settings.angcal_filepath).endswith(".json")


def test_legacy_toml():
    legacy_setting = MythenSettings.load(LEGANCY_CONIG)
    assert legacy_setting


def test_add_bad_channel():

    bad_channels = BadChannels(DEFAULT_BAD_CHANS, 0, 28)

    assert not all(bad_channels.masks[5][0:256])

    bad_channels.add_bad_channel_to_module(5, np.arange(0, 256, 1, dtype=int))

    assert all(bad_channels.masks[5][0:256])

    bad_channels_in_masks = 0
    for module in range(MODULES_IN_DETECTOR):
        bad_channels_in_masks = bad_channels_in_masks + len(
            np.argwhere(bad_channels.masks[module])
        )

    assert len(bad_channels.bad_channels) == bad_channels_in_masks


@pytest.mark.parametrize(
    "module",
    list(range(28)),
)
def test_2d_and_1d_module_conversion_get_same_result_when_untilted(module: int):

    convs = calc_intial_module_conv(MYTHEN_PIXEL_SIZE / PSD_RADIUS)
    offsets = calc_starting_module_offset()

    conv = convs[module]
    pixel_direction = int(math.copysign(1, conv))

    zero_offset = 0
    delta = 5

    module_conv_psi = ModuleConversion(
        conv=conv,
        module_angle=offsets[module],
        centre=CENTRE,
    )

    psi_tth = module_conv_psi.calculate_tth_for_pixel(
        pixel_number=PIXEL_NUMBER, zero_offset=zero_offset, delta=delta
    )

    module_conv_2d = ModuleConversion2D(
        radius=PSD_RADIUS,
        module_angle=offsets[module],
        pixel_direction=pixel_direction,
        tilt=0,
        centre=CENTRE,
        pixel_size=MYTHEN_PIXEL_SIZE,
    )

    tth_2d = module_conv_2d.calculate_tth_for_pixel(
        pixel_number=PIXEL_NUMBER, zero_offset=zero_offset, delta=delta
    )

    for psi, new in zip(psi_tth, tth_2d, strict=True):
        assert psi == pytest.approx(new, abs=1e-12)


@pytest.mark.parametrize(
    "module",
    list(range(28)),
)
def test_channel_to_angle_2d_with_offset_rotation_centre_get_same_result_when_untilted(
    module: int,
):

    convs = calc_intial_module_conv(MYTHEN_PIXEL_SIZE / PSD_RADIUS)
    module_angles = calc_starting_module_offset()

    conv = convs[module]
    module_angle = module_angles[module]

    pixel_direction = int(math.copysign(1, conv))

    zero_offset = 0
    delta = 5

    psi_tth = (
        channel_to_angle(
            pixel_number=PIXEL_NUMBER,
            centre=CENTRE,
            conv=conv,
            module_angle=module_angle,
            zero_offset=zero_offset,
        )
        + delta
    )

    tth_2d = channel_to_angle_2d_offset_rotation_centre(
        pixel_number=PIXEL_NUMBER,
        centre=CENTRE,
        pixel_size=MYTHEN_PIXEL_SIZE,
        radius=PSD_RADIUS,
        module_angle=module_angle,
        tilt=0,
        zero_offset=zero_offset,
        pixel_direction=pixel_direction,
        delta=delta,
    )

    for psi, new in zip(psi_tth, tth_2d, strict=True):
        assert psi == pytest.approx(new, abs=1e-12)


# def test_mythen_angular_calculations_are_correct():
#     raise NotImplementedError()
