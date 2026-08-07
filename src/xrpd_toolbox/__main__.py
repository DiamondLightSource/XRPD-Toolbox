"""Interface for ``python -m xrpd_toolbox``."""

import click

from ._version import __version__

__all__ = ["main"]


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, message="%(version)s")
@click.pass_context
def main(ctx: click.Context) -> None:
    """xrpd_toolbox command line interface."""
    pass


@main.command(name="bad_pixel_gui")
@click.pass_context
def bad_pixel_gui(ctx: click.Context) -> None:
    """Launch the bad pixel GUI."""

    from xrpd_toolbox.gui.bad_pixel_gui import run_bad_pixel_gui

    run_bad_pixel_gui()


@main.command(name="mythen_process_gui")
@click.pass_context
def mythen_process_gui(ctx: click.Context) -> None:
    """Launch the bad pixel GUI."""

    from xrpd_toolbox.gui.mythen3_process_gui import run_mythen_process

    run_mythen_process()


@main.command(name="mythen_alignment")
@click.pass_context
def mythen_alignment(ctx: click.Context) -> None:
    """Launch the bad pixel GUI."""

    from xrpd_toolbox.i11.diagnostic import DetectorAlignmentSimulator

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


if __name__ == "__main__":
    main()
