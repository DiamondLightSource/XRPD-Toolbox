import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted

from xrpd_toolbox.utils.utils import h5_to_array

folder = Path("/dls/staging/dls/i15-1/data/2025/cy40172-1")


def load_xy(file: str):

    tth, inten = np.genfromtxt(file, unpack=True)
    return tth, inten


def get_files(folder: Path, run_number: int) -> list[str]:

    reprocessed = folder / "reprocessed/tth_pe2"
    xy_files = [
        str(reprocessed / f)
        for f in os.listdir(str(reprocessed))
        if f.endswith(".xy") and str(run_number) in str(f) and not "norm" in f
    ]
    xy_files = natsorted(xy_files)
    return xy_files


def get_i0(run_number: int):

    reprocessed = folder / f"i15-1-{run_number}_reprocessed.nxs"

    if not reprocessed.exists():
        raise FileNotFoundError()

    return h5_to_array(reprocessed, "/entry/i0/data")


run_numbers = [83735, 83736]  # , 83736]


for run_number in run_numbers:
    xy_files = get_files(folder, run_number)

    i0_values = get_i0(run_number)

    for file, i0 in zip(xy_files, i0_values, strict=False):
        new_name = file.replace("det2", "det2_norm")

        tth, inten = np.genfromtxt(file, unpack=True)

        norm_inten = inten / i0

        np.savetxt(
            new_name, np.stack((tth, norm_inten), axis=-1), fmt="%.5f", delimiter="\t"
        )

        scratch_file = str(Path("/scratch/cy40172-1") / Path(new_name).name)

        print(file, i0, new_name, scratch_file)

        np.savetxt(
            scratch_file,
            np.stack((tth, norm_inten), axis=-1),
            fmt="%.5f",
            delimiter="\t",
        )
