import matplotlib.pyplot as plt
import numpy as np

from xrpd_toolbox.i15_1.eiger_500k import Eiger500K

wavelength = 0.161699

poni = {
    "dist": 0.7,
    "poni1": 0.0,
    "poni2": 0.1,
    "rot1": 0.0,
    "rot2": 0.0,
    "rot3": 0.0,
    "pixel1": 7.5e-5,
    "pixel2": 7.5e-5,
    "wavelength": wavelength / 1e10,
}

eiger = Eiger500K(poni=poni)


def test_simulated_eiger():
    positions_in_tth = np.linspace(1, 20, 10)

    images, ais = eiger.simulate_data(
        positions_in_tth=positions_in_tth,
        calibrant_name="Si",
        wavelength_in_ang=wavelength,
    )

    assert images[0].shape == (1024, 512)
    assert images[0] > 0


def test_simulated_eiger_1d_scan():
    x, y = eiger.simulate_1d_pattern(
        positions_in_tth=np.linspace(1, 50, 10),
        calibrant_name="Si",
        wavelength_in_ang=wavelength,
    )
    # assert len(x) == 5
    # assert len(y) == 5

    plt.plot(x, y, marker="o")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Simulated 1D Pattern from Eiger Step Scan")
    plt.show()


if __name__ == "__main__":
    eiger.test()
