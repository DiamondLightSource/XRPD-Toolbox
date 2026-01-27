import subprocess
import sys

from XRPD_Toolbox import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "XRPD_Toolbox", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
