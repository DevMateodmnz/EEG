"""Report the versions of the project's direct Python dependencies."""

import matplotlib
import mne
import numpy
import sklearn


def main() -> None:
    """Print dependency versions used by this environment."""
    print(f"MNE: {mne.__version__}")
    print(f"NumPy: {numpy.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")


if __name__ == "__main__":
    main()
