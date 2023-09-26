from pathlib import Path

import numpy as np

parent_dir = Path(__file__).parent
enu_coords = np.loadtxt(parent_dir / 'input/antennae.txt')
print(enu_coords)
