from pathlib import Path

import numpy as np

input_dir = Path(__file__).parent / 'input'
enu_coords = np.loadtxt(input_dir / 'antennae.txt')
# print(enu_coords)
