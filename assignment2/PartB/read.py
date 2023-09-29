from pathlib import Path
from json import load

import numpy as np

input_dir = Path(__file__).parent / 'input'

enu_coords = np.loadtxt(input_dir / 'antennae.txt')

with open(input_dir / 'configurations.json', "rb") as file:
    config = load(file)

print(config)

