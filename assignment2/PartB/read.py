from pathlib import Path

import numpy as np

input_dir = Path(__file__).parent / 'input'

enu_coords = np.loadtxt(input_dir / 'antennae.txt')

config = {}
with open(input_dir / 'configurations.txt', "r") as file:
    for line in file:
        key, *values = line.strip().split()
        config[key] = np.array(values, dtype=np.float_) if len(values) > 1 else float(values[0])

print(config)

