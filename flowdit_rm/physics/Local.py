import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

png_path="/workspace/PPData5D-success/png/06.DenseUrban/T06C0D0002_n00_f00_ss_z00.png"
# 1. Load matrix data
# Assume the file is a txt matrix separated by spaces or commas
matrix = plt.imread(png_path) 

# 2. Use local peak detection to find transmitters
# Parameter notes:
# Note translated to English.
# threshold_abs: absolute threshold，only peaks with signal strength greater than this value“peak”are considered transmitters，used to filter ambient noise or small distant fluctuations
tx_coordinates = peak_local_max(matrix, min_distance=8, threshold_abs=0.5) 
# Note translated to English.

# Print the number and coordinates of detected transmitters
num_tx = len(tx_coordinates)
print(f"Detected {num_tx} transmitters。")

tx_powers = []
for idx, coord in enumerate(tx_coordinates):
    y, x = coord
    power = matrix[y, x]
    tx_powers.append(power)
    print(f"Transmitter {idx+1}: coordinate(x={x}, y={y}), center signal strength={power:.4f}")

# 3. Visual validation; running this step is strongly recommended
plt.figure(figsize=(8, 8))
plt.imshow(matrix, cmap='jet')
plt.colorbar(label='Signal Strength')

# Note translated to English.
plt.plot(tx_coordinates[:, 1], tx_coordinates[:, 0], 'r*', markersize=15, label='Detected Tx')
plt.title(f"Detected {num_tx} Transmitters via Local Peak")
plt.legend()
plt.savefig("Local_T06C0D0002_n00_f00_ss_z00.png", dpi=300, bbox_inches="tight")