import matplotlib.pyplot as plt
import numpy as np
from math import pi


designs = ['multiplier', 'div', 'rc64b', 'rc256b', 'c7552', 'c6288', 'router', 'voter']
methods = ['ABC', 'ABC-Unlimited', 'SLAP', 'LEAP']
categories = ['Delay', 'Area', 'Cuts']


delays = [[4649.10, 4144.31, 4278.48, 4075.31],
          [66126.44, 68530.74, 80775.56, 77108.3],
          [1844.59, 1688.38, 1565.70, 1688.38],
          [7388.03, 6861.76, 6807.02, 6861.76],
          [817.46, 828.46, 800.09, 787.96],
          [1265.88, 1228.32, 1236.59, 1250.44],
          [471.04, 476.92, 538.11, 473.82],
          [1189.56, 1201.89, 1242.51, 1210.45]]

areas = [[25458.31, 24168.27, 24021.07, 24860.42],
         [60509.10, 59953.43, 62818.57, 54247.63],
         [450.7, 601.16, 602.33, 601.16],
         [1794.39, 2482.33, 2428.21, 2482.33],
         [2045.40, 1905.90, 2002.01, 1939.96],
         [3000.45, 3014.68, 3023.54, 2951.93],
         [242.14, 211.35, 238.65, 213.22],
         [19816.2, 19497.54, 17517.46, 18653.54]]

cuts = [[833565, 1484104, 892650, 264185],
        [1305640, 1550747, 1137884, 573851],
        [5491, 10066, 6397, 2900],
        [22387, 40978, 30476, 11732],
        [52150, 93745, 30933, 17658],
        [95519, 197022, 111609, 21741],
        [3026, 4529, 2369, 1686],
        [341522, 532922, 407595, 118069]]

# Initialize arrays for scaled values
scaled_delays = np.zeros((len(designs), len(methods)))
scaled_areas = np.zeros((len(designs), len(methods)))
scaled_cuts = np.zeros((len(designs), len(methods)))

# Scale the data
for i in range(len(scaled_delays)):
    scaled_delays[i][0] = 1
    for j in range(1, 4):
        scaled_delays[i][j] = delays[i][j] / delays[i][0]

for i in range(len(scaled_areas)):
    scaled_areas[i][0] = 1
    for j in range(1, 4):
        scaled_areas[i][j] = areas[i][j] / areas[i][0]

for i in range(len(scaled_cuts)):
    scaled_cuts[i][0] = 1
    for j in range(1, 4):
        scaled_cuts[i][j] = cuts[i][j] / cuts[i][0]

# Create data for radar chart
data = []
for i, design in enumerate(designs):
    data.append([design, *scaled_delays[i], *scaled_areas[i], *scaled_cuts[i]])

# Plot radar charts for each method
fig, axs = plt.subplots(2, 2, subplot_kw=dict(polar=True), figsize=(14, 10))

for idx, method in enumerate(methods):
    ax = axs[idx // 2, idx % 2]
    for i, design in enumerate(designs):
        values = [scaled_delays[i][idx], scaled_areas[i][idx], scaled_cuts[i][idx]]
        values += values[:1]  # To close the circle
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=design)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(f'{method}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.suptitle('Comparison of Metrics Across Designs and Methods')
plt.tight_layout()
plt.savefig('radar_plot.png', dpi=300)
plt.show()
