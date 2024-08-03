import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

designs = ['multiplier', 'div', 'rc64b', 'rc256b', 'c7552', 'c6288', 'router', 'voter']
methods = ['ABC', 'ABC-Unlimited', 'SLAP', 'LEAP']
colors = ['b', 'g', 'r', 'c']


np.random.seed(0)
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

scaled_delays = np.random.rand(len(designs), len(methods))
scaled_areas = np.random.rand(len(designs), len(methods))
scaled_cuts = np.random.rand(len(designs), len(methods))

for i in range(len(scaled_delays)):
        scaled_delays[i][0] = 1
        for j in range(1,4):
            scaled_delays[i][j] = delays[i][j]/delays[i][0]

for i in range(len(scaled_areas)):
        scaled_areas[i][0] = 1
        for j in range(1,4):
            scaled_areas[i][j] = areas[i][j]/areas[i][0]

for i in range(len(scaled_cuts)):
        scaled_cuts[i][0] = 1
        for j in range(1,4):
            scaled_cuts[i][j] = cuts[i][j]/cuts[i][0]


fig, ax = plt.subplots(figsize=(10, 7))

bar_width = 0.2
index = np.arange(len(designs))

for i, method in enumerate(methods):
    ax.bar(index + i*bar_width, scaled_cuts[:, i], bar_width, label=method, color=colors[i])

ax.set_xlabel('Designs', fontsize=18, fontweight='bold')
ax.set_ylabel('Normalised Number of Cuts', fontsize=18, fontweight='bold')
# ax.set_title('Delay Comparison', fontsize=20, fontweight='bold')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(designs, fontsize=16, fontweight='bold')
font_properties = FontProperties(weight='bold', size=16)
for label in ax.get_yticklabels():
    label.set_fontproperties(font_properties)
ax.legend(fontsize=14)

plt.tight_layout()
plt.savefig('cuts_bar_chart.png', dpi=300)