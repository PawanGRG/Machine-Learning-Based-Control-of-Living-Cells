import numpy as np
import matplotlib.pyplot as plt

# Sample data
variables = ['Untrained PPO', 'Trained PPO', 'PID', 'Relay']
means = [6956053022
, 2820298131
, 1367449412
, 794579143

]
# Same init
means = [997262549.3, 140006284.5, 85450182.25, 72942654.31]


std_devs = [1923697475
, 782253116.3
, 361530651.4
, 268620800.8
]

# Same init
std_devs = [126667743.8, 22059657.47, 0, 0]

plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'light'
# Set the bar colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Set the position of the bars on the x-axis
ind = np.arange(len(variables))

# Create the bar chart
fig, ax = plt.subplots()
bars = ax.bar(ind, means, yerr=std_devs, align='center', alpha=0.5, ecolor='black', capsize=10, color=['orange', 'red', 'green', 'cyan'])

# Set the y-axis label
ax.set_ylabel('Mean values')

# Set the x-axis ticks
ax.set_xticks(ind)

# Set the x-axis tick labels
ax.set_xticklabels(variables)
plt.grid()
# Set the title
ax.set_title('Same Initial Conditions')

# Display the graph
plt.show()