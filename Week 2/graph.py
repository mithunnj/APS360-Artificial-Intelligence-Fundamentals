import matplotlib.pyplot as plt
import numpy as np

# Class averages, my grades and total number of classes taken
average = np.array([73, 67, 73, 70, 73, 73, 70, 70, 67, 70, 70, 73, 67, 67, 67, 77, 70, 77, 77, 73, 70, 67, 63, 77, 70, 70, 73, 70, 77])
grade = np.array([66, 41, 17, 58, 38, 79, 69, 65, 70, 68, 68, 76, 54, 65, 43, 74, 84, 90, 87, 76, 66, 70, 60, 80, 72, 61, 63, 61, 72])
num_classes = np.array(range(1, len(grade)+1))

m_periodA, b_periodA = np.polyfit(num_classes[5:21], grade[5:21], 1) # Line of best fit for the period when I was on repeat probation & I needed to satisfy base requirements
m_periodB, b_periodB = np.polyfit(num_classes[21:], grade[21:], 1) # Line of best fit for the period that I was regular student status, and I joined aUTodrive team
m_average, b_average = np.polyfit(num_classes, average, 1) # Line of best fit for the global class averages

plt.plot(average, 'o')
plt.plot(grade, 'o')

# Plot averages over the specific time periods. 
plt.plot(num_classes[5:21], m_periodA*num_classes[5:21] + b_periodA, color="green") 
plt.plot(num_classes[21:], m_periodB*num_classes[21:] + b_periodB, color="orange")
plt.plot(num_classes, m_average*num_classes + b_average, color="red")
plt.xlabel('# of classes taken')
plt.ylabel('Grade (out of 100)')

# Label important periods of time: Repeat probabtion period (3 semesters), aUTodrive/regular student status
plt.axvline(x=5, linestyle='--', color="green")
plt.axvline(x=20, linestyle='--', color="green")
plt.axvline(x=21, linestyle='--', color="orange")
plt.axvline(x=len(grade)+1, linestyle='--', color="orange")

plt.show()