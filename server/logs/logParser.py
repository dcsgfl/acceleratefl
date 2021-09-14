import numpy as np
import matplotlib.pyplot as plt

file = open("pxy_13_1.log")

content = file.readlines()

lineIdx = 0
nLines = 170

epoch = []
accuracy = []
times = []
while (lineIdx < nLines):
    currLine = content[lineIdx]
    splitLine = currLine.split()
    epoch.append(float(splitLine[1]))
    accuracy.append(float(splitLine[3]))
    times.append(float(splitLine[17]))
    lineIdx += 1

cumtimes = np.cumsum(times)
print(max(cumtimes))
x_ticks = np.arange(0, 180, 5)
y_ticks = np.arange(0, 1, 0.1)
plt.plot(epoch, accuracy)
plt.xticks(x_ticks)
plt.xlabel('Epochs')
plt.yticks(y_ticks)
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.show()

x_ticks = np.arange(0, max(cumtimes), 1000)
y_ticks = np.arange(0, 1, 0.1)
plt.plot(cumtimes, accuracy)
plt.xticks(x_ticks)
plt.xlabel('Time taken (sec)')
plt.yticks(y_ticks)
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time taken')
plt.show()