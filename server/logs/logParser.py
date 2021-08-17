import numpy as np
import matplotlib.pyplot as plt

file = open("pyscheduler.log")

content = file.readlines()

lineIdx = 4
lineInc = 5

nLines = len(content)

epoch = []
accuracy = []
times = []
while (lineIdx < nLines):
    currLine = content[lineIdx]
    lineIdx += lineInc
    splitLine = currLine.split()
    epoch.append(float(splitLine[1]))
    accuracy.append(float(splitLine[3]))
    times.append(float(splitLine[5]))

cumtimes = np.cumsum(times)
print(max(cumtimes))
x_ticks = np.arange(0, 190, 5)
y_ticks = np.arange(0, 1, 0.1)
plt.plot(epoch, accuracy)
plt.xticks(x_ticks)
plt.xlabel('Epochs')
plt.yticks(y_ticks)
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.show()

x_ticks = np.arange(0, 4100, 500)
y_ticks = np.arange(0, 1, 0.1)
plt.plot(cumtimes, accuracy)
plt.xticks(x_ticks)
plt.xlabel('Time taken (sec)')
plt.yticks(y_ticks)
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time taken')
plt.show()