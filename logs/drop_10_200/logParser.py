import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.polynomial import poly
from scipy.signal import savgol_filter

pyfh = open("drop_py_10_200.log")
tiflfh = open("drop_tifl_10_200.log")

pycontent = pyfh.readlines()
tiflcontent = tiflfh.readlines()

lineIdx = 0
nLines = 200

pyepoch = []
pyaccuracy = []
pytimes = []

tiflepoch = []
tiflaccuracy = []
tifltimes = []
while (lineIdx < nLines):
    currLine = pycontent[lineIdx]
    splitLine = currLine.split()
    pyepoch.append(float(splitLine[1]))
    pyaccuracy.append(float(splitLine[3]))
    pytimes.append(float(splitLine[17]))

    currLine = tiflcontent[lineIdx]
    splitLine = currLine.split()
    tiflepoch.append(float(splitLine[1]))
    tiflaccuracy.append(float(splitLine[3]))
    tifltimes.append(float(splitLine[17]))

    lineIdx += 1

window = 61
polyorder = 3
pyaccuracysmooth = savgol_filter(pyaccuracy, window, polyorder)
tiflaccuracysmooth = savgol_filter(tiflaccuracy, window, polyorder)
pyctime = np.cumsum(pytimes)
tiflctime = np.cumsum(tifltimes) 
maxtime = max(max(pyctime), max(tiflctime))

x_ticks = np.arange(0, maxtime, 1500)
y_ticks = np.arange(0, 1, 0.1)
plt.plot(pyctime, pyaccuracysmooth, '>',  label = r'P(y)')
plt.plot(tiflctime, tiflaccuracysmooth, '--',  label = r'TiFL')
plt.xticks(x_ticks)
plt.xlabel('Time taken (sec)')
plt.yticks(y_ticks)
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time taken')
plt.legend()
plt.show()