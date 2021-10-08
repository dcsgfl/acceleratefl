import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.polynomial import poly
from scipy.signal import savgol_filter

pyfh = open("drop_py_10_200.log")
pxyfh = open("drop_pxy_10_200.log")
tiflfh = open("drop_tifl_10_200.log")
oortfh = open("drop_oort_10_200.log")

pycontent = pyfh.readlines()
pxycontent = pxyfh.readlines()
tiflcontent = tiflfh.readlines()
oortcontent = oortfh.readlines()

lineIdx = 0
nLines = 200

pyepoch = []
pyaccuracy = []
pytimes = []

pxyepoch = []
pxyaccuracy = []
pxytimes = []

tiflepoch = []
tiflaccuracy = []
tifltimes = []

oortepoch = []
oortaccuracy = []
oorttimes = []
while (lineIdx < nLines):
    currLine = pycontent[lineIdx]
    splitLine = currLine.split()
    pyepoch.append(float(splitLine[1]))
    pyaccuracy.append(float(splitLine[3]))
    pytimes.append(float(splitLine[17]))

    currLine = pxycontent[lineIdx]
    splitLine = currLine.split()
    pxyepoch.append(float(splitLine[1]))
    pxyaccuracy.append(float(splitLine[3]))
    pxytimes.append(float(splitLine[17]))

    currLine = tiflcontent[lineIdx]
    splitLine = currLine.split()
    tiflepoch.append(float(splitLine[1]))
    tiflaccuracy.append(float(splitLine[3]))
    tifltimes.append(float(splitLine[17]))

    currLine = oortcontent[lineIdx]
    splitLine = currLine.split()
    oortepoch.append(float(splitLine[1]))
    oortaccuracy.append(float(splitLine[3]))
    oorttimes.append(float(splitLine[17]))

    lineIdx += 1

window = 101
polyorder = 3
pyaccuracysmooth = savgol_filter(pyaccuracy, window, polyorder)
pxyaccuracysmooth = savgol_filter(pxyaccuracy, window, polyorder)
tiflaccuracysmooth = savgol_filter(tiflaccuracy, window, polyorder)
oortaccuracysmooth = savgol_filter(oortaccuracy, window, polyorder)

pyctime = np.cumsum(pytimes)
pxyctime = np.cumsum(pxytimes)
tiflctime = np.cumsum(tifltimes) 
oortctime = np.cumsum(oorttimes)
maxtime = max(max(pyctime), max(pxyctime),  max(tiflctime), max(oortctime))

x_ticks = np.arange(0, maxtime, 1500)
y_ticks = np.arange(0, 1, 0.1)
# plt.plot(rndctime, rndaccuracysmooth, '-.',  label = r'Random')
plt.plot(pyctime, pyaccuracysmooth, '>',  label = r'P(y)')
plt.plot(pxyctime, pxyaccuracysmooth, '-', label = r'P(X|y)')
plt.plot(tiflctime, tiflaccuracysmooth, '--',  label = r'TiFL')
plt.plot(oortctime, oortaccuracysmooth, '+',  label = r'Oort')
plt.xticks(x_ticks)
plt.xlabel('Time taken (sec)')
plt.yticks(y_ticks)
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time taken')
plt.legend()
plt.show()