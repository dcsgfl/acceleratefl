import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.polynomial import poly
from scipy.signal import savgol_filter

pyfh = open("drop_py_10.log")
pxyfh = open("drop_pxy_10.log")
tiflfh = open("drop_tifl_10.log")

pycontent = pyfh.readlines()
tiflcontent = tiflfh.readlines()
pxycontent = pxyfh.readlines()

lineIdx = 0
nLines = 140

pyepoch = []
pyaccuracy = []
pytimes = []

tiflepoch = []
tiflaccuracy = []
tifltimes = []

pxyepoch = []
pxyaccuracy = []
pxytimes = []

pyCTime = -1
tiflCTime = -1
pxyCTime = -1

while (lineIdx < nLines):
    currLine = pycontent[lineIdx]
    splitLine = currLine.split()
    acc = float(splitLine[3])
    pyepoch.append(float(splitLine[1]))
    pyaccuracy.append(acc)
    pytimes.append(float(splitLine[17]))
    if pyCTime == -1 and acc >= 0.5:
        pyCTime = sum(pytimes)

    currLine = tiflcontent[lineIdx]
    splitLine = currLine.split()
    acc = float(splitLine[3])
    tiflepoch.append(float(splitLine[1]))
    tiflaccuracy.append(acc)
    tifltimes.append(float(splitLine[17]))
    if tiflCTime == -1 and acc >= 0.5:
        tiflCTime = sum(tifltimes)

    currLine = pxycontent[lineIdx]
    splitLine = currLine.split()
    acc = float(splitLine[3])
    pxyepoch.append(float(splitLine[1]))
    pxyaccuracy.append(acc)
    pxytimes.append(float(splitLine[17]))
    if pxyCTime == -1 and acc >= 0.5:
        pxyCTime = sum(pxytimes)

    lineIdx += 1

    if pxyCTime != -1 and tiflCTime != -1 and pyCTime != -1:
        break

print("P(Y)   TTC: ", pyCTime)
print("P(X|Y) TTC: ", pxyCTime)
print("TiFL   TTC: ", tiflCTime)

window = 51
polyorder = 3
pyaccuracysmooth = savgol_filter(pyaccuracy, window, polyorder)
tiflaccuracysmooth = savgol_filter(tiflaccuracy, window, polyorder)
pxyaccuracysmooth = savgol_filter(pxyaccuracy, window, polyorder)
pyctime = np.cumsum(pytimes)
tiflctime = np.cumsum(tifltimes)
pxyctime = np.cumsum(pxytimes)
maxtime = max(max(pyctime), max(tiflctime), max(pxyctime))

#print(max(rndctime), max(pyctime), max(pxctime), max(pxyctime))
# print(max(rndctime), max(pyctime), max(pxyctime))

# x_ticks = np.arange(0, 180, 10)
# y_ticks = np.arange(0, 1, 0.1)
# plt.plot(rndepoch, rndaccuracy, label='random')
# plt.plot(rndepoch, pyaccuracy, label = 'py')
# plt.plot(rndepoch, pxyaccuracy, label ='pxy')
# plt.plot(rndepoch, oortaccuracy, label = 'oort')
# plt.plot(rndepoch, tiflaccuracy, label = 'tifl')
# plt.xticks(x_ticks)
# plt.xlabel('Epochs')
# plt.yticks(y_ticks)
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs Epoch')
# plt.legend()
# plt.show()

x_ticks = np.arange(0, maxtime, 1500)
y_ticks = np.arange(0, 0.6, 0.1)
#plt.plot(pyctime, pyaccuracy, '-.', label = 'py')
#plt.plot(pxyctime, pxyaccuracy, '>', label = 'pxy')
plt.plot(pyctime, pyaccuracysmooth, '>',  label = r'P(y)')
plt.plot(pxyctime, pxyaccuracysmooth, '-', label = r'P(X|y)')
plt.plot(tiflctime, tiflaccuracysmooth, '--',  label = r'TiFL')
plt.xticks(x_ticks)
plt.xlabel('Time taken (sec)')
plt.yticks(y_ticks)
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time taken')
plt.legend()
plt.show()
