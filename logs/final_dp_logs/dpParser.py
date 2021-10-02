import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.polynomial import poly
from scipy.signal import savgol_filter

rndfh = open("dp_rnd.txt")
pyfh = open("dp0_1.txt")
pxfh = open("dp0_01.txt")
pxyfh = open("dp0_001.txt")

rndcontent = rndfh.readlines()
pycontent = pyfh.readlines()
pxcontent = pxfh.readlines()
pxycontent = pxyfh.readlines()

lineIdx = 0
nLines = 200

rndepoch = []
rndaccuracy = []
rndtimes = []

pyepoch = []
pyaccuracy = []
pytimes = []

pxepoch = []
pxaccuracy = []
pxtimes = []

pxyepoch = []
pxyaccuracy = []
pxytimes = []

rndCTime = -1
pyCTime = -1
pxCTime = -1
pxyCTime = -1

while (lineIdx < nLines):

    currLine = rndcontent[lineIdx]
    splitLine = currLine.split()
    acc = float(splitLine[3])
    rndepoch.append(float(splitLine[1]))
    rndaccuracy.append(acc)
    rndtimes.append(float(splitLine[17]))
    if rndCTime == -1 and acc >= 0.5:
        rndCTime = sum(rndtimes)

    currLine = pycontent[lineIdx]
    splitLine = currLine.split()
    acc = float(splitLine[3])
    pyepoch.append(float(splitLine[1]))
    pyaccuracy.append(acc)
    pytimes.append(float(splitLine[17]))
    if pyCTime == -1 and acc >= 0.5:
        pyCTime = sum(pytimes)

    currLine = pxcontent[lineIdx]
    splitLine = currLine.split()
    acc = float(splitLine[3])
    pxepoch.append(float(splitLine[1]))
    pxaccuracy.append(acc)
    pxtimes.append(float(splitLine[17]))
    if pxCTime == -1 and acc >= 0.5:
        pxCTime = sum(pxtimes)

    currLine = pxycontent[lineIdx]
    splitLine = currLine.split()
    acc = float(splitLine[3])
    pxyepoch.append(float(splitLine[1]))
    pxyaccuracy.append(acc)
    pxytimes.append(float(splitLine[17]))
    if pxyCTime == -1 and acc >= 0.5:
        pxyCTime = sum(pxytimes)

    lineIdx += 1

    if pxyCTime != -1 and pxCTime != -1 and pyCTime != -1 and rndCTime != -1:
        break

print("RND   TTC: ", rndCTime)
print("0.1   TTC: ", pyCTime)
print("0.01  TTC: ", pxCTime)
print("0.001 TTC: ", pxyCTime)

window = 15
polyorder = 3
rndaccuracysmooth = savgol_filter(rndaccuracy, window, polyorder)
pyaccuracysmooth = savgol_filter(pyaccuracy, window, polyorder)
pxaccuracysmooth = savgol_filter(pxaccuracy, window, polyorder)
pxyaccuracysmooth = savgol_filter(pxyaccuracy, window, polyorder)
rndctime = np.cumsum(rndtimes)
pyctime = np.cumsum(pytimes)
pxctime = np.cumsum(pxtimes)
pxyctime = np.cumsum(pxytimes)
maxtime = max(max(rndctime), max(pyctime), max(pxctime), max(pxyctime))

print(max(rndctime), max(pyctime), max(pxctime), max(pxyctime))
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
plt.plot(rndctime, rndaccuracysmooth, '-.',  label = 'Random')
plt.plot(pyctime, pyaccuracysmooth, '>',  label = r'P(y), $\epsilon$ = 0.1')
plt.plot(pyctime, pxaccuracysmooth, '--',  label = r'P(y), $\epsilon$ = 0.01')
plt.plot(pxyctime, pxyaccuracysmooth, '-', label = r'P(y), $\epsilon$ = 0.001')
plt.xticks(x_ticks)
plt.xlabel('Time taken (sec)')
plt.yticks(y_ticks)
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time taken')
plt.legend()
#plt.show()
plt.savefig("dp6.png")
