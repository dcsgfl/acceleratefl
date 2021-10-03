import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.polynomial import poly
from scipy.signal import savgol_filter

rndfh = open("cifar10_perf_rnd.log")
pyfh = open("cifar10_perf_py.log")
pxyfh = open("cifar10_perf_pxy.log")
oortfh = open("cifar10_perf_oort.log")
tiflfh = open("cifar10_perf_tifl.log")


rndcontent = rndfh.readlines()
pycontent = pyfh.readlines()
tiflcontent = tiflfh.readlines()
pxycontent = pxyfh.readlines()
oortcontent = oortfh.readlines()

lineIdx = 0
nLines = 200

rndepoch = []
rndaccuracy = []
rndtimes = []

pyepoch = []
pyaccuracy = []
pytimes = []

tiflepoch = []
tiflaccuracy = []
tifltimes = []

pxyepoch = []
pxyaccuracy = []
pxytimes = []

oortepoch = []
oortaccuracy = []
oorttimes = []

rndCTime = -1
pyCTime = -1
tiflCTime = -1
pxyCTime = -1
oortCTime = -1

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

    currLine = oortcontent[lineIdx]
    splitLine = currLine.split()
    acc = float(splitLine[3])
    oortepoch.append(float(splitLine[1]))
    oortaccuracy.append(acc)
    oorttimes.append(float(splitLine[17]))
    if oortCTime == -1 and acc >= 0.5:
        oortCTime = sum(oorttimes)

    lineIdx += 1

    if pxyCTime != -1 and tiflCTime != -1 and pyCTime != -1 and rndCTime != -1 and oortCTime != -1:
        break

print("RND    TTC: ", rndCTime)
print("P(Y)   TTC: ", pyCTime)
print("P(X|Y) TTC: ", pxyCTime)
print("TiFL   TTC: ", tiflCTime)
print("Oort   TTC: ", oortCTime)

window = 171
polyorder = 3
rndaccuracysmooth = savgol_filter(rndaccuracy, window, polyorder)
pyaccuracysmooth = savgol_filter(pyaccuracy, window, polyorder)
tiflaccuracysmooth = savgol_filter(tiflaccuracy, window, polyorder)
pxyaccuracysmooth = savgol_filter(pxyaccuracy, window, polyorder)
oortaccuracysmooth = savgol_filter(oortaccuracy, window, polyorder)
rndctime = np.cumsum(rndtimes)
pyctime = np.cumsum(pytimes)
tiflctime = np.cumsum(tifltimes)
pxyctime = np.cumsum(pxytimes)
oortctime = np.cumsum(oorttimes)
maxtime = max(max(rndctime), max(oortctime), max(pyctime), max(tiflctime), max(pxyctime))

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
plt.plot(rndctime, rndaccuracysmooth, '-.',  label = 'Random')
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
