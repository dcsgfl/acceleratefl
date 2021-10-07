import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.polynomial import poly
from scipy.signal import savgol_filter

rndfh = open("drop_rnd_10_400.log")
pyfh = open("drop_py_10_400.log")
pxyfh = open("drop_pxy_10_400.log")
tiflfh = open("drop_tifl_10_400.log")
oortfh = open("drop_oort_10_400.log")

rndcontent = rndfh.readlines()
pycontent = pyfh.readlines()
pxycontent = pxyfh.readlines()
tiflcontent = tiflfh.readlines()
oortcontent = oortfh.readlines()

lineIdx = 0
nLines = 400

rndepoch = []
rndaccuracy = []
rndtimes = []

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

rndCTime = -1
pyCTime = -1
pxyCTime = -1
tiflCTime = -1
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

    currLine = pxycontent[lineIdx]
    splitLine = currLine.split()
    acc = float(splitLine[3])
    pxyepoch.append(float(splitLine[1]))
    pxyaccuracy.append(acc)
    pxytimes.append(float(splitLine[17]))
    if pxyCTime == -1 and acc >= 0.5:
        pxyCTime = sum(pxytimes)

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

    if rndCTime != -1 and pyCTime != -1 and pxyCTime != -1 and tiflCTime != -1 and oortCTime != -1:
        break

print("RND   TTC: ", rndCTime)
print("P(Y)   TTC: ", pyCTime)
print("P(X|Y) TTC: ", pxyCTime)
print("TiFL   TTC: ", tiflCTime)
print("OORT   TTC: ", oortCTime)

window = 151
polyorder = 3
rndaccuracysmooth = savgol_filter(rndaccuracy, window, polyorder)
pyaccuracysmooth = savgol_filter(pyaccuracy, window, polyorder)
pxyaccuracysmooth = savgol_filter(pxyaccuracy, window, polyorder)
tiflaccuracysmooth = savgol_filter(tiflaccuracy, window, polyorder)
oortaccuracysmooth = savgol_filter(oortaccuracy, window, polyorder)

rndctime = np.cumsum(rndtimes)
pyctime = np.cumsum(pytimes)
pxyctime = np.cumsum(pxytimes)
tiflctime = np.cumsum(tifltimes)
oortctime = np.cumsum(oorttimes)

maxtime = max(max(rndctime), max(pyctime),max(pxyctime),  max(tiflctime), max(oortctime))

x_ticks = np.arange(0, maxtime, 1500)
y_ticks = np.arange(0, 0.6, 0.1)
plt.plot(rndctime, rndaccuracysmooth, '-.',  label = r'Random')
plt.plot(pyctime, pyaccuracysmooth, '>',  label = r'P(y)')
plt.plot(pxyctime, pxyaccuracysmooth, '-', label = r'P(X|y)')
plt.plot(tiflctime, tiflaccuracysmooth, '--',  label = r'TiFL')
plt.plot(oortctime, oortaccuracysmooth, '--',  label = r'Oort')
plt.xticks(x_ticks)
plt.xlabel('Time taken (sec)')
plt.yticks(y_ticks)
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time taken')
plt.legend()
plt.show()
