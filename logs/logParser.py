import numpy as np
import matplotlib.pyplot as plt

rndfh = open("rnd_dist.log")
pyfh = open("py_dist.log")
pxyfh = open("pxy_dist.log")
oortfh = open("oort_dist.log")
tiflfh = open("tifl_dist.log")

# rndfh = open("rnd_beta.log")
# pyfh = open("py_beta.log")
# pxyfh = open("pxy_beta.log")
# oortfh = open("oort_beta_a15.log")

# rndfh = open("rnd_13_1.log")
# pyfh = open("py_13_1.log")
# pxyfh = open("pxy_13_1.log")

rndcontent = rndfh.readlines()
pycontent = pyfh.readlines()
pxycontent = pxyfh.readlines()
oortcontent = oortfh.readlines()
tiflcontent = tiflfh.readlines()

lineIdx = 0
nLines = 170

rndepoch = []
rndaccuracy = []
rndtimes = []

pyepoch = []
pyaccuracy = []
pytimes = []

pxyepoch = []
pxyaccuracy = []
pxytimes = []

oortepoch = []
oortaccuracy = []
oorttimes = []

tiflepoch = []
tiflaccuracy = []
tifltimes = []
while (lineIdx < nLines):
    currLine = rndcontent[lineIdx]
    splitLine = currLine.split()
    rndepoch.append(float(splitLine[1]))
    rndaccuracy.append(float(splitLine[3]))
    rndtimes.append(float(splitLine[17]))

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

    currLine = oortcontent[lineIdx]
    splitLine = currLine.split()
    oortepoch.append(float(splitLine[1]))
    oortaccuracy.append(float(splitLine[3]))
    oorttimes.append(float(splitLine[17]))

    currLine = tiflcontent[lineIdx]
    splitLine = currLine.split()
    tiflepoch.append(float(splitLine[1]))
    tiflaccuracy.append(float(splitLine[3]))
    tifltimes.append(float(splitLine[17]))
    lineIdx += 1

rndctime = np.cumsum(rndtimes)
pyctime = np.cumsum(pytimes)
pxyctime = np.cumsum(pxytimes)
oortctime = np.cumsum(oorttimes) 
tiflctime = np.cumsum(tifltimes) 
maxtime = max(max(rndctime), max(pyctime), max(pxyctime), max(oortctime), max(tiflctime))
# maxtime = max(max(rndctime), max(pyctime), max(pxyctime))

print(max(rndctime), max(pyctime), max(pxyctime), max(oortctime), max(tiflctime))
print(max(rndctime), max(pyctime), max(pxyctime))

x_ticks = np.arange(0, 180, 10)
y_ticks = np.arange(0, 1, 0.1)
plt.plot(rndepoch, rndaccuracy, label='random')
plt.plot(rndepoch, pyaccuracy, label = 'py')
plt.plot(rndepoch, pxyaccuracy, label ='pxy')
plt.plot(rndepoch, oortaccuracy, label = 'oort')
plt.plot(rndepoch, tiflaccuracy, label = 'tifl')
plt.xticks(x_ticks)
plt.xlabel('Epochs')
plt.yticks(y_ticks)
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.show()

x_ticks = np.arange(0, maxtime, 1500)
y_ticks = np.arange(0, 1, 0.1)
plt.plot(rndctime, rndaccuracy, '--', label = 'random')
plt.plot(pyctime, pyaccuracy, '-.', label = 'py')
plt.plot(pxyctime, pxyaccuracy, '>', label = 'pxy')
plt.plot(oortctime, oortaccuracy, '+', label = 'oort')
plt.plot(tiflctime, tiflaccuracy, '-', label = 'tifl')
plt.xticks(x_ticks)
plt.xlabel('Time taken (sec)')
plt.yticks(y_ticks)
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time taken')
plt.legend()
plt.show()