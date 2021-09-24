#!/usr/bin/env python3

import os
import sys

pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common','summary')
sys.path.append(pwd)

from cluster import cluster_mat
from hist import HistSummary, HistMatSummary

from numpy.linalg import norm

l01 = ['0']*1000 + ['1']*50 + ['2']*50 + ['3']*0
l02 = ['0']*1000 + ['1']*50 + ['2']*50 + ['3']*0
l03 = ['0']*1000 + ['1']*50 + ['2']*50 + ['3']*0

l11 = ['0']*0 + ['1']*1000 + ['2']*50 + ['3']*50
l12 = ['0']*0 + ['1']*1000 + ['2']*50 + ['3']*50
l13 = ['0']*0 + ['1']*1000 + ['2']*50 + ['3']*50

l21 = ['0']*50 + ['1']*0 + ['2']*1000 + ['3']*50
l22 = ['0']*50 + ['1']*0 + ['2']*1000 + ['3']*50
l23 = ['0']*50 + ['1']*0 + ['2']*1000 + ['3']*50

l31 = ['0']*50 + ['1']*50 + ['2']*0 + ['3']*1000
l32 = ['0']*50 + ['1']*50 + ['2']*0 + ['3']*1000
l33 = ['0']*50 + ['1']*50 + ['2']*0 + ['3']*1000

l41 = ['0']*750 + ['1']*100 + ['2']*100 + ['3']*0
l42 = ['0']*750 + ['1']*100 + ['2']*100 + ['3']*0
l43 = ['0']*750 + ['1']*100 + ['2']*100 + ['3']*0

l51 = ['0']*0 + ['1']*750 + ['2']*100 + ['3']*100
l52 = ['0']*0 + ['1']*750 + ['2']*100 + ['3']*100
l53 = ['0']*0 + ['1']*750 + ['2']*100 + ['3']*100

l61 = ['0']*100 + ['1']*0 + ['2']*750 + ['3']*100
l62 = ['0']*100 + ['1']*0 + ['2']*750 + ['3']*100
l63 = ['0']*100 + ['1']*0 + ['2']*750 + ['3']*100

l71 = ['0']*100 + ['1']*100 + ['2']*0 + ['3']*750
l72 = ['0']*100 + ['1']*100 + ['2']*0 + ['3']*750
l73 = ['0']*100 + ['1']*100 + ['2']*0 + ['3']*750

l81 = ['0']*1000 + ['1']*0 + ['2']*0 + ['3']*0
l82 = ['0']*1000 + ['1']*0 + ['2']*0 + ['3']*0
l83 = ['0']*1000 + ['1']*0 + ['2']*0 + ['3']*0

l91 = ['0']*0 + ['1']*1000 + ['2']*0 + ['3']*0
l92 = ['0']*0 + ['1']*1000 + ['2']*0 + ['3']*0
l93 = ['0']*0 + ['1']*1000 + ['2']*0 + ['3']*0

h01 = HistSummary(l01)
h02 = HistSummary(l02)
h03 = HistSummary(l03)

h11 = HistSummary(l11)
h12 = HistSummary(l12)
h13 = HistSummary(l13)

h21 = HistSummary(l21)
h22 = HistSummary(l22)
h23 = HistSummary(l23)

h31 = HistSummary(l31)
h32 = HistSummary(l32)
h33 = HistSummary(l33)

h41 = HistSummary(l41)
h42 = HistSummary(l42)
h43 = HistSummary(l43)

h51 = HistSummary(l51)
h52 = HistSummary(l52)
h53 = HistSummary(l53)

h61 = HistSummary(l61)
h62 = HistSummary(l62)
h63 = HistSummary(l63)

h71 = HistSummary(l71)
h72 = HistSummary(l72)
h73 = HistSummary(l73)

h81 = HistSummary(l81)
h82 = HistSummary(l82)
h83 = HistSummary(l83)

h91 = HistSummary(l91)
h92 = HistSummary(l92)
h93 = HistSummary(l93)

hm0 = HistMatSummary({'cat':h01, 'dog':h02, 'bird':h03})
hm1 = HistMatSummary({'cat':h11, 'dog':h12, 'bird':h13})
hm2 = HistMatSummary({'cat':h21, 'dog':h22, 'bird':h23})
hm3 = HistMatSummary({'cat':h31, 'dog':h32, 'bird':h33})
hm4 = HistMatSummary({'cat':h41, 'dog':h42, 'bird':h43})
hm5 = HistMatSummary({'cat':h51, 'dog':h52, 'bird':h53})
hm6 = HistMatSummary({'cat':h61, 'dog':h62, 'bird':h63})
hm7 = HistMatSummary({'cat':h71, 'dog':h72, 'bird':h73})
hm8 = HistMatSummary({'cat':h81, 'dog':h82, 'bird':h83})
hm9 = HistMatSummary({'cat':h91, 'dog':h92, 'bird':h93})

xks = ['0','1','2','3']
yks = ['cat','dog','bird']

m0 = hm0.toMatrix(xks, yks)
m1 = hm1.toMatrix(xks, yks)
m2 = hm2.toMatrix(xks, yks)
m3 = hm3.toMatrix(xks, yks)
m4 = hm4.toMatrix(xks, yks)
m5 = hm5.toMatrix(xks, yks)
m6 = hm6.toMatrix(xks, yks)
m7 = hm7.toMatrix(xks, yks)
m8 = hm8.toMatrix(xks, yks)
m9 = hm9.toMatrix(xks, yks)

print(m0)
print(norm(m1 - m2))
print(cluster_mat([hm0, hm1, hm2, hm3, hm4, hm5, hm6, hm7, hm8, hm9], xks, yks))
print("Expected")
print("[0 1 2 3 0 1 2 3 0 1]")
