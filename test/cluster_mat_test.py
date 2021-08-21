#!/usr/bin/env python3

import os
import sys

pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common','summary')
sys.path.append(pwd)

from cluster import cluster_mat
from hist import HistSummary, HistMatSummary

from numpy.linalg import norm

l1 = ['0'] * 19 + ['1']*5
l2 = ['1'] * 19 + ['2']*5
l3 = ['0'] * 12 + ['1']*2
l4 = ['1'] * 12 + ['2']*2
#l3 = ['2'] * 19 + ['1']*5
#l4 = ['3'] * 19 + ['2']*5

h1 = HistSummary(l1)
h2 = HistSummary(l2)
h3 = HistSummary(l3)
h4 = HistSummary(l4)

hm1 = HistMatSummary({'a':h1, 'b':h2})
hm2 = HistMatSummary({'a':h3, 'b':h4})

xks = ['0','1','2','3']
yks = ['a','b']

m1 = hm1.toMatrix(xks, yks)
m2 = hm2.toMatrix(xks, yks)
print(m1)
print(m2)
print(norm(m1 - m2))
sys.exit(0)

l01 = ['0'] * 19 + ['9']*1
l11 = ['1'] * 19 + ['8']*2
l21 = ['2'] * 19 + ['7']*3
l31 = ['3'] * 19 + ['6']*1
l41 = ['4'] * 19 + ['5']*2
l51 = ['5'] * 19 + ['4']*3
l61 = ['6'] * 19 + ['3']*1
l71 = ['7'] * 19 + ['2']*2
l81 = ['8'] * 19 + ['1']*3
l91 = ['9'] * 19 + ['0']*1

l02 = ['0'] * 19 + ['5']*2
l12 = ['1'] * 19 + ['6']*3
l22 = ['2'] * 19 + ['7']*1
l32 = ['3'] * 19 + ['8']*2
l42 = ['4'] * 19 + ['9']*3
l52 = ['5'] * 19 + ['0']*1
l62 = ['6'] * 19 + ['1']*2
l72 = ['7'] * 19 + ['2']*3
l82 = ['8'] * 19 + ['3']*1
l92 = ['9'] * 19 + ['4']*1

ks = ['0','1','2','3','4','5','6','7','8','9']

h01 = HistSummary(l01)
h11 = HistSummary(l11)
h21 = HistSummary(l21)
h31 = HistSummary(l31)
h41 = HistSummary(l41)
h51 = HistSummary(l51)
h61 = HistSummary(l61)
h71 = HistSummary(l71)
h81 = HistSummary(l81)
h91 = HistSummary(l91)

h02 = HistSummary(l02)
h12 = HistSummary(l12)
h22 = HistSummary(l22)
h32 = HistSummary(l32)
h42 = HistSummary(l42)
h52 = HistSummary(l52)
h62 = HistSummary(l62)
h72 = HistSummary(l72)
h82 = HistSummary(l82)
h92 = HistSummary(l92)

h01.addNoise(0.5)

hlist = [h01, h11, h21, h31, h41, h51, h61, h71, h81, h91,
         h02, h12, h22, h32, h42, h52, h62, h72, h82, h92]

classes = cluster_hist(hlist, ks)
print(classes)
