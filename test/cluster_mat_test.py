#!/usr/bin/env python3

import os
import sys

pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common','summary')
sys.path.append(pwd)

from cluster import cluster_mat
from hist import HistSummary, HistMatSummary

from numpy.linalg import norm

l1 = ['0'] * 1900 + ['1']*50
l2 = ['1'] * 19 + ['2']*50
l3 = ['0'] * 1000 + ['3']*20
l4 = ['1'] * 200 + ['3']*20

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
print(cluster_mat([hm1, hm2], xks, yks))
