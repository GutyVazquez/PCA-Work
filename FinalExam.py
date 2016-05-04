# -*- coding: utf-8 -*-
"""
Created on Wed May 04 00:19:52 2016

@author: Augusto Ruiz Vazquez
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import datasets

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data 
y = iris.target
print X


fig = plt.figure(1, figsize=(4, 3))
plt.clf() 
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

vectorpromedio = np.mean(X, axis=0)
print('Promedio de x y y \n%s'%vectorpromedio)

matrizdecov = (X - vectorpromedio).T.dot((X - vectorpromedio)) / (X.shape[0]-1)
print('Matriz de covarianza \n%s' %matrizdecov)

valprop, vecprop = np.linalg.eig(matrizdecov)

print('vectorespropios \n%s' %vecprop)
print('\valorespropios de mayor a menor \n%s' %valprop)

nmc = vecprop[:,[0,1,2]]
print('\nueva matriz de covarianza sin la columna 3\n%s' %nmc)

nmcT = (nmc).T
print('\nueva matriz de covarianza transpuesta\n%s' % nmcT)
dataT = (X).T
print('\datos de la transpuesta\n%s' % dataT)

nuevosdatos = (nmcT).dot(dataT)
print nuevosdatos

ndsT = (nuevosdatos).T

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(ndsT[y == label, 0].mean(),
              ndsT[y == label, 1].mean() + 1.5,
              ndsT[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(ndsT[:, 0], ndsT[:, 1], ndsT[:, 2], c=y, cmap=plt.cm.spectral)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()