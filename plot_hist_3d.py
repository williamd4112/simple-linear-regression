from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

from util import load_dataset_csv
'''
xpos = [1,2,3,4,5,6,7,8,9,10]
ypos = [2,3,4,5,1,6,2,1,7,2]
num_elements = len(xpos)
zpos = [0,0,0,0,0,0,0,0,0,0]
'''
xy, z = load_dataset_csv('X_train.csv', 'T_train.csv')
xy = xy[:4000, :]
z = z[:4000]
xpos = xy[:, 0]
ypos = xy[:, 1]
n = len(xpos)

zpos = np.zeros(n)
dx = np.ones(n) * 5
dy = np.ones(n) * 5
#dz = [1,2,3,4,5,6,7,8,9,10]
dz = z

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
plt.show()
