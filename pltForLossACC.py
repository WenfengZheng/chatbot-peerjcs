from matplotlib import pyplot as plt

import csv

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
data1_loss =np.loadtxt("training.log")
#data2_loss = np.loadtxt("valid_SCRCA_records.txt")

x = data1_loss[:,0]
y = data1_loss[:,1]
# x1 = data2_loss[:,0]
# y1 = data2_loss[:,1]


###############################################    show   ###############################################

fig = plt.figure(figsize = (7,5))       #figsize is the size of the picture
ax1 = fig.add_subplot(1, 1, 1) # ax1 is the name of the subgraph`

pl.plot(x,y,'g-',label=u'128')
# "g" stands for "green", which means that the drawn curve is green, "-" means that the drawn curve is a solid line, which can be selected by yourself, and label represents the name of the legend. Generally, a u should be added in front of the name. If the name It is Chinese, it will not be displayed, and I don’t know how to solve it yet.
# p2 = pl.plot(x1, y1,'r-', label = u'')
pl.legend()
#show legend
# p3 = pl.plot(x2,y2, 'b-', label = u'SCRCA_Net')
pl.legend()
pl.xlabel(u'ephch')
pl.ylabel(u'loss')
plt.title('Compare loss for different models in training')
pl.show()








