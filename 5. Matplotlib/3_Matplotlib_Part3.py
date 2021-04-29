import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,5,11)
y = x ** 2

## Setting Colors

fig = plt.figure()
ax = fig.add_axes([.1,.1,.8,.8])

ax.plot(x,y,color='cyan',linewidth=2,alpha=.5,linestyle='--',marker='o'
,markersize=10, markeredgecolor='red')
# ^^ can search google for RGB Hex codes for color options
# ^^ alpha argument changes line transparency
# ^^ linestyle argument lets you change the line to dotted, dashed, etc.
# ^^ marker lets you mark each data point with a shape
# ^^ a lot of customization with marker

ax.set_xlim([0,1])
ax.set_ylim([0,2])

plt.show()