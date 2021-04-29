import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,5,11)
y = x ** 2

# In this section we will create a figure and axes in one line

#fig,axes = plt.subplots(nrows=1,ncols=2) # this will create two side-by-side graphs
# axes.plot(x,y)


#fig,axes = plt.subplots(nrows=3,ncols=3)
#plt.tight_layout() # this will space out clutered graphs automatically

fig,axes = plt.subplots(nrows=1,ncols=2)

axes[0].plot(x,y)
axes[0].set_title('First Plot')
axes[1].plot(y,x)
axes[1].set_title('Second Plot')

##################################################################3

# Figure Size and DPI

fig = plt.figure(figsize = (8,2),dpi=100)
ax = fig.add_axes([0.1,0.1,.8,.8])
ax.plot(x,x**2,label='X Squared')
ax.plot(x,x**3,label='X Cubed')

#fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(8,2))
#axes[0].plot(x,y)
#axes[1].plot(y,x)

# Saving a figure

fig.savefig('my_plot.png',dpi=100)

# adding a legend: see also the label arguments in the ax.plot() lines

ax.legend(loc=0)


plt.tight_layout()
plt.show()