import matplotlib.pyplot as plt

# insert plt.show() to display chart or graphic in terminal

import numpy as np

x = np.linspace(0,5,11)
y = x ** 2

# functional method

plt.plot(x,y)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
# plt.show() # displays plot in a separate window

# mulitple plots on same graph
plt.subplot(1,2,1)
plt.plot(x,y,'r')

plt.subplot(1,2,2)
plt.plot(y,x,'b')
# plt.show()

######################################################

# Object Oriented Method

fig = plt.figure() # blank canvas

axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(x,y)
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_title('Title')

# plt.show()

fig = plt.figure()
axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig.add_axes([.4,.2,.4,.3]) # (dist from btm left, dist from bottom, width, height)

axes1.plot(x,y)
axes1.set_title('LARGER PLOT')

axes2.plot(y,x)
axes2.set_title('SMALLER PLOT')

plt.show()




