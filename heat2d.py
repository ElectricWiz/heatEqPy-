#!/usr/bin/env python
# coding: utf-8

# In[170]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

sns.set_theme()


# In[171]:


Nx = 8
Ny = 8
Nt = 100

tf = 10

deltaX = 1/(Nx - 1)
deltaY = 1/(Ny - 1)
deltaT = tf/(Nt - 1)


# In[172]:


x = np.arange(0, 1.01, deltaX)
y = np.arange(0, 1.01, deltaY)
t = np.arange(0, tf + deltaT, deltaT)

X, Y = np.meshgrid(x, y)


# In[173]:


sx = deltaT/deltaX**2
sy = deltaT/deltaY**2


# In[174]:


wij_1 = np.zeros((Nx, Ny))
wij = np.zeros((Nx, Ny))

wij[:, 0] = 1  # Left boundary
wij[:, -1] = 1  # Right boundary
wij[0, :] = 1  # Top boundary
wij[-1, :] = 1  # Bottom boundary


# In[175]:


for n in range(0, Nt):
    wij_1 = np.copy(wij)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            wij[i, j] = wij_1[i, j] + sx*(wij_1[i+1, j] - 2*wij_1[i, j] + wij_1[i-1, j]) + sy*(wij_1[i, j+1] - 2*wij_1[i, j] + wij_1[i, j-1])


# In[176]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming wij is already defined and has the shape (Nx, Ny)
Nx, Ny = wij.shape

# Create a meshgrid for plotting
X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))

# Set up the figure and axis for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, wij, cmap='viridis')

# Set plot labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Temperature')
ax.set_title('3D Surface Plot of wij')

# Show the plot
plt.show()


# In[ ]:




