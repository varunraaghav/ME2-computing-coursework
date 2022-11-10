##  Computing coursework: explicit 2d heat equation, transient, with neumann BC
##  Students: Varun Raaghav, Pablo Capell-Abril

import numpy as np
import matplotlib.pyplot as plt


#### constants of the problem, from physics: 
    
h = float(0.000001)    #### convective coeffcient (h) of water in W/mm^2K
alpha = float(0.000111) ## thermal diffusivity of copper (per second)
kt = float(0.385)  #### conductivity of copper in W/mmK


#### time step discretisation : 

t = 2000  ## total time
nt = 250  ## number of time steps
delta_t = float(t/nt)   #### delta time


### mesh grid discretisation in x,y : 
    
X_length = 2 # (in mm) - width of chip
Y_length = 1 # (in mm)   - height of chip
nx = 30      ##### x axis discretisation (ie. number of nodes )
delta_x = float(X_length / nx)  ### difference in mm between nodes in x axis
delta_y = delta_x  ## for simplicity of solving problem

   
ny = int((Y_length/delta_y)) ### y-axis discretisation


m = ny+1    ## these indexes are used to interate through the matrix of temperatures: T
p = nx+1

r = nt+1 ## no. of time steps


d = (alpha*delta_t)/(delta_y**2)   ## this is called the diffusion number, used to check for conditional stability


## conditional stability is a (consequence of explicit methods)

if d < 0.25:
    print('solution stable. d:', d)
else: 
    print('solution unstable. d:', d)
    
T = np.zeros([r, m, p])       ####### T = function(time, rows, columns)

  
 
T1 , T2, T3, = 60, 30, 118      # initial temperatures for the Dirichlet BCs
T5 = 120   ### Iinitial condition for the internal nodes - the plate cools from 120 deg C
Tinf = 10  ###### cold stream temperature of the water cooling

    
### Initial conditions: the matrix is set up in such a way that T(all time) = initial conditions
T[:,:,:] = T5                   
T[:,0,:] = T1
T[:,-1,:] = T3
T[:,:,0] = T2
T[:,:,-1] = T2


########## iterating from time=1 to t_end (which is pre-set as 2000 s)

for k in range(0,r-1):
    for i in range(1,m):
        for j in range(1,p-1):
    
            if i==m-1 and (j>0 and j<p-1):                
                dt_dx = float((-1*h/kt)*(T[k,i,j] - Tinf))   ### the variable neumann condition                
                T[k+1,i,j] = T[k,i,j] + d*(T[k,i,j-1] +T[k,i,j+1] + 2*T[k,i-1,j]+ 2*delta_y*dt_dx - 4*T[k,i,j])

            else:
                T[k+1,i,j] = T[k,i,j] + d*(T[k,i+1,j] + T[k,i-1,j] + T[k,i,j+1] + T[k,i,j-1]  - 4*T[k,i,j])  ### the explicit formula


#__________________________________________________________________________
### plotting:
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import matplotlib.colors

### making mesh grid for making surface and contour plot
X = np.arange(0,m)
Y = np.arange(0,p)

X2 = np.arange(0,m+1)
Y2 = np.arange(0,p+1)   

X,Y = np.meshgrid(Y,X)   # surface plot meshgrid
X2,Y2 = np.meshgrid(Y2,X2) ## contour plot meshgrid

l = 250  # max at nt = 250, which would be at time = 2000s

### Surface plot:    
fig = plt.figure(dpi=120, figsize=(10,7))
ax = fig.add_subplot(111, projection ='3d')
norm = matplotlib.colors.Normalize(vmin = T2, vmax = T5, clip = False)
ax.set_zlim([T2,T5])


surf = ax.plot_surface(X, Y, T[l,:,:], rstride = 1, cstride = 1, norm=norm, alpha = 1, cmap='coolwarm', linewidth = 0)
fig.colorbar(surf, shrink=0.5, aspect = 5)
time = int(l *delta_t)

ax.set_xlabel('x axis nodes')
ax.set_ylabel('y axis nodes')
ax.set_zlabel('Temperature (deg C)')
ax.set_title('Surface plot at time (s): %s' %time)

#__________________________________________________________________________
# Contour plot (heatmap): 
fig = plt.figure(dpi=120)
ax = fig.add_subplot()
c = ax.pcolormesh(X2, Y2, T[l,:,:], cmap='coolwarm', vmin=T2, vmax=T5)
norm = matplotlib.colors.Normalize(vmin = T2, vmax = T5, clip = False)
fig.colorbar(c, shrink=0.5, aspect=5)
ax.set_xlabel('x axis nodes (0 - 2 mm)')
ax.set_ylabel('y axis nodes (0 - 1 mm)')
ax.set_title('Contour plot (ie. heatmap) at time (s): %s ' %time)
plt.show()
