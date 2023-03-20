
#————————————————————————————————————————————————————————————————————

#EX4 - Simulation of rocket orbits

#————————————————————————————————————————————————————————————————————

# PROGRAM CONTENTS (Lines may be slightly different when a txt):

# - SETUP ( Line: 44-69 )
    # - Importing libraries
    # - Setting text sizes

# - MAIN FUNCTIONS ( Line: 70-190 )
    # - RK4
    # - Euler
    # - Analytical (Circular orbit only)
    # - function to find velocity of cicular orbit

# - SECTION A ( Line: 190-276 ) :
    # - function for newtons law of gravitation (2-body)
    # - function to run RK4 and simulate 2 body orbit in 2D

# - SECTION B ( Line: 276-397 ):
    # - function for newtons law of gravitation (3-body)
    # - function to run RK4 and simulate a moon shot (2D axis)
    # - function to run RK4 and simulate a moon shot (3D axis)

# - ANALYSIS ( Line: 397-512 ):
    # - a chi-squared function
    # - a function to compute and plot accuracy against time (circular orbit)
    # - a function to simulate and compare methods (circular orbit)

# - EXTENSTION ( Line: 512-624 ) (An attempt to simulate a moon shot for a moving moon method could be incorrect):
    # - RK4 method with a single iteration
    # - running and simualting a moon shot

# - RUNNING ( Line: 624-838 ):
    # - A full menu to run all sections

#————————————————————————————————————————————————————————————————————

# SETUP

#————————————————————————————————————————————————————————————————————

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import *
import numpy as np
from typing import Callable, Any


#for report:
#small= 12
#med = 18
#big = 22
#title = 24

#plt.rc('axes', titlesize=title)     
#plt.rc('axes', labelsize=big)    
#plt.rc('xtick', labelsize=small)    
#plt.rc('ytick', labelsize=small)    
#plt.rc('legend', fontsize=med)
#plt.rc('figure', titlesize=title)

# MAIN FUNCTIONS

#————————————————————————————————————————————————————————————————————

    # - RK4
    #————————————————————————————————————————————————————————————————————

#This function takes a function as an argument, and will take 3 dimensional vectors
#For this reason run time is slightly longer than otherwise i.e. 2 dimensions without a function argument. 
#i initially used a more long winded approach that could not be adapted for any function but this way the function is particularly vesitile. 

def RK(f_r: Callable[[np.ndarray],Any],r_0: np.ndarray,v_0: np.ndarray,h: float,N: int,*args):

    t = [0]
    r = np.array([[r_0[0]],[r_0[1]],[r_0[2]]])
    v = np.array([[v_0[0]],[v_0[1]],[v_0[2]]])

    for i in range(N):
        t_i_1 = t[i] + h
        t.append(t_i_1)


        #K_1
        #——————————————————————————————————

        k_1 = np.array([v[0][i],v[1][i],v[2][i]])
        k_1_v = f_r(np.array([r[0][i],r[1][i],r[2][i]]),*args)

        #K_2
        #——————————————————————————————————

        k_2 =  k_1 + h*k_1_v/2
        k_2_v = f_r(np.array([r[0][i],r[1][i],r[2][i]]) + h*k_1/2,*args)

        #K_3
        #——————————————————————————————————

        k_3 =  k_1 + h*k_2_v/2
        k_3_v = f_r(np.array([r[0][i],r[1][i],r[2][i]]) + h*k_2/2,*args)

        #K_4
        #——————————————————————————————————

        k_4 =  k_1 + h*k_3_v
        k_4_v = f_r(np.array([r[0][i],r[1][i],r[2][i]]) + h*k_3,*args)

        r_i_1 = np.array([r[0][i],r[1][i],r[2][i]]) + (h/6)*( k_1 + 2*k_2 + 2*k_3 + k_4)
        v_i_1 = np.array([v[0][i],v[1][i],v[2][i]]) + (h/6)*( k_1_v + 2*k_2_v + 2*k_3_v + k_4_v)

        r = np.append(r,[[r_i_1[0]],[r_i_1[1]],[r_i_1[2]]],1)
        v = np.append(v,[[v_i_1[0]],[v_i_1[1]],[v_i_1[2]]],1)

    return r[0],r[1],r[2]


    # - Euler
    #————————————————————————————————————————————————————————————————————

# Simularly we take a function as an argument. This method is used as a comparison to RK4

def Euler(f_r: Callable[[np.ndarray],Any],r_0: np.ndarray,v_0: np.ndarray,h: float,N: int,*args):

    t = [0]
    r = np.array([[r_0[0]],[r_0[1]],[r_0[2]]])
    v = np.array([[v_0[0]],[v_0[1]],[v_0[2]]])

    for i in range(N):
        t_i_1 = t[i] + h
        t.append(t_i_1)

        v0 = np.array([v[0][i],v[1][i],v[2][i]])
        r_i_1 = np.array([r[0][i],r[1][i],r[2][i]]) + (h)*(v0)

        dv_dt = f_r(r_i_1,*args)
        v_i_1 = np.array([v[0][i],v[1][i],v[2][i]]) + (h)*(dv_dt)

        r = np.append(r,[[r_i_1[0]],[r_i_1[1]],[r_i_1[2]]],1)
        v = np.append(v,[[v_i_1[0]],[v_i_1[1]],[v_i_1[2]]],1)


    return r[0],r[1],r[2]


    # - Analytical (Circular orbit only)
    #————————————————————————————————————————————————————————————————————

#For this function the equation for circular motion has been equated to newtonian gravity and a perfect period has been found.
#Then by using theta = 2pit/T and converting into cartisian cords a perfect circular orbit is found.

def circular_orbit_analytical(r: float,M: float,h: float,N: int):

    G = 6.67e-11

    t = [0]
    theta = [0]

    T_period = sqrt( 4*(pi**2)*(r**3) / (G*M) )

    angle_step = 2*pi / (T_period/h)

    for i in range(N):
        t_i_1 = t[i] + h
        t.append(t_i_1)
        theta_i_1 = theta[i] + angle_step
        theta.append(theta_i_1)

    theta = np.array(theta)

    x = r*np.cos(theta)
    y = r*np.sin(theta)

    return x,y,np.array(t)

    # - min velocity function
    #————————————————————————————————————————————————————————————————————

# Again equation of circular orbit is used to find the velocity v_0 at r_0 which will send a body into a circular orbit. 

def min_velocity(M,r):
    v = sqrt(6.67e-11*M/r)
    return v


#END OF SECTION

#————————————————————————————————————————————————————————————————————

# SECTION A

#————————————————————————————————————————————————————————————————————

    # - 2 body gravitation
    #————————————————————————————————————————————————————————————————————

# newtons law of gravitation for 2 bodys assuming M >> m:

def grav_2_body(r: np.ndarray,M: float) -> np.ndarray: #where r is a vector i.e = r = [x,y,z]

    G = 6.67e-11

    dv_dt = -G*M*r * (r[0]**2 + r[1]**2 + r[2]**2) **(-3/2)

    return dv_dt

    # - section A simulation 
    #————————————————————————————————————————————————————————————————————

# A function that runs RK4 in order to simulate motion defined by grav_2_body:

def run_sectionA(M,start_dis,vel_factor,step,tot_time,time_units:str,line: bool):

    start_vel = vel_factor * min_velocity(M,start_dis)

    #create the figure and objects 

    orbit_fig, orbit_ax = plt.subplots(figsize=(10,10))

    orbit_ax.set_xlim(-3*start_dis,3*start_dis)
    orbit_ax.set_ylim(-3*start_dis,3*start_dis)

    #create objects

    spaceship, = orbit_ax.plot([],[],marker='<',markerfacecolor="red",markeredgecolor='red',markersize=6)
    ship_line, = orbit_ax.plot([0],[0],'-g',lw=1)
    planet, = orbit_ax.plot([0],[0],marker='o',markerfacecolor="blue",markeredgecolor='blue',markersize=50)

    ship_label = orbit_ax.text(-3*0.8*start_dis,3*0.8*start_dis,'Time: 0 Days')
    
    #collecting data 
    
    r_0 = np.array([start_dis,0,0])
    v_0 = np.array([0,start_vel,0])

    x,y,z = RK(grav_2_body,r_0,v_0,step,tot_time,5.97e24)



#updating the data at the ith frame
    def update_frame(i):

        spaceship.set_data(x[i],y[i])
        #Time in minutes = step*i #would expect this to be in seconds but a factor of 60 is unaccounted for somewhere?
        ship_label.set_text(f'Time: {"{0:.2f}".format(step*i)} {time_units}')
        ship_line.set_data(x[:i],y[:i])

        #print(i)
        if line == True:
            return spaceship,ship_label,ship_line
        else:
            return spaceship,ship_label
        


    anim = animation.FuncAnimation(orbit_fig,func=update_frame,frames=len(x),interval=1,blit=True)
    plt.show()


#run_sectionA(5.97e24,408000,1,0.05,1000,'Minutes',True)

# END OF SECTION

#————————————————————————————————————————————————————————————————————

# SECTION B

#————————————————————————————————————————————————————————————————————

    # - 3 body gravitation
    #————————————————————————————————————————————————————————————————————

#3 body gravitation where Me and Mm >> m and Mm is stationary.

def grav_3_body(r: np.ndarray,M1: float,M2: float,R1: np.ndarray,R2: np.ndarray):

    G = 6.67e-11

    r_R1 = r-R1
    r_R2 = r-R2

    dv_dt = -G*M1*(r_R1) * ( (r_R1[0])**2 + (r_R1[1])**2 + (r_R1[2])**2 )**(-3/2) - G*M2*(r_R2) * ( (r_R2[0])**2 + (r_R2[1])**2 + (r_R2[2])**2 )**(-3/2)

    return dv_dt


    # - section B simulation 
    #————————————————————————————————————————————————————————————————————

# A function that runs RK4 in order to simulate motion defined by grav_3_body:

def run_sectionB(start_dis,vel_factor,step,tot_time,time_units:str,line: bool):

    start_vel = vel_factor*min_velocity(5.97e24,abs(start_dis))

    #create the figure and objects 

    x_m = 382_500_000 #distance to moon.

    orbit_fig, orbit_ax = plt.subplots(figsize=(10,10))

    orbit_ax.set_xlim(-x_m,2*x_m)
    orbit_ax.set_ylim(-x_m,x_m)

    spaceship, = orbit_ax.plot([],[],marker='<',markerfacecolor="red",markeredgecolor='red',markersize=3)
    ship_line, = orbit_ax.plot([0],[0],'-g',lw=1)
    earth, = orbit_ax.plot([0],[0],marker='o',markerfacecolor="blue",markeredgecolor='blue',markersize=5)
    moon, = orbit_ax.plot([x_m],[0],marker='o',markerfacecolor="gray",markeredgecolor='gray',markersize=3)

    time_label = orbit_ax.text(-0.8*x_m,0.8*x_m,'Starship, 0 Days')
    
    #collecting data 

    r0 = np.array([start_dis,0,0])
    v0 = np.array([0,start_vel,0])
    R1 = np.array([0,0,0])
    R2 = np.array([x_m,0,0])


    x,y,z = RK(grav_3_body,r0,v0,step,tot_time,5.97e24,7.347e22,R1,R2)


#updating the data at the ith frame
    def update_frame(i):

        spaceship.set_data(x[i],y[i])
        ship_line.set_data(x[:i],y[:i])
        time_label.set_text(f'Time: {"{0:.2f}".format(step*i/(60*60*24))} {time_units}')

        #print(i)
        if line == True:
            return spaceship,time_label,ship_line
        else:
            return spaceship,time_label


    anim = animation.FuncAnimation(orbit_fig,func=update_frame,frames=len(x),interval=1,blit=True)
    plt.show()


#run_sectionB(-7_000_000,1.402,250,24*60*20,'Days',True)

    # - section B simulation in 3D
    #————————————————————————————————————————————————————————————————————


def run_3D(start_dis,vel_factor,step,tot_time,time_units:str,line:bool):

    start_vel = vel_factor*min_velocity(5.97e24,abs(start_dis))
    
    x_m = 382_500_000

    orbit_fig = plt.figure(figsize=(10,10))
    orbit_ax = plt.axes(projection='3d')
    orbit_ax.set_xlim(-2*start_dis,1.2*x_m)
    orbit_ax.set_ylim(-0.5*x_m,0.5*x_m)
    orbit_ax.set_zlim(-0.5*x_m,0.5*x_m)

    spaceship, = orbit_ax.plot([],[],[],marker='<',markerfacecolor="red",markeredgecolor='red',markersize=3)
    ship_line, = orbit_ax.plot([0],[0],[0],'-g',lw=1)

    earth, = orbit_ax.plot([0],[0],[0],marker='o',markerfacecolor="blue",markeredgecolor='blue',markersize=5)
    moon, = orbit_ax.plot([x_m],[0],[0],marker='o',markerfacecolor="gray",markeredgecolor='gray',markersize=3)
    time_label = orbit_ax.text(-0.4*x_m,0.4*x_m,0.4*x_m,'Starship, 0 Days')

    r0 = np.array([start_dis,0,0])
    v0 = np.array([0,start_vel,0])
    R1 = np.array([0,0,0])
    R2 = np.array([x_m,0,0])


    x,y,z = RK(grav_3_body,r0,v0,step,tot_time,5.97e24,7.347e22,R1,R2)

    def update_frame(i):

        spaceship.set_data_3d(x[i],y[i],z[i])
        ship_line.set_data_3d(x[:i],y[:i],z[:i])
        time_label.set_text(f'Time: {"{0:.2f}".format(step*i/(60*60*24))} {time_units}')

        #print(i)
        if line == True:
            return spaceship,time_label,ship_line
        else:
            return spaceship,time_label


    anim = animation.FuncAnimation(orbit_fig,func=update_frame,frames=len(x),interval=1,blit=True)
    plt.show()

#run_3D(-7_000_000,1.402,250,24*60*10,'Days',True)


#END OF SECTION

#————————————————————————————————————————————————————————————————————

# ANALYSIS

#————————————————————————————————————————————————————————————————————

    # - chi squared function
    #————————————————————————————————————————————————————————————————————

def chi_squard(O,E):
    Y = (O-E) **2 / E
    X = np.sum(Y)
    return X

    # - plotting accuracy against time: Euler and RK4
    #————————————————————————————————————————————————————————————————————

def accuracy_measurement_circular(M,start_dis,step,tot_time):

    start_vel = min_velocity(M,start_dis)

    r_0 = np.array([start_dis,0,0])
    v_0 = np.array([0,start_vel,0])

    #methods: 

    x_RK,y_RK,z_RK = RK(grav_2_body,r_0,v_0,step,tot_time,5.97e24)
    x_Eu,y_Eu,z_Eu = Euler(grav_2_body,r_0,v_0,step,tot_time,5.97e24)
    x_An,y_An,t = circular_orbit_analytical(start_dis,5.97e24,step,tot_time)

    r_RK = np.sqrt( x_RK**2 + y_RK**2 )
    r_Eu = np.sqrt( x_Eu**2 + y_Eu**2 )
    r_An = np.sqrt( x_An**2 + y_An**2 )


    fig,ax = plt.subplots(figsize=(7.5,6))
    ax.set(ylabel='Accuracy (%)',xlabel='Time',title=f'Method accuracy h = {step}')

    print('RKchi: ',chi_squard(r_RK,start_dis))
    print('Euchi: ',chi_squard(r_Eu,start_dis))

    #accuracy calculation and time

    ax.plot(t, (1 - (abs(r_RK - start_dis) / start_dis)) * 100 ,label='RK')
    ax.plot(t, (1 - (abs(r_Eu - start_dis) / start_dis)) * 100  ,label='Euler')
    ax.legend()

    plt.show()



#accuracy_measurement_circular(5.97e24,408000,0.05,24*60*2)


    # - visual comparison of methods: Analytical vs RK4 vs Euler
    #————————————————————————————————————————————————————————————————————

#uses a same method to simulate as section A and B
#lines are default as important for comparison.

def run_compare(M,start_dis,step,tot_time,time_units:str):
    start_vel = min_velocity(M,start_dis)

    #create the figure and objects 

    orbit_fig, orbit_ax = plt.subplots(figsize=(7.5,7.5))

    orbit_ax.set_xlim(-3*start_dis,3*start_dis)
    orbit_ax.set_ylim(-3*start_dis,3*start_dis)

    spaceship_RK, = orbit_ax.plot([],[],marker='<',markerfacecolor="red",markeredgecolor='red',markersize=6)
    ship_RK_line, = orbit_ax.plot([0],[0],'-g',lw=1,color='red',label='RK4')
    spaceship_Euler, = orbit_ax.plot([],[],marker='*',markerfacecolor="green",markeredgecolor='green',markersize=6)
    ship_Euler_line, = orbit_ax.plot([0],[0],'-g',lw=1,color='green',label='Euler')
    spaceship_Ana, = orbit_ax.plot([],[],marker='<',markerfacecolor="purple",markeredgecolor='purple',markersize=6)
    ship_Ana_line, = orbit_ax.plot([0],[0],'-g',lw=1,color='purple',label='Analytical')
    orbit_ax.set(title=f'Comparing methods: h = {step}')
    orbit_ax.legend()



    planet, = orbit_ax.plot([0],[0],marker='o',markerfacecolor="blue",markeredgecolor='blue',markersize=50)
    ship_label = orbit_ax.text(-0.8*3*start_dis,0.8*3*start_dis,'Starship, 0 Days')
    
    #collecting data 
    r_0 = np.array([start_dis,0,0])
    v_0 = np.array([0,start_vel,0])

    x_RK,y_RK,z_RK = RK(grav_2_body,r_0,v_0,step,tot_time,5.97e24)
    x_Eu,y_Eu,z_Eu = Euler(grav_2_body,r_0,v_0,step,tot_time,5.97e24)
    x_An,y_An,t = circular_orbit_analytical(start_dis,5.97e24,step,tot_time)



#updating the data at the ith frame
    def update_frame(i):

        spaceship_RK.set_data(x_RK[i],y_RK[i])
        ship_RK_line.set_data(x_RK[:i],y_RK[:i])
        spaceship_Euler.set_data(x_Eu[i],y_Eu[i])
        ship_Euler_line.set_data(x_Eu[:i],y_Eu[:i])
        spaceship_Ana.set_data(x_An[i],y_An[i])
        ship_Ana_line.set_data(x_An[:i],y_An[:i])
        ship_label.set_text(f'Starship, {"{0:.2f}".format(step*i)} {time_units}')

        #print(i)
        return spaceship_RK,spaceship_Ana,spaceship_Euler,ship_label,ship_RK_line,ship_Euler_line,ship_Ana_line


    anim = animation.FuncAnimation(orbit_fig,func=update_frame,frames=len(x_RK),interval=1,blit=True)
    plt.show()


#END OF SECTION

#————————————————————————————————————————————————————————————————————

# EXTENSION

#————————————————————————————————————————————————————————————————————

    # - single iteration of RK4 ( copied the function and removed loop )
    #————————————————————————————————————————————————————————————————————

def RK_once(f_r: Callable[[np.ndarray],Any],r_0: np.ndarray,v_0: np.ndarray,h: float,*args):

    t = [0]
    r = np.array([r_0[0],r_0[1],r_0[2]])
    v = np.array([v_0[0],v_0[1],v_0[2]])


    #K_1
    #——————————————————————————————————

    k_1 = np.array([v[0],v[1],v[2]])
    k_1_v = f_r(np.array([r[0],r[1],r[2]]),*args)

    #K_2
    #——————————————————————————————————

    k_2 =  k_1 + h*k_1_v/2
    k_2_v = f_r(np.array([r[0],r[1],r[2]]) + h*k_1/2,*args)

    #K_3
    #——————————————————————————————————

    k_3 =  k_1 + h*k_2_v/2
    k_3_v = f_r(np.array([r[0],r[1],r[2]]) + h*k_2/2,*args)

    #K_4
    #——————————————————————————————————

    k_4 =  k_1 + h*k_3_v
    k_4_v = f_r(np.array([r[0],r[1],r[2]]) + h*k_3,*args)

    r_i_1 = np.array([r[0],r[1],r[2]]) + (h/6)*( k_1 + 2*k_2 + 2*k_3 + k_4)
    v_i_1 = np.array([v[0],v[1],v[2]]) + (h/6)*( k_1_v + 2*k_2_v + 2*k_3_v + k_4_v)

    return r_i_1[0],r_i_1[1],r_i_1[2],v_i_1[0],v_i_1[1],v_i_1[2]


    # - run_move, calculates and simulates moon shot in one function (not sure if correct)
    #————————————————————————————————————————————————————————————————————

def run_move(start_dis,vel_factor,step,tot_time,time_units:str,moon_x):

    start_vel = vel_factor*(min_velocity(5.97e24,abs(start_dis)))

    #create the figure and objects 

    r_m = 382_500_000

    moon_y = sqrt( r_m**2 - moon_x**2 )

    vy = -min_velocity(5.97e24,r_m) * (moon_x/r_m)
    vx = min_velocity(5.97e24,r_m) * (moon_y/r_m)

    R2_0 = np.array([moon_x,moon_y,0])
    M_vel_v = np.array([vx,vy,0])

    R2_x,R2_y,R2_z = RK(grav_2_body,R2_0,M_vel_v,step,tot_time,5.97e24)

    r = np.array([[start_dis],[0],[0]])
    v = np.array([[0],[-start_vel],[0]])
    R1 = np.array([0,0,0])

    for i in range(len(R2_x)):
        x,y,z,vx,vy,vz = RK_once(grav_3_body,np.array([r[0][i],r[1][i],r[2][i]]),np.array([v[0][i],v[1][i],v[2][i]]),step,5.97e24,7.347e22,R1,np.array([R2_x[i],R2_y[i],R2_z[i]]))

        r = np.append(r,[[x],[y],[z]],1)
        v = np.append(v,[[vx],[vy],[vz]],1)

    x,y,z = r[0],r[1],r[2]
        

    orbit_fig, orbit_ax = plt.subplots(figsize=(10,10))

    orbit_ax.set_xlim(-1.1*r_m,1.1*r_m)
    orbit_ax.set_ylim(-1.1*r_m,1.1*r_m)

    spaceship, = orbit_ax.plot([],[],marker='<',markerfacecolor="red",markeredgecolor='red',markersize=3)
    ship_line, = orbit_ax.plot([0],[0],'-g',lw=1,color='red',label='Ship')
    earth, = orbit_ax.plot([0],[0],marker='o',markerfacecolor="blue",markeredgecolor='blue',markersize=5)
    moon, = orbit_ax.plot([],[],marker='o',markerfacecolor="gray",markeredgecolor='gray',markersize=3)
    moon_line, = orbit_ax.plot([0],[0],'-g',lw=1,color='green',label='Moon')
    orbit_ax.legend()
    time_label = orbit_ax.text(-0.8*r_m,0.8*r_m,'Starship, 0 Days')


#updating the data at the ith frame
    def update_frame(i):

        spaceship.set_data(x[i],y[i])
        ship_line.set_data(x[:i],y[:i])
        moon.set_data(R2_x[i],R2_y[i])
        moon_line.set_data(R2_x[:i],R2_y[:i])
        time_label.set_text(f'Time: {"{0:.2f}".format(step*i/(60*60*24))} {time_units}')

        #print(i)
        return spaceship,time_label,moon,ship_line,moon_line


    anim = animation.FuncAnimation(orbit_fig,func=update_frame,frames=int(len(x)/2),interval=1,blit=True)
    plt.show()

#run_move(-7_000_000,1.4025,250,24*60*20,'Days',200_000_000)

#END OF SECTION

#————————————————————————————————————————————————————————————————————

# RUNNING

#————————————————————————————————————————————————————————————————————

MyInput = '0'
while MyInput != 'q':
    MyInput = input('Enter a choice, "a (SECTION A)", "b (SECTION B)", "c (ANALYSIS)", "d (EXTENSION)" or "q" to quit: ')
    print('You entered the choice: ',MyInput)

     #————————————————————————————————————————————————————————————————————
    if MyInput == 'a':
        print('————————————————————————————————————————————————————————————————————')
        print('SECTION A - 2 body orbit around the earth (set mass) (M >> m)')
        print('————————————————————————————————————————————————————————————————————')
        
        choice_input = input("Would you like to see example circular orbit around the earth? (A) or set your own parameters? (B): ")
        if choice_input == 'A':
            run_sectionA(5.97e24,408000,1,0.05,24*60*10,'Minutes',True)

        else:
            N = 0
            while N == 0:
                N_input = input('Enter a value of N steps: ')
                try:
                    N = int(N_input)
                except:
                    print('Not a valid answer')
                    N = 0

            h = 0
            while h == 0:
                h_input = input('Enter a value of time step (h): ')
                try:
                    h = float(h_input)
                except:
                    print('Not a valid answer')
                    h = 0

            start_dis = 0 
            while start_dis == 0:
                dis_input = input('Enter a starting distance (m) : ')
                try:
                    start_dis = float(dis_input)
                except:
                    print('Not a valid answer')
                    start_dis = 0

            vel_factor = 0
            while vel_factor == 0:
                vel_input = input('Enter a velocity factor (to multiply by velocity of circular orbit): ')
                try:
                    vel_factor= float(vel_input)
                except:
                    print('Not a valid answer')
                    vel_factor = 0

            line = True
            line_input = input('Would you like to see a trail (Y/N): ')
            if line_input == 'N':
                line = False


            run_sectionA(5.97e24,start_dis,vel_factor,h,N,'Minutes',line)

    #————————————————————————————————————————————————————————————————————

    if MyInput == 'b':
        print('————————————————————————————————————————————————————————————————————')
        print('SECTION B - Moon shot, stationary moon')
        print('————————————————————————————————————————————————————————————————————')
        
        choice3D_input = input("Would you like to view on a 2D (A) or 3D (B) axis: ")
        choice_input = input("Would you like to see example moon shot? (A) or set your own parameters? (B): ")
        if choice_input == 'A':
            if choice3D_input == 'B':
                run_3D(-7_000_000,1.402,250,24*60*10,'Days',True)
            else:
                run_sectionB(-7_000_000,1.402,250,24*60*20,'Days',True)

        else:
            N = 0
            while N == 0:
                N_input = input('Enter a value of N steps: ')
                try:
                    N = int(N_input)
                except:
                    print('Not a valid answer')
                    N = 0

            h = 0
            while h == 0:
                h_input = input('Enter a value of time step (h) (example: 250) : ')
                try:
                    h = float(h_input)
                except:
                    print('Not a valid answer')
                    h = 0

            start_dis = 0 
            while start_dis == 0:
                dis_input = input('Enter a starting distance (m) (example: 7_000_000) : ')
                try:
                    start_dis = float(dis_input)
                except:
                    print('Not a valid answer')
                    start_dis = 0

            vel_factor = 0
            while vel_factor == 0:
                vel_input = input('Enter a velocity factor (to multiply by velocity of circular orbit) (example: 1.402) : ')
                try:
                    vel_factor= float(vel_input)
                except:
                    print('Not a valid answer')
                    vel_factor = 0

            line = True
            line_input = input('Would you like to see a trail (Y/N): ')
            if line_input == 'N':
                line = False

            if choice3D_input == 'B':
                run_3D(-start_dis,vel_factor,h,N,'Days',line)
            else:
                run_sectionB(-start_dis,vel_factor,h,N,'Days',line)


    #————————————————————————————————————————————————————————————————————

    if MyInput == 'c':
        print('————————————————————————————————————————————————————————————————————')
        print('ANALYSIS - Circular orbit method comparison')
        print('————————————————————————————————————————————————————————————————————')
        choice_input = input('Would you like to plot accuracy (A) or visually compare methods (B): ')

        h = 0
        while h == 0:
            h_input = input('Enter a value of time step (h): ')
            try:
                h = float(h_input)
            except:
                print('Not a valid answer')
                h = 0

        if choice_input == 'A':
            accuracy_measurement_circular(5.97e24,408000,h,24*60*2)

        else:
            run_compare(5.97e24,408000,h,24*60*2,'Minutes')


    #————————————————————————————————————————————————————————————————————

    if MyInput == 'd':
        print('————————————————————————————————————————————————————————————————————')
        print('EXTENSION - Moon shot, moving moon (circular orbit)')
        print('————————————————————————————————————————————————————————————————————')


        choice_input = input("Would you like to see example moon shot (A) or set your own parameters? (B): ")
        if choice_input == 'A':
            run_move(-7_000_000,1.4024,250,24*60*20,'Days',200_000_000)

        else:
            N = 0
            while N == 0:
                N_input = input('Enter a value of N steps (example: 28000): ')
                try:
                    N = int(N_input)
                except:
                    print('Not a valid answer')
                    N = 0

            h = 0
            while h == 0:
                h_input = input('Enter a value of time step (h) (example: 250) : ')
                try:
                    h = float(h_input)
                except:
                    print('Not a valid answer')
                    h = 0

            start_dis = 0 
            while start_dis == 0:
                dis_input = input('Enter a starting distance (rocket) (m) (example: 7_000_000) : ')
                try:
                    start_dis = float(dis_input)
                except:
                    print('Not a valid answer')
                    start_dis = 0

            start_x_m = 0
            while start_x_m == 0:
                x_input = input('Enter a starting distance in x direction (moon) (m) (example: 200_000_000) : ')
                try:
                    start_x_m = float(x_input)
                except:
                    print('Not a valid answer')
                    start_x_m = 0

            vel_factor = 0
            while vel_factor == 0:
                vel_input = input('Enter a velocity factor (to multiply by velocity of circular orbit) (example: 1.4025): ')
                try:
                    vel_factor= float(vel_input)
                except:
                    print('Not a valid answer')
                    vel_factor = 0

            run_move(-start_dis,vel_factor,h,N,'Days',start_x_m)

    elif MyInput != 'q':
       print('This is not a valid choice')

print('You have chosen to finish - goodbye.')


    #END OF SECTION  

#—————————————————————————————————————————————————————————————————

#END OF PROGRAM


