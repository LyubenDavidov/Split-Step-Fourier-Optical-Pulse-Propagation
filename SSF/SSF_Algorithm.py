# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:47:39 2024

@author: 20235303
"""


import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt

from SSF_Dependencies import Gain_Calc, G_Bezier



# Speed of light in vacuum (m/s)
c0 = 3e8
global pi; pi=np.pi 

# Wavelength range (in meters)
lambda_min = 1530e-9
lambda_max = 1570e-9
lambda_0   = 1550e-9

# Convert wavelength range to frequency range
f_min = c0 / lambda_max
f_max = c0 / lambda_min
f_0   = c0 / lambda_0
omega_0 = 2 * np.pi * f_0

# Define simulation parameters
N  = 2**15                                              # Number of points
dt = 0.1e-15                                            # Time resolution [s]

def getFreqRangeFromTime(time):
    return fftshift(fftfreq(len(time), d=time[1]-time[0]))


t=np.linspace(0,N*dt,N) #Time step array
t=t-np.mean(t)          #Center so middle entry is t=0

f_bad = fftfreq(len(t), d=t[1]-t[0]) 
f=getFreqRangeFromTime(t)



plt.figure()
plt.plot(f_bad,label="f_bad")
plt.plot(f,label="f")
plt.legend()
plt.show()

#NOTE: scipy is (for some reason) coded so the first entry is freq=0, the next 
#(N-1)/2 are all the positive frequencies and the remaining ones are the 
#negative ones. The function, fftshift, rearranges the entries so negative 
#frequencies come first


def getPhase(pulse):
    phi=np.unwrap(np.angle(pulse)) #Get phase starting from 1st entry
    phi=phi-phi[int(len(phi)/2)]   #Center phase on middle entry
    return phi    


def getChirp(time,pulse):
    phi=getPhase(pulse)
    dphi=np.diff(phi ,prepend = phi[0] - (phi[1]  - phi[0]  ),axis=0) #Change in phase. Prepend to ensure consistent array size 
    dt  =np.diff(time,prepend = time[0]- (time[1] - time[0] ),axis=0) #Change in time.  Prepend to ensure consistent array size

    return -1.0/(2*pi)*dphi/dt #Chirp = - 1/(2pi) * d(phi)/dt


