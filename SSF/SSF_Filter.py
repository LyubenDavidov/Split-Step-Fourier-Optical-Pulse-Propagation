# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:10:22 2024

@author: 20235303
"""



import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt

from SSF_Dependencies import Gain_Calc, G_Bezier



# Speed of light in vacuum (m/s)
c0 = 3e8
global pi; pi=np.pi 

# Wavelength range (in meters) (in vacuum)
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
