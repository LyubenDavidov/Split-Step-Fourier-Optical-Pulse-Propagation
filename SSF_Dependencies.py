# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:52:36 2024

@author: Lyuben Davidov
@email1: l.davidov@tue.nl
@email2: ljubo.davidov@tue.nl

Dependencies for the FDML Nonlinear Schrödinger Equation
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
c0 = 3e8

# Bezier Curve Points for the Input/Output Gain
x0 = 20
y0 = 10

x1 = 0.07
y1 = 5.91

x2 = -10
y2 = 3
 
x3 = -40
y3 = -32


a = 77
b = 10
c = 26
d = 67


# The input "power" is in dBm
# Power is for 500um SOA length
def G_dBin_dBout(power):
    if power > 0 and power <= 10:
        p_out = a * np.exp( -((power-b)**2)/(2*c**2)) - d
    elif power <= 0 and power > -8:
        p_out = 0.564*power + 4.512
    elif power <= -8:
        p_out = power + 8
    elif power > 10:
        p_out = 10
    return p_out


# The input "power" is in dBm
# Power is for 500um SOA length
def G_Bezier(x, omega):
    
    # 'x' is the input power in dBm at a frequency omega    
    s = (4*(-9*x1**2+9*x2*x1+9*x3*x1-9*x2**2+9*x0*x2-9*x0*x3)**3+(-54*x1**3+243*x*x1**2+81*x2*x1**2-162*x3*x1**2+81*x2**2*x1-162*x*x0*x1-486*x*x2*x1+81*x0*x2*x1+162*x*x3*x1+81*x0*x3*x1+81*x2*x3*x1-54*x2**3+27*x*x0**2+243*x*x2**2-162*x0*x2**2+27*x*x3**2-27*x0*x3**2+162*x*x0*x2-27*x0**2*x3-54*x*x0*x3-162*x*x2*x3+81*x0*x2*x3)**2)**0.5
    c = np.cbrt(-54*x1**3+243*x*x1**2+81*x2*x1**2-162*x3*x1**2+81*x2**2*x1-162*x*x0*x1-486*x*x2*x1+81*x0*x2*x1+162*x*x3*x1+81*x0*x3*x1+81*x2*x3*x1-54*x2**3+27*x*x0**2+243*x*x2**2-162*x0*x2**2+27*x*x3**2-27*x0*x3**2+162*x*x0*x2-27*x0**2*x3-54*x*x0*x3-162*x*x2*x3+81*x0*x2*x3+s)
    t = -(x0-2*x1+x2)/(-x0+3*x1-3*x2+x3)+c/(3*np.cbrt(2)*(-x0+3*x1-3*x2+x3))-(np.cbrt(2)*(-9*x1**2+9*x2*x1+9*x3*x1-9*x2**2+9*x0*x2-9*x0*x3))/(3*(-x0+3*x1-3*x2+x3)*c)
    y = (1-t)*((1-t)*((1-t)*y0+t*y1)+t*((1-t)*y1+t*y2))+t*((1-t)*((1-t)*y1+t*y2)+t*((1-t)*y2+t*y3))
    
    # Introduces wavelength/frequency dependence to the output gain in dBm
    _lambda_nm = (2*np.pi*c0/omega)*1e9
    y_wvl = y - 0.003143*_lambda_nm**2 + 9.687*_lambda_nm - 7464
    return y_wvl


def u_Gain(P_Gain): 
    return 10**(P_Gain/20)
    


# Important: dz is in microns, not meters
def Gain_Calc(P0,dz,omega):
    
    # Input Power in dBm
    P0 = P0 + 1e-323 #
    P0_dBm = 10*np.log10([P0])                          # [dBm]
    #print("Wavelength in nm", 1e9*(2*np.pi*c0)/(omega))
    #print("Input Power in dBm",P0_dBm)
    
    # Output Power depending on Input Power and Wavelength
    Pout_dBm = G_Bezier(P0_dBm, omega)                      # [dBm]
    #print("Output Power in dBm, 500um SOA",Pout_dBm)
    
    # Power Gain for a 500um long SOA with J = 3kA/cm^2
    G_500um = Pout_dBm - P0_dBm                             # [dB]
    
    # Power Gain per meter
    G_dB_m  = G_500um / (500e-6)                            # [dB/m]
    
    # Power Gain per micron
    G_dB_um = G_dB_m / 1e6                                  # [dB/um]
    #print("Power Gain in dB/um",G_dB_um)
    
    # Linear Power Gain per micron
    G_lin_um = 10**(G_dB_um/10)                             # [1/m]
    
    # Power Gain per SOA segment dz
    G_out = G_dB_um * dz                                    # [-]
    
    '''
    The experimentally-measured gain is for the optical power, which is the
    squared absolute of the optical field envelope u. To get the gain for the
    optical field envelope u, you need to divide by 20, instead of 10. If you
    still want to acquire the output optical power, you can square u_out.
    '''
    
    gain_factor = u_Gain(G_out)
    
    return gain_factor, G_dB_um

'''
# Wavelength/Frequency
omega_ = (2*np.pi*c0)/(1555e-9)


# Output power for Input Power 0.1mW and 500 microns of gain medium    
print("u_out [sqrt(W)]",Gain_Calc(np.sqrt(1e-4), 500, omega_)[1])



print(G_dBin_dBout(-20))

print(G_Bezier(-20, omega_))


# Parabola fit for the Power Transmission Spectrum of the TF
omega_c = 1.2152967e+15
delta_omega = 0.02202925e+15

x = np.array([omega_c - delta_omega/2, omega_c, omega_c + delta_omega/2])
y = np.array([-10, -0.200, -10])
A,B,C = np.polyfit(x, y, 2)
print(A,B,C)

A = -8.077679989452024e-26                           # [dB/Hz^2]
B = 1.9633555669674626e-10                          # [db/Hz]
C = -119303.17707311233                             # [dB]

Omega_T = (0.0214534e+15)/2


def TF_dB_PT(omega0, omega):
    
    # omega0      = central frequency of the filter
    # dOmega_TR   = width between the left and right -10dB points of the parabola
    # dOmega_TUNE = width of the tuning range of the filter
    
    dOmega_TR   = 0.02202925e+15                          # [Hz]
    dOmega_TUNE = 0.0214534e+15                           # [Hz]
    
    x = np.array([omega0 - dOmega_TR/2, omega0, omega0 + dOmega_TR/2])
    y = np.array([-10, -0.200, -10])
    A,B,C = np.polyfit(x, y, 2)
    
    res = A * (omega0 + (omega0 - omega))**2 + B *(omega0 + (omega0 - omega)) + C
    return res

omega_CC = 1.25e+15
print(TF_dB_PT(omega0 = omega_CC, omega = omega_CC+0.5*0.02202925e+15))




def parabola(x_0):
    def y(x):
        return -2/5 * (x - x_0) ** 2
    return y

# Example usage:
x_0 = 1550  # Central peak at x = 1550nm
x_m = -5
x_p = +5
parabola_function = parabola(x_0)

# Test the function with some values
print(parabola_function(x_0))  # Should output 0
print(parabola_function(x_0 + x_m))  # Should output -10
print(parabola_function(x_0 + x_p))  # Should output -10
'''


# Plots the Bezier Curve's Gain vs Wavelength Responce
def Gain_Wavl_plot():
    lambda_min      = 1500e-9;       # shortest wavelength we calculate for [m]
    lambda_max      = 1600e-9;       # longest wavelength we calculate for  [m]
    lambda_centr    = 1550e-9;       # central wavelength [m]
    
    omega_min   = 2 * np.pi * c0 / lambda_min
    omega_0     = 2 * np.pi * c0 / lambda_centr
    omega_max   = 2 * np.pi * c0 / lambda_max
    
    x = np.linspace(-100,20,121)
    Bezier_out_min = G_Bezier(x, omega_min)
    Bezier_out_cen = G_Bezier(x, omega_0)
    Bezier_out_max = G_Bezier(x, omega_max)
    
    plt.plot(x, Bezier_out_min, x, Bezier_out_cen, x, Bezier_out_max)
    plt.legend([r'$\lambda=1500nm$',r'$\lambda=1550nm$',r'$\lambda=1600nm$'])
    plt.xlabel("Input Power [dBm]")
    plt.ylabel("Output Power [dBm]")
    plt.title(r'500$\mu$m SOA Output Power')
        