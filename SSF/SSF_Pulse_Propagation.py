# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:00:13 2024

@author: 20235303
"""

import numpy as np
from scipy.fft import fft, ifft

# Define the time grid
T = 10  # Total time window
N = 1024  # Number of time points
dt = T / N  # Time step size
t = np.linspace(-T/2, T/2, N)

# Initial pulse: Gaussian pulse
A0 = np.exp(-t**2 / 2)

# Define the frequency grid
omega = np.fft.fftfreq(N, d=dt) * 2 * np.pi

# Parameters
dz = 0.1                                                         # [mm] = 100um
num_steps = 100
v_g = 2e8  # Group velocity (m/s)  
gamma = 1.0 # Example nonlinear coefficient

def beta(omega):
    return -0.2 * omega**2  # Example for standard fiber

def gain_loss_function(omega):
    return 10**(-0.034/20)   # Example: linear loss per 100um (-3.4dB/cm)

def time_dependent_effect(A, t_abs):
    # Example time-dependent effect: apply a phase shift depending on absolute time
    phase_shift = np.exp(1j * 0.1 * t_abs)
    return A * phase_shift

def ssfm(A0, dz, num_steps, beta, gamma, v_g, gain_loss_function=None, time_dependent_segments=None):
    A = A0
    t_abs = 0  # Initialize absolute time
    for n in range(num_steps):
        # Update absolute time
        t_abs += dz / v_g
        
        # Fourier transform of the pulse
        A_hat = np.fft.fftshift(fft(A))
        
        # Linear step with dispersion
        A_hat *= np.exp(1j * beta(omega) * dz)
        
        # Apply gain/loss if provided
        if gain_loss_function is not None:
            A_hat *= np.exp(dz * -0.457 * 0.1) #gain_loss_function(omega) 
        
        # Inverse Fourier transform to time domain
        
        A = ifft(A_hat)

        
        # Nonlinear step
        A *= np.exp(1j * gamma * np.abs(A)**2 * dz)
        
        # Apply time-dependent effect if in specified segments
        if time_dependent_segments and n in time_dependent_segments:
            A = time_dependent_effect(A, t_abs)
    
    return A

# Example usage
time_dependent_segments = [10, 20, 30]  # Apply time-dependent effect at these segment indices

# Run the SSFM
A_final = ssfm(A0, dz, num_steps, beta, gamma, v_g, gain_loss_function, time_dependent_segments)

# To visualize the result, you can use plotting libraries like matplotlib
import matplotlib.pyplot as plt

plt.figure()
plt.plot(t, np.abs(A0), t, np.abs(A_final))
plt.legend(['Initial','Propagated'])
plt.title('Pulse after propagation')
plt.xlabel('Relative Time')
plt.ylabel('Amplitude')
plt.show()

# Calculate the energy of the pulse
energy_initial = np.trapz(np.abs(A0)**2, t)
energy_final = np.trapz(np.abs(A_final)**2, t)

print("Initial energy of the pulse: {:.2f} mJ".format(energy_initial))
print("Final energy of the pulse: {:.2f} mJ".format(energy_final))
