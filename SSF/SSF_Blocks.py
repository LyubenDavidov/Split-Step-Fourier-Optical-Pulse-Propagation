# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:41:55 2024

@author: 20235303
"""


import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

from SSF_Dependencies import Gain_Calc, G_Bezier


# Speed of light in vacuum (m/s)
c0 = 3e8

# Wavelength range (in meters)
lambda_min = 1530e-9
lambda_max = 1570e-9
lambda_0   = 1550e-9

# Convert wavelength range to frequency range
f_min = c0 / lambda_max
f_max = c0 / lambda_min
f_0   = c0 / lambda_0
omega_0 = 2 * np.pi * f_0

# Calculate the bandwidth in frequency domain (Hz)
delta_f = f_max - f_min

# Convert the bandwidth to angular frequency (rad/s)
delta_omega = 2 * np.pi * delta_f

# Estimate the pulse duration in seconds (for a Gaussian pulse with TBP ~ 0.44)
delta_t = 0.44 / delta_f

# Total time window should be several times the pulse duration (let's use 5 times)
T = 20*delta_t  #!!! If you increase T, you must increase N approproately

# Number of time points (N) can be chosen to balance resolution and computational cost
N = 2048  # This is an arbitrary choice; can be increased for higher resolution

# Sampling frequency
fs = N / T
print("FFT Sampling Frequency is: {:.3e} Hz".format(fs))

# Frequency bin
delta_f = fs/N
delta_omega = delta_f * 2 * np.pi

# Time step size
dt = T / N

# Time axis
t = np.linspace(-T/2, T/2, N)

# Define the initial Gaussian pulse
factor = 2*np.sqrt(2*np.log(2))
A0 = np.exp(-t**2 / (2 * (delta_t / factor)**2))  # FWHM is related to sigma by factor 2.355 for Gaussian

# Define the frequency grid (angular frequency)
f = np.fft.fftfreq(N, d=dt)  # Frequency in Hz
omega = 2 * np.pi * f        # Angular frequency in rad/s




# Parameters
dz = 15.5                                            # [um] Spatial step size
num_steps = 645                                      # Number of spatial points/segments
n_g = 3.665                                          # Group Index of Refraction for 1550nm
v_g = c0/n_g                                         # [m/s] Group velocity

gamma = 36.85e-6                                     # Nonlinear coefficient [1/W/um] 
# Source: S. Saeidi et al., "Demonstration of Optical Nonlinearity in InGaAsP/InP Passive Waveguides"


# Power Loss
loss_m = -340                                        # Waveguide loss per meter [dB]
loss = loss_m * dz * 1e-6                            # Waveguide loss per segment [dB]

# Apply Dispersion
D = 500e-6                                           # Fiber dispersion [s/m/m], originally [ps/nm/km]

def beta(omega):
    return -1j * np.pi * c0 * D * (omega/omega_0)**2 # Group Velocity Dispersion [1/m]

# Define Gain/Loss Function 
def gain_loss_function(gain, loss, omega):
    # Divisor is 20 to covert Power Loss to Envelope Loss     
    return 10**((gain+loss)/20)   

def gamma_nonlinearity(A):
    return np.exp(1j * gamma * np.abs(A)**2 * dz)

# Define Absolute-Time Time-Dependent Effects (i.e. Tunable Filter)
def time_dependent_effect(A, t_abs):
    # Example time-dependent effect: apply a phase shift depending on absolute time
    phase_shift = np.exp(01j * 0.01 * t_abs)
    return A * phase_shift

# Define Output FFT Plots
def out_plot(omega, A0, A_hat):
    A0_FFT = np.fft.fftshift(fft(A0))
    AF_FFT = np.fft.fftshift(A_hat)
    omega_centered = np.fft.fftshift(omega)
    
    # Magnitude Squared
    A0_FFT_M = delta_omega * (np.abs(A0_FFT)**2)/((2*np.pi*fs)*N)
    AF_FFT_M = delta_omega * (np.abs(AF_FFT)**2)/((2*np.pi*fs)*N)
    
    # Phase
    A0_FFT_P = np.angle(A0_FFT)*180/np.pi
    AF_FFT_P = np.angle(AF_FFT)*180/np.pi
    
    
    # Plot
    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
    
    # Plot the Magnitude
    ax1.plot(omega_centered*1e-12/(2*np.pi), A0_FFT_M * 1e3, omega_centered*1e-12/(2*np.pi), AF_FFT_M * 1e3)
    ax1.legend(['Initial','Propagated'])
    ax1.set_title('Power Spectrum of the Optical Envelope')
    #ax1.set_xlabel(r'Frequency $\nu$ [THz]')
    ax1.set_xlim([-30, 30])
    ax1.set_ylabel(r'Magnitude [$\mu W$]')
    #plt.plot()
    
    
    # Plot the Phase
    ax2.plot(omega_centered*1e-12/(2*np.pi), A0_FFT_P, omega_centered*1e-12/(2*np.pi), AF_FFT_P)
    ax2.legend(['Initial','Propagated'])
    ax2.set_title('Phase Angle')
    ax2.set_xlabel(r'Frequency $\nu$ [THz]')
    ax2.set_xlim([-30, 30])
    ax2.set_ylabel(r'Phase [Deg]')
    plt.plot()
    

# Define SSF Algorithm
def ssfm(A0, dz, num_steps, beta, v_g, gamma=None, gain_loss_function=None, time_dependent_segments=None, gain_segments=None):
    A = A0
    t_abs = 0  # Initialize absolute time
    for n in range(num_steps):
        # Update absolute time
        t_abs += dz / v_g
        
        # Fourier transform of the pulse
        A_hat = fft(A)

        # Linear step with dispersion
        A_hat *= np.exp(beta(omega) * dz * 1e-6)
        
        # Apply gain/loss if provided
        if gain_loss_function is not None and n not in gain_segments:
                A_hat += (gain_loss_function(0, loss, omega) - 1)*A_hat 
        
        if gain_segments and n in gain_segments:
            #if n == gain_segments[0]:
            PSD = (np.abs(A_hat)**2)/((2*np.pi*fs)*N)
            PS = PSD * delta_omega
            '''
            Gain_Calc takes as input the power of a certain frequency/wavelength
            of light at the entrance of a 500um SOA gain block and returns the 
            G_dB_um (Power Gain in dB per micron). This gain can later be used
            for amplification without the need to recalculate the Power Spectrum
            (PS) at every gain segment. The PS is only needed at the entrance to
            the gain block.
            '''
            
            # Power Gain per Segment 
            dB_Gain = dz * Gain_Calc(PS, dz, omega + omega_0)[1]
            dB_Gain = np.transpose(dB_Gain) #+ 1e-323
        
            print("Gain per segment for 1550nm in dB: {}".format(dB_Gain[0]))
            #print(gain_loss_function(dB_Gain, loss, omega)[:,0])
            #print("Power Gain in dB per Segment is: {}".format(dB_Gain))
            #_loss_ = loss * np.ones((len(omega)))
            A_hat += (gain_loss_function(dB_Gain, loss, omega)[:,0] - 1)*A_hat 
            #else:
            #    A_hat += (gain_loss_function(dB_Gain, loss, omega)[:,0] - 1)*A_hat 
        
        # Inverse Fourier transform to time domain
        A = ifft(A_hat)
        
        # Apply Kerr Nonlinearity
        if gamma is not None:
            A *= gamma(A)
        
        # Apply time-dependent effect if in specified segments
        if time_dependent_segments and n in time_dependent_segments:
            A = time_dependent_effect(A, t_abs)
            
        if n == num_steps-1:
            # Plot the FFTs
            out_plot(omega, A0, A_hat)
            

    return A, t_abs, PS

# Example usage
gain_segments = tuple(range(640, 645+1))
time_dependent_segments = [10, 20, 30]  # Apply time-dependent effect at these segment indices

# Run the SSFM
A_final = ssfm(A0, dz, num_steps, beta, v_g, gamma_nonlinearity, gain_loss_function, time_dependent_segments, gain_segments)
PS = A_final[2]

# Visualize Results
plt.figure()
plt.plot(t*1e15, np.abs(A0), t*1e15, np.abs(A_final[0]))
plt.legend(['Initial','Propagated'])
plt.title('Pulse after propagation')
plt.xlabel(r'Relative Time [fs]')
plt.ylabel(r'Amplitude $\sqrt{mW}$')
plt.show()

# Print Propagation Time
print("Total Absolute Propagation Time {:.2e}us".format(A_final[1]))

# Calculate the energy of the pulse
energy_initial = np.trapz(np.abs(A0)**2, t*1e15)
energy_final = np.trapz(np.abs(A_final[0])**2, t*1e15)

print("Initial energy of the pulse: {:.2f} aJ".format(energy_initial))
print("Final energy of the pulse: {:.2f} aJ".format(energy_final))




'''

# Speed of light in vacuum (m/s)
c = 3e8

# Wavelength range (in meters)
lambda_min = 1530e-9
lambda_max = 1570e-9

# Convert wavelength range to frequency range
f_min = c / lambda_max
f_max = c / lambda_min

# Calculate the bandwidth in frequency domain (Hz)
delta_f = f_max - f_min

# Convert the bandwidth to angular frequency (rad/s)
delta_omega = 2 * np.pi * delta_f

# Estimate the pulse duration in seconds (for a Gaussian pulse with TBP ~ 0.44)
delta_t = 0.44 / delta_f

# Total time window should be several times the pulse duration (let's use 5 times)
T = 5 * delta_t

# Number of time points (N) can be chosen to balance resolution and computational cost
N = 1024  # This is an arbitrary choice; can be increased for higher resolution

# Time step size
dt = T / N

# Time axis
t = np.linspace(-T/2, T/2, N)

# Define the initial Gaussian pulse
A0 = np.exp(-t**2 / (2 * (delta_t / 2.355)**2))  # FWHM is related to delta_t by factor 2.355 for Gaussian

# Define the frequency grid (angular frequency)
f = np.fft.fftfreq(N, d=dt)  # Frequency in Hz
omega = 2 * np.pi * f  # Angular frequency in rad/s

# Plot the initial pulse
plt.figure()
plt.plot(t, np.abs(A0))
plt.title('Initial Gaussian Pulse')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Print calculated values for verification
print(f"Minimum frequency: {f_min / 1e12:.2f} THz")
print(f"Maximum frequency: {f_max / 1e12:.2f} THz")
print(f"Bandwidth: {delta_f / 1e12:.2f} THz")
print(f"Pulse duration: {delta_t * 1e15:.2f} fs")
print(f"Total time window: {T * 1e15:.2f} fs")
print(f"Time step size: {dt * 1e15:.2f} fs")
print(f"Number of points: {N}")
'''


# Unwrap the phase of the optical pulse
def getPhase(pulse):
    phi=np.unwrap(np.angle(pulse)) #Get phase starting from 1st entry
    phi=phi-phi[int(len(phi)/2)]   #Center phase on middle entry
    return phi    

# Get the chirp of the optical pulse
def getChirp(time,pulse):
    phi=getPhase(pulse)
    dphi=np.diff(phi, prepend = phi[0] - (phi[1]  - phi[0]  ), axis=0) #Change in phase. Prepend to ensure consistent array size   
    dt  =np.diff(time, prepend = time[0]- (time[1] - time[0] ), axis=0) #Change in time.  Prepend to ensure consistent array size
    return -1.0/(2*np.pi)*dphi/dt                                     #Chirp = - 1/(2pi) * d(phi)/dt

plt.figure()
plt.title("Chirp of the pulses")
plt.plot(t*1e15,getChirp(t,A0)*1e-12,t*1e15,getChirp(t, A_final[0])*1e-12)
plt.legend(['Initial Pulse', 'Final Pulse'])
plt.xlabel("Time [fs]")
plt.ylabel("Chirp [THz]")
#plt.axis([-duration*5*1e12,duration*5*1e12,-1/duration/1e9,1/duration/1e9])
#plt.legend(bbox_to_anchor=(1.05,0.8))
plt.show()