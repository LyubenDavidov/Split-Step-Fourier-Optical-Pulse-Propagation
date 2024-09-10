# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:02:49 2024

@author: 20235303
"""







import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from SSF_Dependencies import Gain_Calc, G_Bezier

# Constants
c0 = 3e8  # Speed of light in vacuum (m/s)

# Helper Functions
def beta(omega, D, c0, omega_0):
    return -1j * np.pi * c0 * D * (omega/omega_0)**2  # Group Velocity Dispersion [1/m]

def gain_loss_function(gain, loss, omega):
    return 10**((gain + loss)/20)  # Convert Power Loss to Envelope Loss

def gamma_nonlinearity(A, gamma, dz):
    return np.exp(1j * gamma * np.abs(A)**2 * dz)

def time_dependent_effect(A, t_abs):
    phase_shift = np.exp(1j * 0.01 * t_abs)
    return A * phase_shift

def out_plot(omega, A0, A_hat, delta_omega, fs, N):
    A0_FFT = np.fft.fftshift(fft(A0))
    AF_FFT = np.fft.fftshift(A_hat)
    omega_centered = np.fft.fftshift(omega)

    A0_FFT_M = delta_omega * (np.abs(A0_FFT)**2)/((2*np.pi*fs)*N)
    AF_FFT_M = delta_omega * (np.abs(AF_FFT)**2)/((2*np.pi*fs)*N)
    A0_FFT_P = np.angle(A0_FFT)*180/np.pi
    AF_FFT_P = np.angle(AF_FFT)*180/np.pi

    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)

    ax1.plot(omega_centered*1e-12/(2*np.pi), A0_FFT_M * 1e3, omega_centered*1e-12/(2*np.pi), AF_FFT_M * 1e3)
    ax1.legend(['Initial','Propagated'])
    ax1.set_title('Power Spectrum of the Optical Envelope')
    ax1.set_xlim([-30, 30])
    ax1.set_ylabel(r'Magnitude [$\mu W$]')

    ax2.plot(omega_centered*1e-12/(2*np.pi), A0_FFT_P, omega_centered*1e-12/(2*np.pi), AF_FFT_P)
    ax2.legend(['Initial','Propagated'])
    ax2.set_title('Phase Angle')
    ax2.set_xlabel(r'Frequency $\nu$ [THz]')
    ax2.set_xlim([-30, 30])
    ax2.set_ylabel(r'Phase [Deg]')
    plt.show()

# Segment Base Class
class Segment:
    def propagate(self, A, omega, t_abs):
        raise NotImplementedError()

class GainSegment(Segment):
    def __init__(self, dz, omega, loss, gain_segments):
        self.dz = dz
        self.omega = omega
        self.loss = loss
        self.gain_segments = gain_segments

    def propagate(self, A, omega, t_abs, n):
        if n in self.gain_segments:
            A_hat = fft(A)
            PSD = (np.abs(A_hat)**2)/((2*np.pi*fs)*N)
            PS = PSD * delta_omega
            dB_Gain = self.dz * Gain_Calc(PS, self.dz, omega + omega_0)[1]
            A_hat += (gain_loss_function(dB_Gain, self.loss, omega)[:, 0] - 1) * A_hat
            A = ifft(A_hat)
        return A

class LossSegment(Segment):
    def __init__(self, dz, beta_func):
        self.dz = dz
        self.beta_func = beta_func

    def propagate(self, A, omega, t_abs, n):
        A_hat = fft(A)
        A_hat *= np.exp(self.beta_func(omega) * self.dz * 1e-6)
        return ifft(A_hat)

class NonlinearSegment(Segment):
    def __init__(self, gamma, dz):
        self.gamma = gamma
        self.dz = dz

    def propagate(self, A, omega, t_abs, n):
        return A * gamma_nonlinearity(A, self.gamma, self.dz)

class TimeDependentSegment(Segment):
    def __init__(self, time_dependent_segments):
        self.time_dependent_segments = time_dependent_segments

    def propagate(self, A, omega, t_abs, n):
        if n in self.time_dependent_segments:
            A = time_dependent_effect(A, t_abs)
        return A

# Ring Cavity Class
class RingCavity:
    def __init__(self, segments, dz, num_steps, v_g):
        self.segments = segments
        self.dz = dz
        self.num_steps = num_steps
        self.v_g = v_g

    def propagate_pulse(self, A0, omega):
        A = A0
        t_abs = 0  # Initialize absolute time
        for n in range(self.num_steps):
            t_abs += self.dz / self.v_g
            for segment in self.segments:
                A = segment.propagate(A, omega, t_abs, n)
            if n == self.num_steps-1:
                out_plot(omega, A0, fft(A), delta_omega, fs, N)
        return A, t_abs

# Setup the simulation parameters
lambda_min = 1530e-9
lambda_max = 1570e-9
lambda_0 = 1550e-9

f_min = c0 / lambda_max
f_max = c0 / lambda_min
f_0 = c0 / lambda_0
omega_0 = 2 * np.pi * f_0
delta_f = f_max - f_min
delta_omega = 2 * np.pi * delta_f

delta_t = 0.44 / delta_f
T = 20 * delta_t
N = 2048

fs = N / T
dt = T / N
t = np.linspace(-T/2, T/2, N)

factor = 2 * np.sqrt(2 * np.log(2))
A0 = np.exp(-t**2 / (2 * (delta_t / factor)**2))

f = np.fft.fftfreq(N, d=dt)
omega = 2 * np.pi * f

# Parameters for the segments
dz = 15.5  # [um] Spatial step size
num_steps = 645  # Number of spatial points/segments
n_g = 3.665  # Group Index of Refraction for 1550nm
v_g = c0/n_g  # [m/s] Group velocity

gamma_val = 36.85e-6  # Nonlinear coefficient [1/W/um]
loss_m = -340  # Waveguide loss per meter [dB]
loss = loss_m * dz * 1e-6  # Waveguide loss per segment [dB]
D = 500e-6  # Fiber dispersion [s/m/m]

# Define the ring cavity configuration
segments = [
    LossSegment(dz, lambda omega: beta(omega, D, c0, omega_0)),
    GainSegment(dz, omega, loss, gain_segments=tuple(range(640, 645+1))),
    NonlinearSegment(gamma_val, dz),
    TimeDependentSegment(time_dependent_segments=[10, 20, 30]),
]

cavity = RingCavity(segments, dz, num_steps, v_g)

# Run the simulation
A_final, t_abs = cavity.propagate_pulse(A0, omega)

# Visualize Results
plt.figure()
plt.plot(t*1e15, np.abs(A0), t*1e15, np.abs(A_final))
plt.legend(['Initial','Propagated'])
plt.title('Pulse after propagation')
plt.xlabel(r'Relative Time [fs]')
plt.ylabel(r'Amplitude $\sqrt{mW}$')
plt.show()

# Print Propagation Time
print("Total Absolute Propagation Time {:.2e}us".format(t_abs))

# Calculate the energy of the pulse
energy_initial = np.trapz(np.abs(A0)**2, t*1e15)
energy_final = np.trapz(np.abs(A_final)**2, t*1e15)

print("Initial energy of the pulse: {:.2f} aJ".format(energy_initial))
print("Final energy of the pulse: {:.2f} aJ".format(energy_final))





























'''
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from SSF_Dependencies import Gain_Calc, G_Bezier


#================================
# Create Classes for the Segments

class Segment:
    def propagate(self, pulse):
        """Propagate the pulse through the segment.
        This should be overridden by subclasses."""
        raise NotImplementedError()

class GainSegment(Segment):
    def __init__(self, gain_coefficient):
        self.gain_coefficient = gain_coefficient

    def propagate(self, pulse):
        # Implement the propagation logic for gain
        return pulse * self.gain_coefficient

class LossSegment(Segment):
    def __init__(self, loss_coefficient, dispersion):
        self.loss_coefficient = loss_coefficient
        self.dispersion = dispersion

    def propagate(self, pulse):
        # Implement the propagation logic for loss with dispersion
        return pulse * self.loss_coefficient  # Include dispersion logic
    


#=========================    
# Create Ring-Cavity Class

class RingCavity:
    def __init__(self, segments):
        self.segments = segments

    def propagate_pulse(self, pulse):
        for segment in self.segments:
            pulse = segment.propagate(pulse)
        return pulse
'''