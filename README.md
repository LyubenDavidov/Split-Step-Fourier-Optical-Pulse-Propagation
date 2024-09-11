# SSF-Optical-Pulse-Propagation



## Introduction: End Goal

This simulation models a Fourier Domain Mode Locked (FDML) laser using the Split Step Fourier method. The laser cavity consists of a finite number of building blocks, including Semiconductor Optical Amplifiers (SOAs), a tunable filter, and a long waveguide delay line. It is designed to be self-starting from spontaneous emission. Each block in the cavity is defined by its type, length, material parameters, and arrangement, allowing flexibility in simulating different laser configurations. The goal is to analyze the dynamic behavior of the FDML laser system.

## Current Status:

You can model transform-limited optical pulse propagation in dispersive nonlinear medium, where you can define certain segments as gain-segments (SOA segments), and you can also add real-time effects (Tunable Filtering). The propagation is unidirectional and doesn't include reflections. In the plotted results you can observe the initial vs propagated pulse in Time-Domain, the Power Spectral Density of the two, as well as, the chirp of the propagated pulse.

## File Overview:

`SSF_Propagation.py : Transform-limited optical pulse propagation in dispersive nonlinear medium.

`SSF_Dependencies.py: Some functions calculating the gain of the SOA given the wavelength and the input power at the entrance. (As per Smart Design Manual)

`SSF_Laser_Sim.ipynb: Simulation of an FDML Laser - Work in progress...

## Authors:

Lyuben Davidov
