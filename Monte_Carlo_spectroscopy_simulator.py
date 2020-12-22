# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 19:35:30 2020

@author: Ivan52x53

Monte Carlo simulation of a gamma-ray distribution measured by a detector.
Written according to the methods and algorithms described in 
Vassilev 2017 - Monte Carlo Methods for Radiation Therapy
"""

import numpy as np
import random
import scipy.integrate
import matplotlib.pyplot as plt

E = 1332
resolution = 0.1 # Should be a function that depends on detected photon energy

class Photon:
    """
    Energy (float) = the energy of the photon
    Direction (list) = the direction vector of the photon in three dimensions
    Position (list) = the coordinates of the photon in three dimensions
    """
    def __init__(self, energy, direction, position):
        self.energy = energy
        self.direction = direction
        self.position = position
        
    def change_energy(self, new_energy):
        self.energy = new_energy
        
    def change_direction(self, new_direction):
        self.direction = [new_direction[i] for i in range(len(new_direction))]
    
    def change_position(self, new_position):
        self.position = [new_position[i] for i in range(len(new_position))]

def normal_distribution(E, E_full_peak, resolution=resolution):
    """
    Normal distribution according to Vassilev 2.10 Method 1.
    Returns energy sampled from normal distribution.
    Can be used for other things, but be mindful of the definition of the "resolution" parameter.
    
    E = photon energy
    E_full_peak = energy value of the full peak, i.e. original photon energy
    resolution = resolution of the detector at that energy
    """
    n = 12 # Recommended by literature
    mu = E # Energy of the photon in question
    sigma = E_full_peak*resolution/2.35 # Should be whatever the detector resolution is at that energy.
    gammas = [random.random()-0.5 for i in range(n)]
    delta = np.sqrt(12/n)*sum(gammas)
    eps = sigma*delta+mu
    return eps
    
def compton(E):
    """
    Returns the energy of a Compton scattered photon, according to Vassilev 2.8.
    The energy is returned sampled from a Gaussian distribution. This can be disabled in the code, see comments.
    
    E = photon energy
    """
    E = E/511 # Change to energy in units of electron mass
    continue_loop = True
    while continue_loop:
        # Determine Emin, Emax
        #E = 140 # keV
        Emin = E/(1+2*E)
        Emax = E
        
        # Determine normalisation constants
        A1_f = lambda E: E
        A2_f = lambda E: 1/E
        A1 = 1/scipy.integrate.quad(A1_f, Emin, Emax)[0]
        A2 = 1/scipy.integrate.quad(A2_f, Emin, Emax)[0]
        
        # Determine alpha
        a1 = 1/(A1*E)
        a2 = E/A2
        
        gamma = random.random() # Random num to determine which distribution eps should be sampled from
        if gamma < a1/(a1+a2):
            gamma = random.random() # Need to sample new gamma to sample an eps from this distribution
            eps = np.sqrt(Emin**2 + 2*gamma/A1)
        else:
            gamma = random.random() # Need to sample new gamma to sample an eps from this distribution
            eps = Emin*np.exp(gamma/A2)
            
        # Determine theta using eps and E
        sin2 = 1 - (1 - 1/eps + 1/E)**2
        theta = np.arcsin(np.sqrt(1 - (1 - 1/eps + 1/E)**2))
        
        gamma = random.random()
        if gamma < (1 - sin2/(eps/E + E/eps)): # I honestly don't know what this condition is. Refer to Vassilev ch. 2.8
            continue_loop = False
            eps = normal_distribution(eps,E)*511 # Comment this line to get rid of normal distribution. 511 returns us to units of keV
            return eps, theta
        
def photoelectric_absorption(E):
    """
    Neglects electron binding energy.
    """
    return normal_distribution(E,E)

def emitted_counts_this_second(A, uncertainty=0):
    """Determines a random amount of emitted counts in a single second according to a normal distribution around the given activity A.
    A = activity
    uncertainty = uncertainty in A (percentage)
    
    returns number of emitted counts in a single second
    """
    emitted = normal_distribution(A, A, uncertainty)
    return emitted

"""
# Plotting results
# Compton
data_compton = [compton(E) for i in range(10000)]
photon_energies_compton, theta = zip(*data_compton)
energies_compton = [E-i for i in photon_energies_compton]

#data_compton2 = [compton(1173) for i in range(10000)]
#photon_energies_compton2, theta2 = zip(*data_compton2)
#energies_compton2 = [E-i for i in photon_energies_compton2]

# Photo
energies_photo = [photoelectric_absorption(E) for i in range(6000)]
#energies_photo2 = [photoelectric_absorption(1173) for i in range(6000)]

energies = energies_photo + energies_compton


#plt.hist(energies)
fig, ax = plt.subplots()
ax.hist(energies, 500, color="black")
#ax.set_yscale("log")
#ax.set_xscale("log")
ax.set_xlabel("Energy [keV]")
ax.set_ylabel("Counts")

A = 1000 # Bq
emitted_cps = [emitted_counts_this_second(A,0.03*2.35) for i in range(10000)]
fig2, ax2 = plt.subplots()
ax2.hist(emitted_cps,100)
"""