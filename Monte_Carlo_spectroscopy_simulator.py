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

# Detector properties
distance = 20
height = 2.54*3 # 3 inches
radius = 2.54*3/2 # 1.5 inches

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

def detector(centre_x, centre_y, height_z, radius, distance):
    """
    Defines a cylinder with certain dimensions with a certain distance from the source, i.e. the origin.
    """
    z = np.linspace(0, height_z, 50)
    z = [i+distance for i in z]
    
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    
    x_grid = radius*np.cos(theta_grid) + centre_x
    y_grid = radius*np.sin(theta_grid) + centre_y
    return x_grid, y_grid, z_grid

def check_point_in_detector(p, radius=radius, height=height, distance=distance):
    """
    Checks if a given point is in the detector volume
    """
    if p[0]**2 + p[1]**2 <= radius**2: # Are the x and y coordinates in the circle?
        if (p[2] >= distance) and (p[2] <= height+distance): # Is the z coordinate between the distance and the height?
            return True
        else:
            return False
    else:
        return False

def sample_direction():
    """
    Samples a random unit vector assuming an isotropic distribution. Returns a list with x, y, z vector composants.
    """
    gamma = random.random()
    mu = 2*gamma-1
    gamma = random.random()
    phi = 2*np.pi*gamma
    direction = [np.sqrt(1-mu**2)*np.cos(phi), np.sqrt(1-mu**2)*np.sin(phi), mu]
    return direction

def point_of_intersection(l, pz=distance):
    """
    Determines the point of intersection between the plane of the detectors front side and the direction of the photon.
    Returns the point of intersection.
    l = vector that makes the line
    pz = the z-coordinate of the point on the plane (the detectors circular face that points towards the source)
    """
    # The definitions below assume that the detector is centred in the origin and its length is oriented along the z-axis.
    p0 = np.array([0,0,pz]) # Point on the plane
    l0 = np.array([0,0,0]) # Point on the line
    n = np.array([0,0,1]) # Normal vector of the plane
    d = np.dot(p0-l0, n)/np.dot(l, n)
    point = [i*d for i in l]
    return point

#def next_collision_point(p0, mu):
#    return p