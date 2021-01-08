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

measurement_time = 60 # seconds

# Source properties
E = 662
activity = 10000 # becquerel
rho = 3.67
tau = 8.544e-3*rho
sigma_c = 6.540e-2*rho
sigma_r = 2.671e-3*rho
kappa = 0*rho
mu = tau + sigma_c + sigma_r + kappa

# Detector properties
distance = 10
height = 2.54*3 # 3 inches
radius = 2.54*3/2 # 1.5 inches
resolution = 0.07 # Should be a function that depends on detected photon energy

class Photon:
    """
    Energy (float) = the energy of the photon
    Direction (list) = the direction vector of the photon in three dimensions
    Position (list) = the coordinates of the photon in three dimensions
    """
    def __init__(self, energy, direction, position=[0,0,0]):
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

def scattering_direction(v, theta):
    """
    Determines the direction after a scattering event given the original vector and the scattering angle theta, according to Vassilev 2.13.
    Returns list with the new vector.
    v = photon direction vector before scattering
    theta = scattering angle determined/sampled accourding to the scattering theory
    """
    # Sample cos_phi and sin_phi, phi is the azimuthal angle of the scattering event
    continue_loop = True
    while continue_loop:
        eta1 = 1-2*random.random()
        eta2 = 1-2*random.random()
        alpha = eta1**2 + eta2**2
        if alpha <= 1:
            continue_loop = False
    cos_phi = eta1/np.sqrt(alpha)
    sin_phi = eta2/np.sqrt(alpha)
    
    new_x = v[0]*np.cos(theta) - np.sin(theta)/np.sqrt(1-v[2]**2) * (v[0]*v[2]*cos_phi + v[1]*sin_phi)
    new_y = v[1]*np.cos(theta) - np.sin(theta)/np.sqrt(1-v[2]**2) * (v[1]*v[2]*cos_phi - v[0]*sin_phi)
    new_z = v[2]*np.cos(theta) + np.sqrt(1-v[2]**2)*np.sin(theta)*cos_phi
    
    return [new_x, new_y, new_z]
    
def compton(E, v):
    """
    Returns the energy of a Compton scattered photon, according to Vassilev 2.8.
    The energy is returned sampled from a Gaussian distribution. This can be disabled in the code, see comments.
    
    E = photon energy
    v = photon direction vector
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
            #eps *= 511 # Comment this line if you want the normal distribution
            new_v = scattering_direction(v, theta)
            return eps, new_v
        
def photoelectric_absorption(E):
    """
    Neglects electron binding energy. (For the moment?)
    Simply assumes a normal distribution. This will be changed later when detector theory is considered.
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

def next_interaction_point(p0, v0, mu=mu):
    """
    Determines the next interaction point of the photon in a medium.
    p0 = last known position
    v0 = photon trajectory vector
    mu = macroscopic interaction cross section (linear attenuation coefficient)
    """
    d = -1/mu * np.log(random.random()) # Path length in cm according to Vassilev section 4.2.2
    p = [p0[i] + v0[i]*d for i in range(3)]
    return p

def interaction_type(mu=mu, tau=tau, sigma_c=sigma_c, sigma_r=sigma_r, kappa=kappa):
    """
    Determines the interaction type given the macroscopic cross sections.
    Returns a string identifying the interaction type.
    mu = total cross section
    tau = photoelectric absorption cross section
    sigma_c = compton cross section
    sigma_r = rayleigh cross section
    kappa = pair production cross section
    """
    gamma = random.random()
    if gamma <= tau/mu:
        return "photoelectric absorption"
    elif (gamma > tau/mu) and (gamma <= tau/mu + sigma_c/mu):
        return "compton"
    elif (gamma > tau/mu + sigma_c/mu) and (gamma <= tau/mu + sigma_c/mu + sigma_r/mu):
        return "rayleigh"
    elif (gamma > tau/mu + sigma_c/mu + sigma_r/mu) and (gamma <= tau/mu + sigma_c/mu + sigma_r/mu + kappa/mu):
        return "pair production"

def rayleigh(v0):
    """
    Takes the direction vector of the photon and samples a new direction according to Rayleigh scattering, by using the Rayleigh phase function to sample theta.
    The function uses the rejection method described in Vassilev 2.4.
    v0 = current direction vector.
    Returns new direction vector.
    """
    # Need to sample the angle theta from the phase function
    loop_condition = True
    while loop_condition:
        eps = random.random()*np.pi # Sampled x coordinate from 0 to pi
        eta = random.random()*(3/4)*2 # Sampled y coordinate from 0 to max of Rayleigh phase function for unpolarised light
        if eta < 3/4*(1 + (np.cos(eps))**2): # Checks if eta is less than the Rayleigh phase function using the angle eps
            loop_condition = False
            
    # Get a new direction vector for the photon
    v = scattering_direction(v0, eps)
    return v

def mag(x): 
    return np.sqrt(sum(i**2 for i in x))

def main():
    #counts = []
    for i in range(measurement_time): # Loop over the number of seconds
        no_of_photons = emitted_counts_this_second(activity) # Sample number of emitted photons
        for i in range(int(no_of_photons)): # Create a photon and follow it until it is outside the detector
            # Create new photon
            photon = Photon(E, sample_direction()) # sample photon with random direction
            
            # Change the position of the photon to the point of intersection with the plane of the detector face
            photon.change_position(point_of_intersection(photon.direction))
            
            # Check if the photon is in the detector, or is on the surface of the detector
            if check_point_in_detector(photon.position): # It has been tested that 6-7% of the photons hit the detector
                # Sample new interaction point using the sampled path length
                photon.change_position(next_interaction_point(photon.position, photon.direction, mu))
                
                # Check if the initial point is in the detector
                loop_condition = check_point_in_detector(photon.position)
                energy_to_be_deposited = 0
                while loop_condition:
                    # Determine interaction type
                    interaction = interaction_type()
                    if interaction == "photoelectric absorption":
                        energy_to_be_deposited += photoelectric_absorption(photon.energy)
                        loop_condition = False
                    elif interaction == "compton":
                        # Determine energy and the direction of the scattered photon
                        new_energy, new_direction = compton(photon.energy, photon.direction)
                        energy_to_be_deposited += photon.energy - new_energy
                        
                        # Change the energy and direction of the photon
                        photon.change_energy(new_energy)
                        photon.change_direction(new_direction)
                        
                        # Sample new interaction point and check whether it is in the detector
                        photon.change_position(next_interaction_point(photon.position, photon.direction))
                        loop_condition = check_point_in_detector(photon.position)
                        #loop_condition = False # Need to write something that return new interaction cross sections to do multiple compton scatterings.
                        
                    elif interaction == "rayleigh":
                        # Determine direction of the scattered photon
                        new_direction = rayleigh(photon.direction)
                        photon.change_direction(new_direction)
                        
                        # Sample new interaction point and check whether it is in the detector
                        photon.change_position(next_interaction_point(photon.position, photon.direction))
                        loop_condition = check_point_in_detector(photon.position)
                        
                    elif interaction == "pair production":
                        # Not written yet, sample new interaction
                        loop_condition = True
                        
                if energy_to_be_deposited != 0:
                    counts.append(energy_to_be_deposited)
                    
# Things to do:
                    # energy dependent cross sections
                        # cross sections could be set as attributes in the Photon class
                    # pair production
                    # backscattering?

if __name__ == "__main__":
    counts = []
    main()
    plt.hist(counts, 100)
    plt.xlim(0,1400)
    plt.yscale("log")
    plt.grid()
            