# -*- coding: utf-8 -*-
"""
_______________________TITLE_______________________
PHYS20161 - Final Assignment - Doppler Spectroscopy
___________________________________________________

This python script reads in .csv files. It then validates the data by removing
non number and anomalous data points that lie outside of 5 standard deviations
of the mean.

The script then performs a minimised chi squared fit. If the reduced chi
squared result is unacceptable, the script will remove data points that lie
too far from the expected curve.

Finally, once obtaining values for V_0 (the velocity of the star) and Omega
(the angular velocity of the orbit), the script calculates the radius of
the orbit (in astronomical units) and the mass (in Jovian masses) of the
planet.


Lewis Zigante 08/12/2020
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import scipy.constants as sp

# Constants
JOVIAN_MASS = 1.898E27
ASTRONOMICAL_UNIT = sp.astronomical_unit
GRAVITATIONAL_CONSTANT = sp.gravitational_constant
SPEED_OF_LIGHT = sp.speed_of_light
LAMBDA_0 = 656.281
PHI = sp.pi
MASS_OF_SUN = 1.988E30

# Enter the file names here
FILE_1 = 'doppler_data_1.csv'
FILE_2 = 'doppler_data_2.csv'

def read_in_data():
    '''
    This function reads in the data from .csv files. It checks if the files
    may be found, then creates two arrays of the data and combines them. This
    combined array is returned.

    Returns
    -------
    wavelength_data : Array of Floats
        Array containing all the data from the .csv files.

    Raises:
        FileNotFoundError: When a file or directory is requested but doesn't
        exist

    '''
    try:
        file_1 = open(FILE_1, 'r')
        file_2 = open(FILE_2, 'r')
        file_1.close()
        file_2.close()
    except FileNotFoundError:
        print('File/s not found. Please ensure files are in same file '
              'location as the script.')
        sys.exit()

    doppler_data_1 = np.genfromtxt(FILE_1, delimiter=',',
                                   comments='%')
    doppler_data_2 = np.genfromtxt(FILE_2, delimiter=',',
                                   comments='%')
    wavelength_data = np.vstack((doppler_data_1, doppler_data_2))

    return wavelength_data

def standard_deviation(wavelength_data):
    '''
    Calculates the standard deviation of the set of data.

    Parameters
    ----------
    wavelength_data : Array of floats

    Returns
    -------
    std : Float
        The standard deviation.
    mean : Float
        The mean.

    '''
    mean = np.average(wavelength_data[:, 1])
    std = np.std(wavelength_data[:, 1])

    return std, mean

def data_validation(wavelength_data):
    '''
    Removes non-number points and values that lie outside of 5 standard
    deviations of the mean.
    Returns a validated and cleaned array of the data.

    Parameters
    ----------
    wavelength_data : Array of floats

    Returns
    -------
    wavelength_data : Array of floats
        Validated and cleaned array of all the data.

    '''
    wavelength_data = wavelength_data[~np.isnan(wavelength_data).any(axis=1)]
    row_delete = np.array([], dtype="int")
    std, mean = standard_deviation(wavelength_data)
    mean_plus_5_sigma = mean + 5 * std
    mean_minus_5_sigma = mean - 5 * std
    for i in range(len(wavelength_data)):
        if wavelength_data[i, 2] == 0:
            row_delete = np.append(row_delete, i)

        if (wavelength_data[i, 1] > mean_plus_5_sigma
                or wavelength_data[i, 1] < mean_minus_5_sigma):
            row_delete = np.append(row_delete, i)
    wavelength_data = np.delete(wavelength_data, row_delete, 0)

    return wavelength_data

def time_array(wavelength_data):
    '''
    Creates a 1D array of the time values and returns this array. This is
    done to fix a broadcasting error when using fmin.

    Parameters
    ----------
    wavelength_data : Array of floats

    Returns
    -------
    time_values : array of floats
        An array of just the time values

    '''
    times = np.array(wavelength_data[:, 0])
    time_values = np.array([])
    for _, time in enumerate(times):
        time_value = time * 31536000
        time_values = np.append(time_values, time_value)

    return time_values

def wavelength_calculation(time, v_0, omega):
    '''
    Calculates the wavelength according to the expected equation and returns
    the value.

    Parameters
    ----------
    time : float

    v_0 : float
        The velocity of the star
    omega : float
        The angular velocity of the orbit

    Returns
    -------
    float
        The expected wavelength according to the model equation.

    '''
    return (((SPEED_OF_LIGHT + (v_0 * np.sin((omega * time) + PHI)))
             /SPEED_OF_LIGHT) * LAMBDA_0)

def chi_squared(v_0omega_array, wavelength_data, times):
    '''
    Calculates the chi squared value.

    Parameters
    ----------
    v_0omega_array : array of floats
        An array containing the velocity and angular velocity
    wavelength_data : array of floats
        DESCRIPTION.
    times : array of floats
        DESCRIPTION.

    Returns
    -------
    float
        The chi squared value.

    '''
    v_0 = v_0omega_array[0]
    omega = v_0omega_array[1]
    chi_squared_array = np.array([])
    for i, time in enumerate(times):
        chi_squared_value = ((wavelength_calculation(time, v_0, omega) -
                              wavelength_data[i, 1])
                             / wavelength_data[i, 2]) ** 2
        chi_squared_array = np.append(chi_squared_array, chi_squared_value)

    return np.sum(chi_squared_array)

def chi_squared_outliers(wavelength_data, v_0, omega, times, counter):
    '''
    Will be called if the reduced chi squared calculated as a result of the
    minimised chi squared fit is unacceptable (> 2). This function removes
    the points from the wavelength data array that lie greater than a given
    distance from the expected curve that the data points are being compared
    against.

    Parameters
    ----------
    wavelength_data : array of floats

    v_0 : float

    omega : float

    times : array of floats

    counter : int
        Increases with each successive fmin calculation that produces an
        unsatisfactory reduced chi squared

    Returns
    -------
    wavelength_data : array of floats

    '''
    number_of_uncertainties_from_fit = 5
    row_delete = np.array([], dtype="int")
    for i, time in enumerate(times):
        point_distance_test = np.abs(((wavelength_calculation(time, v_0, omega)
                                       - wavelength_data[i, 1])
                                      / (2 * wavelength_data[i, 2])))
        if point_distance_test > (number_of_uncertainties_from_fit - counter):
            row_delete = np.append(row_delete, i)
    wavelength_data = np.delete(wavelength_data, row_delete, 0)
    print('Data points that lie greater than {0} times uncertainty from the '
          'fit have been removed. Number of data points removed = {1}'
          .format((number_of_uncertainties_from_fit - counter),
                  len(row_delete)))

    return wavelength_data

def reduced_chi_squared_calculation(chi_squared_minimised, times):
    '''
    Returns the reduced chi squared value.

    Parameters
    ----------
    chi_squared_minimised : float

    times : array of floats

    Returns
    -------
    float
        The reduced chi squared value.

    '''


    return chi_squared_minimised / (len(times)-2)

def chi_squared_mesh(v_0, omega, wavelength_data):
    chi_squared_array = ([])
    for i in range(len(v_0)):
        for j in range(len(omega)):
            chi_squared_value = np.sum(((wavelength_calculation(wavelength_data[:,0]*31536000, v_0[i][j], omega[i][j]) -
                              wavelength_data[:, 1])
                             / wavelength_data[:, 2]) ** 2)
            chi_squared_array = np.append(chi_squared_array, chi_squared_value)

    return chi_squared_array.reshape(len(v_0), len(omega))

def radius_of_orbit(omega):
    '''
    Returns the radius of the orbit of the planet in metres.

    Parameters
    ----------
    omega : float

    Returns
    -------
    float

    '''
    period = (2 * np.pi) / omega

    return np.cbrt(((GRAVITATIONAL_CONSTANT * MASS_OF_SUN * 2.78) /
                    (4 * np.pi**2)) * period**2)

def planet_mass_calculation(v_0, radius):
    '''
    Returns the mass of the planet in kg.

    Parameters
    ----------
    v_0 : float

    radius : float

    Returns
    -------
    float

    '''

    velocity_planet = np.sqrt((GRAVITATIONAL_CONSTANT * MASS_OF_SUN * 2.78)
                              /(radius))

    return (MASS_OF_SUN * 2.78 * v_0)/(velocity_planet)

def main():
    '''
    Main function. Takes the wavelength data; performs a minimised chi
    squared fit; and uses the resulting values for V_0 and omega to calculate
    the mass of the planet and the radius of the orbit.

    Returns
    -------
    int

    '''
    wavelength_data = read_in_data()
    wavelength_data = data_validation(wavelength_data)

    start_v_0 = 50
    start_omega = 3E-8
    counter = 0
    while True:
        time_values = time_array(wavelength_data)
        chi_squared_minimised_fit = fmin(chi_squared, (start_v_0, start_omega),
                                         args=(wavelength_data, time_values),
                                         full_output=True)

        [v_0_minimum, omega_minimum] = chi_squared_minimised_fit[0]
        chi_squared_minimum = chi_squared_minimised_fit[1]
        reduced_chi_squared = reduced_chi_squared_calculation(
            chi_squared_minimum, time_values)

        if reduced_chi_squared > 2:
            print('Reduced chi squared = {0:.3f}'.format(reduced_chi_squared))
            print()
            print('This reduced chi squared value is too large: removing '
                  'potential outliers and repeating minimisation...')
            print()
            counter += 1
            wavelength_data = chi_squared_outliers(wavelength_data, start_v_0,
                                                   start_omega, time_values,
                                                   counter)
        else:
            break

    # Plot of predicted wavelength and observed wavelengths using v_0_minimum
    # and omega_minimum found from minimised chi squared
    figure_wavelength = plt.figure()
    axes = figure_wavelength.add_subplot(111)
    axes.set_title(r'Plot of $\lambda$ against time, $\chi^2_{{\mathrm{{red.'
                   '}}}}$ = {0:.3f}'.format(reduced_chi_squared))
    axes.set_xlabel('Time (s)')
    axes.set_ylabel(r'$\lambda$ (nm)')
    axes.errorbar(time_values, wavelength_data[:, 1],
                  yerr=wavelength_data[:, 2],
                  linestyle="None", color="orange", label="Error bar")
    axes.scatter(time_values, wavelength_data[:, 1],
                 label=("Observed wavelength"))
    time_values_ordered = np.sort(time_values)
    axes.plot(time_values_ordered, wavelength_calculation(time_values_ordered,
                                                          v_0_minimum,
                                                          omega_minimum),
              dashes=[4, 2], color="k", label="Predicted wavelength")
    axes.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('wavelength_against_time', dpi=300)
    plt.show()

    v_0_values = np.linspace(v_0_minimum-0.05*v_0_minimum,v_0_minimum+0.05*v_0_minimum,250)
    omega_values = np.linspace(omega_minimum-0.05*omega_minimum, omega_minimum + 0.05*omega_minimum,250)
    v_0_mesh, omega_mesh = np.meshgrid(v_0_values, omega_values)
    chi_squared_array = chi_squared_mesh(v_0_mesh, omega_mesh, wavelength_data)
    
    PARAMETERS_CONTOUR_FIGURE = plt.figure()

    PARAMETERS_CONTOUR_PLOT = PARAMETERS_CONTOUR_FIGURE.add_subplot(111)

    PARAMETERS_CONTOUR_PLOT.set_title(r'$\chi^2$ contours against parameters.',
                                      fontsize=14)
    PARAMETERS_CONTOUR_PLOT.set_xlabel('velocity of star, v (m/s)', fontsize=10)
    PARAMETERS_CONTOUR_PLOT.set_ylabel(r'angular velocity of orbit, $\omega$ (rad/s)', fontsize=10)
    
    
    # Place minimum as single point
    PARAMETERS_CONTOUR_PLOT.scatter(v_0_minimum, omega_minimum,
                                    label='Minimum')
    
    
    # chi^2 min + 1 contour, treated separately as we want it dashed.
    PARAMETERS_CONTOUR_PLOT.contour(v_0_mesh, omega_mesh,
                                    chi_squared_array,
                                    levels=[chi_squared_minimum + 1.00],
                                    linestyles='dashed',
                                    colors='k')

    # Contours to be plotted
    # Ideally these numbers would be defined elsewhere so they are easy to ammend
    # without having to ammend several things.
    CHI_SQUARED_LEVELS = (chi_squared_minimum + 2.30, chi_squared_minimum + 5.99)
    
    CONTOUR_PLOT = PARAMETERS_CONTOUR_PLOT.contour(v_0_mesh, omega_mesh, chi_squared_array, levels=CHI_SQUARED_LEVELS)
    LABELS = ['Minimum', r'$\chi^2_{{\mathrm{{min.}}}}+1.00$',
              r'$\chi^2_{{\mathrm{{min.}}}}+2.30$',
              r'$\chi^2_{{\mathrm{{min.}}}}+5.99$']

    PARAMETERS_CONTOUR_PLOT.clabel(CONTOUR_PLOT)
    
    # Want plot legend outside of plot area, need to adjust size of plot so that
    # it is visible
    
    BOX = PARAMETERS_CONTOUR_PLOT.get_position()
    PARAMETERS_CONTOUR_PLOT.set_position([BOX.x0, BOX.y0, BOX.width * 0.7,
                                          BOX.height])
    
    # Add custom plot labels
    for index, label in enumerate(LABELS):
        PARAMETERS_CONTOUR_PLOT.collections[index].set_label(label)
    PARAMETERS_CONTOUR_PLOT.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                                    fontsize=14)
    
    plt.savefig('contour_plot_3.png', dpi=300)
    
    plt.show()
    
    
    # Calculations
    radius = radius_of_orbit(omega_minimum)
    planet_mass = planet_mass_calculation(v_0_minimum, radius)

    # Print Results
    print('Reduced chi squared = {0:.3f}'.format(reduced_chi_squared))
    print('V_0 = {0:.4g} m/s'.format(v_0_minimum))
    print('Omega = {0:.4g} rad/s'.format(omega_minimum))
    print('radius of the orbit = {0:.4g} AU'.format(radius /
                                                    ASTRONOMICAL_UNIT))
    print('Mass of the planet = {0:.4g} Jovian masses'.format(planet_mass
                                                              / JOVIAN_MASS))

    return 0

if __name__ == '__main__':
    main()
