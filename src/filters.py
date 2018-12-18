"""
ColECM: Collagen ExtraCellular Matrix Simulation
ANALYSIS ROUTINE 

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 19/04/2018
"""

import sys, os
import numpy as np
import scipy as sp
import networkx as nx

from scipy import interpolate
from scipy.misc import derivative
from scipy.ndimage import filters, imread, sobel

from skimage.feature import hessian_matrix, hessian_matrix_eigvals



def gaussian(image, sigma=None):

	if sigma != None: return filters.gaussian_filter(image, sigma=sigma)
	else: return image


def spiral_tv(image):

	pass


def tubeness(image, sigma=None):

	H_elems = hessian_matrix(image, order="xy", sigma=sigma, mode='wrap')
	H_eigen = hessian_matrix_eigvals(H_elems)
	tube = np.where(H_eigen[1] < 0, abs(H_eigen[1]), 0)

	return tube


def curvelet(image):

	pass


def vesselness(eig1, eig2, beta1=0.1, beta2=0.1):

	A = np.exp(-(eig1/eig2)**2 / (2 * beta1))
	B = (1 - np.exp(- (eig1**2 + eig2**2) / (2 * beta2)))

	return A * B

