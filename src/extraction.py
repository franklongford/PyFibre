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
import sknw
import networkx as nx

from scipy import interpolate
from scipy.misc import derivative
from scipy.ndimage import distance_transform_edt
from scipy.signal import argrelextrema

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import threshold_otsu

import image_tools as it


def FIRE(image):

	filtered = image > threshold_otsu(image)
	cleared = it.clear_border(filtered)

	distance = distance_transform_edt(cleared)
	nodes = argrelextrema(distance, np.greater)

	print(nodes)



