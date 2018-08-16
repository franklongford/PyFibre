"""
ColECM: Collagen ExtraCellular Matrix Simulation
MACHINE LEARNING ROUTINE 

Created by: Frank Longford
Created on: 13/06/2018

Last Modified: 13/06/2018
"""

import numpy as np
import scipy as sp

from sklearn import metrics
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering, DBSCAN, FeatureAgglomeration, Birch
from sklearn.metrics.pairwise import pairwise_distances

from keras.models import Model, Sequential, load_model # basic class for specifying and training a neural network
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, to_categorical # utilities for one-hot encoding of ground truth values
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3d
import matplotlib.animation as animation

import sys, os, re, time,itertools

import utilities as ut
from analysis import select_samples
import pickle
import setup


class cnn_model:

	def __init__(self, model_path=None, classes=None, image_shape=None, ow_model=False):

		self.model_path = model_path

		try: self.load_model()
		except IOError:
			try: self._init_model(classes, image_shape)
			except: 
				self.model = None
				self.classes = list(classes)
				self.image_shape = image_shape

	def _init_model(self, classes, image_shape):

		self.classes = list(np.unique(classes))
		self.n_classes = len(self.classes)
		self.class_range = np.arange(self.n_classes)
		self.image_shape = image_shape

		self._create_cnn_model()

		param = {'model_path' : self.model_path, 'classes' : self.classes, 'image_shape': self.image_shape}
		pickle.dump(param, open(self.model_path + '.pkl', 'wb'))

		self.model.summary()
		self.save_model()

	def save_model(self):

		self.model.save(self.model_path)

	def load_model(self):

		self.model = load_model(self.model_path)

		param = pickle.load(open(self.model_path + '.pkl', 'rb'))

		self.model_path = param['model_path']
		self.classes = param['classes']
		self.image_shape = param['image_shape']
		self.n_classes = len(self.classes)

	def delete_model(self):
	
		try: 
			os.remove(self.model_path)
			os.remove(self.model_path + '.pkl')

			self.model = None
			self.classes = None
			self.image_shape = None

		except: pass
			
	def _create_cnn_model(self):

		kernel_size = (3, 3) # we will use 3x3 kernels throughout
		pool_size = (2, 2) # we will use 2x2 pooling throughout
		conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
		conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
		conv_depth_3 = 128
		conv_depth_4 = 256 
		drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
		drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
		hidden_size = 512 # the FC layer will have 512 neurons

		self.model = Sequential()

		self.model.add(Conv2D(conv_depth_1, kernel_size, padding='same', activation='relu', input_shape=self.image_shape))
		self.model.add(Conv2D(conv_depth_1, kernel_size, activation='relu'))
		self.model.add(MaxPooling2D(pool_size=pool_size))
		self.model.add(Dropout(drop_prob_2))

		self.model.add(Conv2D(conv_depth_2, kernel_size, padding='same', activation='relu'))
		self.model.add(Conv2D(conv_depth_2, kernel_size, activation='relu'))
		self.model.add(MaxPooling2D(pool_size=pool_size))
		self.model.add(Dropout(drop_prob_1))

		self.model.add(Conv2D(conv_depth_3, kernel_size, padding='same', activation='relu'))
		self.model.add(Conv2D(conv_depth_3, kernel_size, activation='relu'))
		self.model.add(MaxPooling2D(pool_size=pool_size))
		self.model.add(Dropout(drop_prob_1))

		self.model.add(Flatten())
		self.model.add(Dense(hidden_size, activation='relu'))
		self.model.add(Dropout(drop_prob_2))
		self.model.add(Dense(self.n_classes, activation='softmax'))

		self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


	def format_data(self, data_set):

		n_sample = data_set.shape[0]

		try: data_set = data_set.reshape((n_sample,) + self.image_shape)
		except: data_set = data_set[:, :self.image_shape[0], :self.image_shape[1]]

		data_set /= np.max(data_set)

		return data_set


	def input_training_data(self, data_set, data_labels):

		batch_size = 32 # in each iteration, we consider 32 training examples at once
		num_epochs = 5 # we iterate 200 times over the entire training set

		classes = np.unique(data_labels)
		image_shape = data_set.shape[1:] + (1,)

		if self.model is None: self._init_model(classes, image_shape)

		data_set = self.format_data(data_set)
		one_hot_labels = to_categorical([self.classes.index(label) for label in data_labels], num_classes=self.n_classes)

		(training_set, test_set, training_labels, test_labels) = train_test_split(data_set,
			one_hot_labels, test_size=0.2, random_state=42)

		"""
		datagen = ImageDataGenerator(rotation_range=20)

		# compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied)
		datagen.fit(training_set)
		# fits the model on batches with real-time data augmentation:
		history = self.model.fit_generator(datagen.flow(training_set, training_labels, batch_size=batch_size), 
						steps_per_epoch=len(training_set)//batch_size, epochs=num_epochs)
		#datagen.fit(test_set)
		#score = self.model.evaluate_generator(datagen.flow(test_set, test_labels, batch_size=batch_size))
		score = self.model.evaluate(test_set, test_labels)
		"""
		history = self.model.fit(training_set, training_labels, batch_size=batch_size, epochs=num_epochs, verbose=1)
		score = self.model.evaluate(test_set, test_labels)
		#"""
		print(' Test Loss:', score[0])
		print(" Test Accuracy: {:.2f}%".format(score[1]*100))

		self.save_model()


	def input_prediction_data(self, data_set):

		data_set = self.format_data(data_set)

		score = self.model.predict(data_set)
		scores = 100 * np.sum(score, axis=0) / np.sum(score)

		print("\n Prediction Results:")
		for i, score in enumerate(scores): print(" Class {}: {:.2f}%".format(self.classes[i], score))


def print_cnn_samples(fig_dir, fig_name, n, title, images, n_col, n_row, image_shape):

	print('\n Creating CNN Gallery {}{}_cnn.png'.format(fig_dir, fig_name))

	plt.figure(n, figsize=(2. * n_col, 2.26 * n_row))
	plt.suptitle(title, size=16)

	for i, comp in enumerate(images):
		plt.subplot(n_row, n_col, i + 1)
		vmax = max(comp.max(), -comp.min())
		plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
			   interpolation='nearest',
			   vmin=-vmax, vmax=vmax)
		plt.xticks(())
		plt.yticks(())

	plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
	plt.savefig('{}{}_cnn.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close('all')


def print_nmf_results(fig_dir, fig_name, n, title, images, n_col, n_row, image_shape):

	print('\n Creating NMF Gallery {}{}_nmf.png'.format(fig_dir, fig_name))

	plt.figure(n, figsize=(2. * n_col, 2.26 * n_row))
	plt.suptitle(title, size=16)

	for i, comp in enumerate(images):
		plt.subplot(n_row, n_col, i + 1)
		vmax = max(comp.max(), -comp.min())
		plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
			   interpolation='nearest',
			   vmin=-vmax, vmax=vmax)
		plt.xticks(())
		plt.yticks(())

	plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
	plt.savefig('{}{}_nmf.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close('all')


def print_cnn_training_results(history):

	# Display results
	fig = plt.figure(figsize=(8, 5)) 
	axes = np.zeros((2, 4), dtype=np.object)

	axes[0, 0] = fig.add_subplot(2, 4, 1)

	for i in range(1, 4):
	    axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
	for i in range(0, 4):
	    axes[1, i] = fig.add_subplot(2, 4, 5+i)

	ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
	ax_img.set_title('Low contrast image')
	y_min, y_max = ax_hist.get_ylim()
	ax_hist.set_ylabel('Number of pixels')
	ax_hist.set_yticks(np.linspace(0, y_max, 5))
	ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
	ax_img.set_title('Contrast stretching')
	ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
	ax_img.set_title('Histogram equalization')
	ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
	ax_img.set_title('Adaptive equalization')
	ax_cdf.set_ylabel('Fraction of total intensity')
	ax_cdf.set_yticks(np.linspace(0, 1, 5))
	# prevent overlap of y-axis labels
	fig.tight_layout()
	plt.show()


def convolutional_neural_network_analysis(model_name, model_dir, classes=None, predict_set=None, data_set=None, data_labels=None, ow_model=False):
	"""
	fourier_transform_analysis(image_shg, area, n_sample)

	Calculates fourier amplitude spectrum of over area^2 pixels for n_samples

	Parameters
	----------

	"""

	model = cnn_model(model_path=model_dir + model_name, classes=classes)

	if ow_model: model.delete_model()

	if data_set is not None: model.input_training_data(data_set, data_labels)

	if predict_set is not None:
		if len(predict_set) > 1:
			for data_set in predict_set: model.input_prediction_data(data_set)
		else: model.input_prediction_data(predict_set[0])

	return 


def nmf_analysis(data_set, n_components):
	"""
	nmf_analysis(data_set, n_sample)

	Calculates non-negative matrix factorisation of over area^2 pixels for n_samples

	Parameters
	----------

	image_shg:  array_like (float); shape=(n_images, n_x, n_y)
		Array of images corresponding to each trajectory configuration

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample


	Returns
	-------

	"""

	print("\n Performing NMF Analysis")

	rng = np.random.RandomState(0)

	model = NMF(n_components=n_components, init='random', random_state=0)

	data_set = data_set.reshape(data_set.shape[0], data_set.shape[1] * data_set.shape[2])
	W = model.fit_transform(data_set)
	H = model.components_

	return np.dot(W, H)


def hierarchical_clustering(data_set, data_labels, n_clusters, classes):
	"""
	nmf_analysis(data_set, n_sample)

	Calculates non-negative matrix factorisation of over area^2 pixels for n_samples

	Parameters
	----------

	image_shg:  array_like (float); shape=(n_images, n_x, n_y)
		Array of images corresponding to each trajectory configuration

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample


	Returns
	-------

	"""

	print("\n Performing Clustering Analysis")

	rng = np.random.RandomState(0)

	n_sample = data_set.shape[0]

	samples = np.random.choice(data_set.shape[0], n_sample)
	#samples = np.arange(n_sample)
	labels_true = np.array([classes.index(label) for label in data_labels])[samples]
	sample_set = data_set[samples].reshape(n_sample, data_set.shape[1] * data_set.shape[2])

	db = Birch(n_clusters=n_clusters).fit(sample_set/ np.max(sample_set))
	labels = db.labels_
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	scores = []

	labels = np.array(labels)
	new_labels = []

	for item in itertools.permutations(np.arange(n_clusters)):
		new_labels.append(np.copy(labels))
		for i, n in enumerate(item):
			indices = np.argwhere(labels == n) 
			new_labels[-1][indices] = i
		scores.append(np.mean(np.where(new_labels[-1] == labels_true, 1, 0)))

	correct = np.max(scores)
	labels = new_labels[np.argmax(scores)]

	print(' Estimated number of clusters: %d' % n_clusters_)
	print(" Correct Prediction: {} %".format(100 * correct))
	print(" Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
	print(" Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
	print(" V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
	print(" Adjusted Rand Index: %0.3f"
	      % metrics.adjusted_rand_score(labels_true, labels))
	print(" Adjusted Mutual Information: %0.3f"
	      % metrics.adjusted_mutual_info_score(labels_true, labels))
	print(" Silhouette Coefficient: %0.3f \n"
		% metrics.silhouette_score(sample_set, labels))

	return sample_set.reshape(n_sample, data_set.shape[1], data_set.shape[2]), [classes[i] for i in labels]


def get_data_set(file_names, fig_dir, data_list, image_size, n_images, nmf=False):


	for i, file_name in enumerate(file_names):
		data_list[i], _ = select_samples(data_list[i], image_size, 1)
		samples = np.random.choice(data_list[i].shape[0], n_images)
		#print_cnn_samples(fig_dir, file_name, 12, 'CNN Sample Selection', data_list[i][samples], 
		#			np.sqrt(n_images), np.sqrt(n_images), (image_size, image_size))
		#"""
		if nmf:
			"Perform Non-Negative Matrix Factorisation"
			n_components = 2
			data_list[i] = nmf_analysis(data_list[i], n_components).reshape(data_list[i].shape)
		#	print_nmf_results(fig_dir, file_name, 12, 'NMF filtered Images', data_list[i][samples], 
		#			np.sqrt(n_images), np.sqrt(n_images), (image_size, image_size))
		#"""

	return data_list


def learning(current_dir):

	data_dir = current_dir + '/data/'
	fig_dir = current_dir + '/fig/'
	model_dir = current_dir + '/model/'

	ow_mod = ('-ow_model' in sys.argv)
	nmf = ('-nmf' in sys.argv)
	cluster = ('-cluster' in sys.argv)

	print("\n " + " " * 5 + "----Beginning Machine Learning Analysis----\n")
	print("\n Algorithms used:\n Non-Negative Matrix Factorisation from scikit_learn library\n Convoluted Neural Network from keras library\n")
	if not os.path.exists(data_dir): os.mkdir(data_dir)
	if not os.path.exists(fig_dir): os.mkdir(fig_dir)
	if not os.path.exists(model_dir): os.mkdir(model_dir)

	train_file_names = []
	train = []
	train_ref = []

	predict_file_names = []
	predict = []

	n_images = 16

	if ('-model' in sys.argv): model_name = sys.argv[sys.argv.index('-model') + 1]
	else: model_name = 'colecm_cnn_model'

	if ('-train' in sys.argv):
		for arg in sys.argv[sys.argv.index('-train')+1:]: 
			if not re.match('-', arg): train_file_names.append(arg)
			else: break
	if ('-predict' in sys.argv): 
		for arg in sys.argv[sys.argv.index('-predict')+1:]: 
			if not re.match('-', arg): predict_file_names.append(arg)
			else: break
	if ('-classes' in sys.argv): 
		classes = []
		for arg in sys.argv[sys.argv.index('-classes')+1:]: 
			if not re.match('-', arg): classes.append(arg)
			else: break
	else: classes = np.arange(len(train_file_names))
	anis = ('-anis' in sys.argv)

	print(anis)
	

	for i, file_name in enumerate(train_file_names):
		data_set = ut.load_npy(data_dir + file_name + '_data')
		print(" Data set {} imported, sample size: {}".format(file_name, data_set.shape[0]))
		train.append(data_set)
		if anis:	
			anis_data = ut.load_npy(data_dir + file_name + '_anis')
			train_ref += [anis_data]
		else: train_ref += [classes[i] * data_set.shape[0]]

	print(" Training data set loaded, contains {} samples\n".format(np.sum([data_set.shape[0] for data_set in train])))

	for file_name in predict_file_names: 
		data_set = ut.load_npy(data_dir + file_name + '_data')
		print(" Data set {} imported, sample size: {}".format(file_name, data_set.shape[0]))
		predict.append(data_set)

	print(" Prediction data set loaded, contains {} samples\n".format(np.sum([data_set.shape[0] for data_set in predict])))

	image_size = np.min([np.min(data_set.shape[1:]) for data_set in train + predict])
	#image_size = 150
	print("\n Sample image size = {} x {} pixels\n".format(image_size, image_size))

	train = get_data_set(train_file_names, fig_dir, train, image_size, n_images, nmf)
	predict = get_data_set(predict_file_names, fig_dir, predict, image_size, n_images, nmf)

	try:
		if len(train) > 1: train_data_set = np.concatenate((train))
		else: train_data_set = np.array(train[0])
		if anis: 
			train_data_labels = np.concatenate((train_ref))
			hist, bin_edges = np.histogram(train_data_labels, bins=5, density=True)
			train_data_labels = np.digitize(train_data_labels, bin_edges[:-1])
			classes = list(np.unique(train_data_labels))
		else: 
			train_data_labels = train_ref
			classes = list(np.unique(classes))
	except: 
		train_data_set = None
		train_data_labels = None

	try: predict_data_set = predict
	except: predict_data_set = None

	"Perform Non-Negative Matrix Factorisation"
	n_clusters = len(classes)
	if cluster: hierarchical_clustering(train_data_set, train_data_labels, n_clusters, classes)

	"Perform convolutional neural network analysis"
	#convolutional_neural_network_analysis(model_name, model_dir, classes=classes, predict_set=predict_data_set, 
	#								data_set=train_data_set, data_labels=train_data_labels, ow_model=ow_mod)
