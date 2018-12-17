def test_analysis(current_dir, size=None, sigma=None, ow_anis=False):
	"""
	plot_labeled_figure(fig_dir, fig_name, image, label_image, labels)

	Plots a figure showing identified areas of anisotropic analysis

	Parameter
	---------

	fig_name:  string
		Name of figures to be saved

	fig_dir:  string
		Directory of figures to be saved

	image:  array_like (float); shape=(n_x, n_y)
		Image under analysis of pos_x and pos_y

	label_image:  array_like (int); shape=(n_x, n_y)
		Labelled array with identified anisotropic regions 

	labels:  array_like (int)
		List of labels to plot on figure

	"""

	N = 200

	fig_dir = current_dir + '/'
	fig_names = [ 'test_image_line', 
				'test_image_cross', 'test_image_noise', 
				'test_image_checker', 'test_image_rings',
				'test_image_fibres_flex', 'test_image_fibres_stiff']

	ske_clus = np.zeros(len(fig_names))
	ske_path = np.zeros(len(fig_names))
	ske_solid = np.zeros(len(fig_names))
	ske_lin = np.zeros(len(fig_names))
	ske_curve = np.zeros(len(fig_names))
	ske_cover = np.zeros(len(fig_names))
	mean_img_anis = np.zeros(len(fig_names))
	mean_pix_anis = np.zeros(len(fig_names))
	mean_ske_anis = np.zeros(len(fig_names))

	for n, fig_name in enumerate(fig_names):

		if fig_name == 'test_image_rings':
			image_grid = np.mgrid[:N, :N]
			for i in range(2): image_grid[i] -= N * np.array(2 * image_grid[i] / N, dtype=int)
			image_grid = np.fft.fftshift(np.sqrt(np.sum(image_grid**2, axis=0)))
			test_image = np.sin(10 * np.pi * image_grid / N ) * np.cos(10 * np.pi * image_grid / N)
		elif fig_name == 'test_image_circle':
			image_grid = np.mgrid[:N, :N]
			for i in range(2): image_grid[i] -= N * np.array(2 * image_grid[i] / N, dtype=int)
			image_grid = np.fft.fftshift(np.sqrt(np.sum(image_grid**2, axis=0)))
			test_image = 1 - ut.gaussian(image_grid, N / 4, 5)
		elif fig_name == 'test_image_line':
			test_image = np.zeros((N, N))
			for i in range(10): test_image += np.eye(N, N, k=5-i)
		elif fig_name == 'test_image_cross':
			test_image = np.zeros((N, N))
			for i in range(10): test_image += np.eye(N, N, k=5-i)
			for i in range(10): test_image += np.rot90(np.eye(N, N, k=5-i))
			test_image = np.where(test_image != 0, 1, 0)
			FIRE(test_image)
		elif fig_name == 'test_image_noise':
			test_image = np.random.random((N, N))
		elif fig_name == 'test_image_checker':
			test_image = data.checkerboard()
			test_image = swirl(test_image, rotation=0, strength=10, radius=120)
		elif fig_name == 'test_image_fibres_stiff':
			test_image = np.load('col_2D_stiff_data.npy')[0]
		elif fig_name == 'test_image_fibres_flex':
			test_image = np.load('col_2D_flex_data.npy')[0]

		res = analyse_image(current_dir, fig_name, test_image, sigma=sigma, ow_anis=ow_anis, mode='test')
		(ske_clus[n], ske_lin[n], ske_cover[n], ske_curve[n], ske_solid[n],
			mean_ske_anis[n], mean_pix_anis[n], mean_img_anis[n]) = res

		print(' Skeleton Clustering = {:>6.4f}'.format(ske_clus[n]))
		print(' Skeleton Linearity = {:>6.4f}'.format(ske_lin[n]))
		print(' Skeleton Coverage = {:>6.4f}'.format(ske_cover[n]))
		print(' Skeleton Solidity = {:>6.4f}'.format(ske_solid[n]))
		print(' Skeleton Curvature = {:>6.4f}'.format(ske_curve[n]))
		print(' Skeleton Anistoropy = {:>6.4f}'.format(mean_ske_anis[n]))
		print(' Total Pixel anistoropy = {:>6.4f}'.format(mean_pix_anis[n]))
		print(' Total Image anistoropy = {:>6.4f}\n'.format(mean_img_anis[n]))

	x_labels = fig_names
	col_len = len(max(x_labels, key=len))

	predictor = predictor_metric(ske_clus, ske_lin, ske_cover, ske_solid,
								mean_ske_anis, mean_pix_anis, mean_img_anis)

	for i, file_name in enumerate(x_labels): 
		if np.isnan(predictor[i]):
			predictor = np.array([x for j, x in enumerate(predictor) if j != i])
			x_labels.remove(file_name)

	ut.bubble_sort(x_labels, predictor)
	x_labels = x_labels[::-1]
	predictor = predictor[::-1]
	#sorted_predictor = np.argsort(predictor)

	print("Order of total predictor:")
	print(' {:{col_len}s} | {:10s} | {:10s}'.format('', 'Predictor', 'Order', col_len=col_len))
	print("_" * 75)

	for i, name in enumerate(x_labels):
		print(' {:{col_len}s} | {:10.3f} | {:10d}'.format(name, predictor[i], i, col_len=col_len))

	print('\n')

	