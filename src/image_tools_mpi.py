def nematic_tensor_analysis_mpi(n_vector, area, min_sample, comm, size, rank, thresh = 0.05):
	"""
	nematic_tensor_analysis(n_vector, area, n_frame, n_sample)

	Calculates eigenvalues and eigenvectors of average nematic tensor over area^2 pixels for n_samples

	Parameters
	----------

	n_vector:  array_like (float); shape(n_frame, n_y, n_x, 4)
		Flattened 2x2 nematic vector for each pixel in dx_shg, dy_shg (n_xx, n_xy, n_yx, n_yy)

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample

	Returns
	-------

	av_eigval:  array_like (float); shape=(n_frame, n_sample, 2)
		Eigenvalues of average nematic tensors for n_sample areas

	av_eigvec:  array_like (float); shape=(n_frame, n_sample, 2, 2)
		Eigenvectors of average nematic tensors for n_sample areas

	"""

	n_frame = n_vector.shape[0]
	n_y = n_vector.shape[2]
	n_x = n_vector.shape[3]

	tot_q = np.zeros(n_frame)
	av_q = []

	pad = int(area / 2 - 1)

	analysing = True
	sample = size

	while analysing:

		av_eigval = np.zeros((n_frame, 2))
		av_eigvec = np.zeros((n_frame, 2, 2))

		try: start_x = np.random.randint(pad, n_x - pad)
		except: start_x = pad
		try: start_y = np.random.randint(pad, n_y - pad) 
		except: start_y = pad

		cut_n_vector = n_vector[:, :, start_y-pad: start_y+pad, 
					      start_x-pad: start_x+pad]

		av_n = np.reshape(np.mean(cut_n_vector, axis=(2, 3)), (n_frame, 2, 2))

		for frame in range(n_frame):

			eig_val, eig_vec = np.linalg.eigh(av_n[frame])

			av_eigval[frame] = eig_val
			av_eigvec[frame] = eig_vec

		tot_q += (av_eigval.T[1] - av_eigval.T[0])

		gather_q = comm.gather(np.mean(tot_q) / sample)

		if rank == 0:
			av_q += gather_q
			if sample >= min_sample:
				q_mov_av = ut.cum_mov_average(av_q)
				analysing = (q_mov_av[-1] - q_mov_av[-2]) > thresh

		analysing = comm.bcast(analysing, root=0)

		sample += size

	tot_q = comm.allreduce(tot_q, op=MPI.SUM)

	return tot_q / sample, sample


def fourier_transform_analysis_mpi(image_shg, comm, size, rank):
	"""
	fourier_transform_analysis(image_shg, area, n_sample)

	Calculates fourier amplitude spectrum of over area^2 pixels for n_samples

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

	angles:  array_like (float); shape=(n_bins)
		Angles corresponding to fourier amplitudes

	fourier_spec:  array_like (float); shape=(n_bins)
		Average Fouier amplitudes of FT of image_shg

	"""

	n_sample = image_shg.shape[0]

	image_fft = np.fft.fft2(image_shg[0])
	image_fft[0][0] = 0
	image_fft = np.fft.fftshift(image_fft)
	average_fft = np.zeros(image_fft.shape, dtype=complex)

	fft_angle = np.angle(image_fft, deg=True)
	angles = np.unique(fft_angle)
	fourier_spec = np.zeros(angles.shape)
	
	n_bins = fourier_spec.size

	for n in range(rank, n_sample, size):
		image_fft = np.fft.fft2(image_shg[n])
		image_fft[0][0] = 0
		average_fft += np.fft.fftshift(image_fft) / n_sample

	average_fft = comm.allreduce(average_fft, op=MPI.SUM)

	for i in range(n_bins):
		indices = np.where(fft_angle == angles[i])
		fourier_spec[i] += np.sum(np.abs(average_fft[indices])) / 360

	return angles, fourier_spec