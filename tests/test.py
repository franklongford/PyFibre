import numpy as np

from skimage import data

from main import analyse_image


N = 200

fig_names = ['test_image_line', 
			'test_image_cross', 'test_image_noise', 
			'test_image_checker', 'test_image_rings']

test_images = []

for n, fig_name in enumerate(fig_names):

	if fig_name == 'test_image_rings':
		image_grid = np.mgrid[:N, :N]
		for i in range(2): image_grid[i] -= N * np.array(2 * image_grid[i] / N, dtype=int)
		image_grid = np.fft.fftshift(np.sqrt(np.sum(image_grid**2, axis=0)))
		test_images.append(np.sin(10 * np.pi * image_grid / N ) * np.cos(10 * np.pi * image_grid / N))

	elif fig_name == 'test_image_circle':
		image_grid = np.mgrid[:N, :N]
		for i in range(2): image_grid[i] -= N * np.array(2 * image_grid[i] / N, dtype=int)
		image_grid = np.fft.fftshift(np.sqrt(np.sum(image_grid**2, axis=0)))
		test_images.append(1 - ut.gaussian(image_grid, N / 4, 5))

	elif fig_name == 'test_image_line':
		test_image = np.zeros((N, N))
		for i in range(10): test_image += np.eye(N, N, k=5-i)
		test_images.append(test_image)

	elif fig_name == 'test_image_cross':
		test_image = np.zeros((N, N))
		for i in range(10): test_image += np.eye(N, N, k=5-i)
		for i in range(10): test_image += np.rot90(np.eye(N, N, k=5-i))
		test_images.append(np.where(test_image != 0, 1, 0))

	elif fig_name == 'test_image_noise':
		test_images.apend(np.random.random((N, N)))

	elif fig_name == 'test_image_checker':
		test_image = data.checkerboard()
		test_images.apend(swirl(test_image, rotation=0, strength=10, radius=120))


def test_FIRE():



	