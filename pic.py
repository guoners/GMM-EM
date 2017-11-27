#pic.py
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import time

M = 8

# mu = np.loadtxt('mu.txt')
# sigma = np.load('sigma.npy')
# pi = np.loadtxt('pi.txt')
# ca = np.loadtxt('cate.txt')

#td = np.loadtxt('train.txt') #train-data
def pic_data(td):
	(x1, y1) = td[0:2400,0:2].T
	(x2, y2) = td[2400:4801,0:2].T
	plt.scatter(x1, y1, s=1.5)
	plt.scatter(x2, y2, s=1.5)

	#plt.show()
	return

# delta = 0.025
# x = np.arange(-3.0, 3.0, delta)
# y = np.arange(-2.0, 2.0, delta)
# X, Y = np.meshgrid(x, y)
# mg = ss.multivariate_normal(mu[1], sigma[1])

# Z = mg.pdf(X,Y)
# plt.contour(X,Y,Z)
def pic_mg(m, mu, sigma, td):
	pic_data(td)
	for m in range(M):
		x, y = np.mgrid[-1.5:4:.01, -2:3:.01]
		pos = np.empty(x.shape + (2,))
		pos[:, :, 0] = x; pos[:, :, 1] = y
		rv = ss.multivariate_normal(mu[m], sigma[m])
		z = rv.pdf(pos)
		dx = np.sqrt(sigma[m][0][0])
		dy = np.sqrt(sigma[m][1][1])
		d = rv.pdf([mu[m][0]-dx,mu[m][1]-dy])
		plt.contour(x, y, z, [d], linewidths=5)
	plt.show()
	return