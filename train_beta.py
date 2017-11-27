#train_beta.py
#try to train Gauss Mixtured Model with EM algorithm
#Nur3e
#initialize 17.11.23
#and this is my first python:)
from pic import pic_mg
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

#log_name = src('train_log_')

#value set
M = 8											#m Gausses
K = 0

#import data
trainDataOrig = np.loadtxt('train.txt')			#4800,3			n,3
trainData = trainDataOrig[0:4800,0:2]	 		#4800,2			n,2
N = trainData.shape[0]

#pic_data()

#initialize
mu = np.random.rand(M, 2)*(5.5, 5)-(1.5,2)				#mu/mean 		m, 2
sigma = np.zeros([M, 2, 2])								#sigma/dev mat	m, 2,2
pi = np.random.rand(M)									#pi				m
piSum = pi.sum()
#sigmaLine = np.random.rand(M, 2)
for m in range(M):
	sigma[m] = np.random.rand(2, 2)
	sigma[m][0][1] = sigma[m].min()
	sigma[m][1][0] = sigma[m][0][1]
	pi[m] = pi[m]/piSum
	print sigma[m]

K = 1
deltaNow = np.ones(M)

f = open('train_log.dat', 'w')

#iteration data set
upsilon = np.zeros([M, N])						#upsilonMN		m, n
upsilonSumN = np.zeros(M)						#upsilonM		n

pic_mg(m, mu, sigma, trainData)

#iteration 
while (deltaNow>=1e-5).any():
##calc upsilon
	print >> f, "in iteration #", K
	print >> f, "mu:\n", mu
	deno = np.zeros(N)							#denominator	n
	nume = np.zeros([M,N])						#numerator		m, n

	for m in range(M):
		#multivaritate_Gauss
		mg = ss.multivariate_normal(mu[m],sigma[m])					
		nume[m] = pi[m]*mg.pdf(trainData)
		deno += nume[m]

	for m in range(M):
		upsilon[m] = nume[m]/deno
		upsilonSumN[m] = upsilon[m].sum()

	#update
	oldmu = np.zeros([M,2])
	oldmu += mu
	upsilonSumMN = upsilonSumN.sum()
	for m in range(M):
		mu[m] = np.zeros(2)
		for n in range(N):
			mu[m] += upsilon[m][n]*trainData[n]
		mu[m] /= upsilonSumN[m]

		sigma[m] = np.zeros([2,2])
		for n in range(N):
			sigma[m] += upsilon[m][n]*((trainData[n]-mu[m])*(trainData[n]-mu[m]).reshape(-1,1))
		sigma[m] /= upsilonSumN[m]

		pi[m] = upsilonSumN[m]/upsilonSumMN

	K+=1
	deltaNow = np.abs(mu - oldmu)
	#print K, ' ',
	if K%10 == 0:
		pic_mg(m, mu, sigma, trainData)
		print K
print >> f, "sigma:\n", sigma
print >> f, "mu:\n", mu
print >> f, "pi:\n", pi

np.savetxt('mu.txt', mu)
np.savetxt('pi.txt', pi)
np.save('sigma.npy', sigma)


SMR = np.zeros(N)
myM = np.zeros(N)	#dot: my Gauss Model
myN = np.zeros(N)	#dot: my Gauss Model Number

#define category

for m in range(M):
	sm = ss.multivariate_normal(mu[m], sigma[m])
	SMR = sm.pdf(trainData)
	for n in range(N):
		if SMR[n] > myM[n]:
			myN[n] = m
			myM[n] = SMR[n]

vote = np.zeros(8)
for n in range(N):
	index = int(myN[n])
	if trainDataOrig[n][2] == 1:
		vote[index] += 1
	else:
		vote[index] -= 1

cate = np.zeros(8)
for m in range(M):
	print >> f, "VOTE:", m, " ", vote[m]
	if vote[m] > 0:
		cate[m] = 1
	else:
		cate[m] = 2

np.savetxt('cate.txt', cate)
f.close()