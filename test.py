#test.py
#Nur3e
#17.11.24

import numpy as np
import scipy.stats as ss

f = open('ans.txt', 'w')
M = 8

testData = np.loadtxt('test.txt')
N = testData.shape[0]

mu = np.loadtxt('mu.txt')
sigma = np.load('sigma.npy')
pi = np.loadtxt('pi.txt')
ca = np.loadtxt('cate.txt')

iRst = np.zeros(N)			#inferResult
iRstP = np.zeros(N)			
tRst = np.zeros(N)			#testResult

SMR = np.zeros([M,N])			#SigleModelResult

for m in range(M):
	#single model
	sm = ss.multivariate_normal(mu[m], sigma[m])
	SMR[m] = pi[m]*sm.pdf(testData)
	for n in range(N):
		if iRstP[n] < SMR[m][n]:
			iRstP[n] = SMR[m][n]
			iRst[n] = m

for n in range(N):
	index = int(iRst[n])
	iRst[n] = ca[index]

for n in range(N):
	print >> f, "%1.6f %1.6f %d" %(testData[n][0], testData[n][1], iRst[n])