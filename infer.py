#infer.py
#Nur3e
#17.11.24

import numpy as np
import scipy.stats as ss
from pic import pic_mg

M = 8

testDataOrig = np.loadtxt('dev.txt')
testData = testDataOrig[:,[0,1]]
testRef	= testDataOrig[:,2]
N = testData.shape[0]

mu = np.loadtxt('mu.txt')
sigma = np.load('sigma.npy')
pi = np.loadtxt('pi.txt')
ca = np.loadtxt('cate.txt')

iRst = np.zeros(N)			#inferResult
iRstP = np.zeros(N)			
tRst = np.zeros(N)			#testResult

SMR = np.zeros([M,N])		#SigleModelResult

f = open('dev_log.dat', 'w')

for m in range(M):
	#single model
	sm = ss.multivariate_normal(mu[m], sigma[m])
	SMR[m] = pi[m]*sm.pdf(testData)
	for n in range(N):
		if iRstP[n] < SMR[m][n]:
			#print >> f, "C: in", m, " ", n, " ", SMR[m][n] 
			iRstP[n] = SMR[m][n]
			iRst[n] = m

for n in range(N):
	index = int(iRst[n])
	iRst[n] = ca[index]

cnt = np.zeros(1)
for n in range(N):
	#print >> f, n, ' ', iRst[n]
	if iRst[n] == testRef[n]:
		cnt+=1
	else:
		if (iRst[n] == 1):
			testDataOrig[n,2] = 3	#2->1
		else:
			testDataOrig[n,2] = 4	#1->2

print cnt/N
f.close()
testDataOrig = testDataOrig[testDataOrig[:,2].argsort()]
pic_mg(M, mu, sigma, testDataOrig)