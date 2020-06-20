from pprint import pprint
import numpy as np
from scipy.linalg import  inv
import math


def cholesky(matrx, lower):
    #print('my version')
    n = matrx.shape[0]
    UTri = np.zeros((n,n))
    UTri[0][0] = np.sqrt(matrx[0][0])
    for i in range(1, n):
        UTri[0][i] = matrx[0][i] / UTri[0][0]

    for i in range(1, n):
        for j in range(i, n):
            UTri[i][j] = matrx[i][j]
            if i == j:
                for k in range(0, j):
                    UTri[i][j] -= (UTri[k][j] * UTri[k][j])
                UTri[i][j] = np.sqrt(UTri[i][j])
            else:
                for k in range(0, i):
                    UTri[i][j] -= (UTri[k][i] * UTri[k][j])
                UTri[i][j] /= UTri[i][i]
    if lower:
        return UTri.T
    else:
        return UTri

class montiCarloSimulator():
    def __init__(self, assetCnt, covMatrix, simCnt, repeatCnt):
        self.assetCnt = assetCnt
        self.covMatrix = covMatrix
        self.UTri = cholesky(self.covMatrix, lower=False)
        self.simCnt = simCnt
        self.repeatCnt = repeatCnt
        self.varReduResult = None

        self.result_origin = np.random.normal(
            loc=0.0,
            scale=1.0,
            size = (self.repeatCnt, self.simCnt, self.assetCnt)
        )

        self.varReduResult = np.array(self.result_origin)
        self.result = np.array(self.result_origin)

    def dotCovMatrix(self, Zi):
        for i, trial in enumerate(Zi):
                self.result[i] = trial.dot(self.UTri)

    
    def sample(self, varReduMode = ''):
        # 沒有 bonus 的話 varReduResult 就是原始的Z_i
        self.varReduResult = np.array(self.result_origin)
        if varReduMode == 'bonus1' or varReduMode == 'bonus2':
            # bonus1 的話 先取原始Z_i 的前半部
            self.varReduResult = self.result_origin[:, :int(self.simCnt/2), :]
            # 鏡射一下
            self.mirror = self.varReduResult * (-1)
            # 把鏡射的部分加到後半部
            self.varReduResult = np.concatenate((self.varReduResult, self.mirror), axis = 1)
            # 除以標準差 (反正 mean=0 就不扣了)
            self.varReduResult = self.varReduResult / self.varReduResult.std(axis =  1)[:, np.newaxis]

            if varReduMode == 'bonus2':
                # 如果有 bonus2 的話再多做J個調整
                for i, trial in enumerate(self.varReduResult):
                    c_lambda = np.corrcoef(trial, rowvar = False)
                    c_lambda_utri_inv = inv(cholesky(c_lambda, lower=False))
                    self.varReduResult[i] = self.varReduResult[i].dot(c_lambda_utri_inv)
                    


        self.dotCovMatrix(self.varReduResult)

class rainbow_mcSim():
    def __init__(self, K, r, T, assetCnt, S0Lst, qLst, sigmaLst, corrMatrix, simCnt, repeatCnt):
        self.K = K
        self.r = r
        self.T = T
        self.assetCnt = assetCnt
        self.S0Lst = np.array(S0Lst)
        self.qLst = np.array(qLst)
        self.sigmaLst = np.array(sigmaLst)
        self.corrMatrix = np.array(corrMatrix)

        self.simCnt = simCnt
        self.repeatCnt = repeatCnt

        self.simulator = montiCarloSimulator(self.assetCnt, self.corrMatrix, self.simCnt, self.repeatCnt)
        self.muArr = np.log(self.S0Lst) + self.T * (self.r - self.qLst - np.power(self.sigmaLst, 2)/2)
        self.sdArr = self.sigmaLst * math.pow(self.T, 0.5)


    def compute_covMatrix(self):
        sigmaArr = np.array(self.sigmaLst)
        sigmaDiag = np.diag(sigmaArr)
        sigmaTriu = np.triu(np.array([self.sigmaLst] * self.assetCnt))
        crossSigma = sigmaDiag.dot(sigmaTriu)
        crossSigma = crossSigma + np.transpose(crossSigma)
        crossSigma = crossSigma - sigmaDiag.dot(sigmaDiag)

        self.covMatrix = self.corrMatrix *  crossSigma

    def sim(self, varReduMode):
        self.simulator.sample(varReduMode = varReduMode)
        #print(self.simulator.varReduResult.mean(axis = 1))
        #print(self.simulator.varReduResult.std(axis = 1))
        print(np.corrcoef(self.simulator.varReduResult[-1], rowvar=False))
        self.LnSTArray = self.simulator.result * self.sdArr + self.muArr
        self.STArray = np.exp(self.LnSTArray)
        self.maxSTArray = self.STArray.max(axis = 2)
        self.maxSTArray_minus_K = self.maxSTArray - self.K 
        self.payOffArray =  self.maxSTArray_minus_K * (self.maxSTArray_minus_K > 0)
        self.disctedPayoffArr = self.payOffArray * np.exp(-self.r * self.T)

        
    def answer(self):
        self.meanPrice = self.disctedPayoffArr.mean(axis = 1).mean()
        self.priceStd = self.disctedPayoffArr.mean(axis = 1).std()

        print(f"mean: {self.meanPrice} std: {self.priceStd}")
        print(self.meanPrice - 2*self.priceStd, self.meanPrice, self.meanPrice + 2*self.priceStd)
        
