import numpy as np
import pandas as pd
from scipy.stats import norm
import math
from datetime import datetime
from matplotlib import pyplot as plt
class hw1Derivative():
    def __init__(self, S0=42, r=0.1, q=0, sigma=0.2, T=0.5, K1=40, K2=10000000, K3=10000001, K4=10000002):
        self._S0 = S0
        self._r = r
        self._q = q
        self._sigma = sigma
        self._T = T
        self._K1 = K1
        self._K2 = K2
        self._K3 = K3
        self._K4 = K4 

        self.computeSummary = "not computed yet"
        self.expectedPayoff = -999999
        self.price = -9999999

    @classmethod
    def fromLst(cls, paraLst):
        return(
            cls(
                paraLst[0], paraLst[1], paraLst[2], paraLst[3], paraLst[4],
                paraLst[5], paraLst[6], paraLst[7], paraLst[8]
            )
        )
    @classmethod
    def fromDict(cls, paraDict):
        #s0, r, q, sigma, t, k1, k2, k3, k4
        return(
            cls(
                paraDict['s0'], paraDict['r'], paraDict['q'], paraDict['sigma'],
                paraDict['t'], paraDict['k1'], paraDict['k2'], paraDict['k3'], paraDict['k4']
            )
        )

    def summary(self):
        print("==========================================================================")
        print(f"S_0: {self._S0}")
        print(f"r: {self._r}, q: {self._q}, sigma: {self._sigma}, T: {self._T}")
        print(f"K1: {self._K1}, K2: {self._K2}, K3: {self._K3}, K4: {self._K4}")
        print("==========================================================================\n")
    def __str__(self):
        toReturn = ''
        toReturn += f"S_0: {self._S0}\n"
        toReturn += f"r: {self._r}, q: {self._q}, sigma: {self._sigma}, T: {self._T}\n"
        toReturn += f"K1: {self._K1}, K2: {self._K2}, K3: {self._K3}, K4: {self._K4}"
        return toReturn
    def __repr__(self):
        return self.__str__()


class hw1MontiCarlo(hw1Derivative):
    def __init__(self, S0=42, r=0.1, q=0, sigma=0.2, T=0.5, K1=40, K2=10000000, K3=10000001, K4=10000002, trialCnt=20, stCnt_perTrial=10000):
        super().__init__(S0, r, q, sigma, T, K1, K2, K3, K4)

        self._trialCnt = trialCnt
        self._stCnt_perTrial = stCnt_perTrial
        self._shape = (trialCnt, stCnt_perTrial)
        self._normSampleResult = np.zeros(self._shape)
        self._stSampleResult = np.zeros(self._shape)
        self._payoffResult = np.zeros(())

        self._trialSd = 1
        self._trustinterval = (-1, 1)


    def payoff(self, st):
        if(self._K1 <= st and st < self._K2):
            return float(st - self._K1)
        elif(self._K2 <= st < self._K3):
            return float(self._K2 - self._K1)
        elif(self._K3 <= st < self._K4):
            slope = (self._K2 - self._K1) / (self._K4 - self._K3)
            return float((self._K2 - self._K1) - (st - self._K3) * slope)
        else:
            return 0.0

    payoff_v = np.vectorize(payoff)

    def mcSim(self):
        mu = math.log(self._S0) + self._T * (self._r - self._q - math.pow(self._sigma, 2)/2)
        sd = self._sigma * math.pow(self._T, 0.5)
        self._normSampleResult = np.random.normal(loc= mu, scale= sd, size=self._shape)
        self._stSampleResult = np.exp(self._normSampleResult)
        self._payoffResult = self.payoff_v(self, self._stSampleResult)

    def trialstat(self):
        self.mcSim()
        means = np.mean(self._payoffResult, axis=1)
        discountedMean = means * math.exp((-1) * self._r * self._T)
        self.expectedPayoff = np.mean(means)
        self.price = np.mean(discountedMean)
        self._trialSd = np.std(discountedMean)
        self._trustinterval = (self.price - 2 * self._trialSd, self.price + 2 * self._trialSd)


class hw1CloseForm(hw1Derivative):
    def __init__(self, S0=42, r=0.1, q=0, sigma=0.2, T=0.5, K1=40, K2=10000000, K3=10000001, K4=10000002):
        super().__init__(S0, r, q, sigma, T, K1, K2, K3, K4)
        
        self.__KiDict = {1:self._K1, 2:self._K2, 3:self._K3, 4:self._K4}
        self.__diqrDict = dict()
        self.__ndiqrDict = dict()
        self.__d_ndi_ndjDict = dict()
        self.__ixDict = dict()

        
    @classmethod
    def fromLst(cls, paraLst):
        return(
            cls(
                paraLst[0], paraLst[1], paraLst[2], paraLst[3], paraLst[4],
                paraLst[5], paraLst[6], paraLst[7], paraLst[8]
            )
        )

    @classmethod
    def fromDict(cls, paraDict):
        #s0, r, q, sigma, t, k1, k2, k3, k4
        return(
            cls(
                paraDict['s0'], paraDict['r'], paraDict['q'], paraDict['sigma'],
                paraDict['t'], paraDict['k1'], paraDict['k2'], paraDict['k3'], paraDict['k4']
            )
        )
    
    def __diq(self, i):
        Ki = self.__KiDict[i]
        numerator_1 = math.log(Ki/self._S0, math.e)
        numerator_2 = (self._r - self._q - math.pow(self._sigma,2)/2) * self._T
        denominator = self._sigma * math.pow(self._T, 0.5)
        return (numerator_1 - numerator_2)/denominator
    
    def __dir(self, i):
        Ki = self.__KiDict[i]
        numerator_1 = math.log(Ki/self._S0, math.e)
        numerator_2 = (self._r - self._q + math.pow(self._sigma,2)/2) * self._T
        denominator = self._sigma * math.pow(self._T, 0.5)
        return (numerator_1 - numerator_2)/denominator

    def __compute_diqr(self):
        for i in range(1,5):
            self.__diqrDict[f"d{i}q"] = self.__diq(i)
            self.__diqrDict[f"d{i}r"] = self.__dir(i)

    def __compute_ndiqr(self):
        for key in self.__diqrDict:
            self.__ndiqrDict[key] = norm.cdf(self.__diqrDict[key])

    def __compute_d_ndi_ndj(self):
        toCompute=[('q',2,1), ('q',3,2), ('q',4,3), ('r',2,1), ('r',4,3)]
        for tup in toCompute:
            key = f"nd{tup[1]}{tup[0]}-nd{tup[2]}{tup[0]}"
            a = self.__ndiqrDict[f"d{tup[1]}{tup[0]}"]
            b = self.__ndiqrDict[f"d{tup[2]}{tup[0]}"]
            self.__d_ndi_ndjDict[key] = a - b

    def __computeIX(self):
        s0er_qt = self._S0 * math.exp((self._r - self._q) * self._T)
        slope = (self._K2 - self._K1) / (self._K4 - self._K3)

        self.__ixDict["IA_1"] = s0er_qt * self.__d_ndi_ndjDict["nd2r-nd1r"]
        self.__ixDict["IA_2"] = self._K1 * self.__d_ndi_ndjDict["nd2q-nd1q"]
        self.__ixDict["IC_1"] = (self._K2 - self._K1) * self.__d_ndi_ndjDict["nd4q-nd3q"]
        self.__ixDict["IC_2"] = self._K3 * slope * self.__d_ndi_ndjDict["nd4q-nd3q"]
        self.__ixDict["IC_3"] = slope * s0er_qt * self.__d_ndi_ndjDict["nd4r-nd3r"]

        self.__ixDict["A"] = self.__ixDict["IA_1"] - self.__ixDict["IA_2"]
        self.__ixDict["B"] = (self._K2 - self._K1) * self.__d_ndi_ndjDict["nd3q-nd2q"]
        self.__ixDict["C"] = self.__ixDict["IC_1"] + self.__ixDict["IC_2"] - self.__ixDict["IC_3"]

    def __computeSummary(self):
        cpSumStr = ""

        cpSumStr += "==========================================================================\n"
        for key in self.__diqrDict:
            cpSumStr += f"{key} {self.__diqrDict[key]}, {self.__ndiqrDict[key]}\n"
        cpSumStr += "==========================================================================\n\n"

        cpSumStr += "==========================================================================\n"
        for key in self.__d_ndi_ndjDict:
            cpSumStr += f"{key}, {self.__d_ndi_ndjDict[key]}\n"
        cpSumStr += "==========================================================================\n\n"

        cpSumStr += "==========================================================================\n"
        for key in self.__ixDict:
            cpSumStr += f"{key}, {self.__ixDict[key]}\n"
        cpSumStr += "==========================================================================\n\n"

        self.computeSummary = cpSumStr

    def computePrice(self):
        self.__compute_diqr()
        self.__compute_ndiqr()
        self.__compute_d_ndi_ndj()
        self.__computeIX()
        self.expectedPayoff = self.__ixDict["A"] + self.__ixDict["B"] + self.__ixDict["C"]
        self.price = self.expectedPayoff * math.exp((-1) * self._r * self._T)
        self.__computeSummary()


    def compare_mcSim(self, stCnt_perTrial, trialCnt):
        mctest = hw1MontiCarlo(self._S0, self._r, self._q, self._sigma, self._T, self._K1, self._K2, self._K3, self._K4, trialCnt, stCnt_perTrial)
        mctest.trialstat()
        return mctest


if __name__ == '__main__':
    #s0, r, q, sigma, t, k1, k2, k3, k4
    testParaLst = [42, 0.1, 0, 0.2, 0.5, 40, 42, 43, 55]
    testCF = hw1CloseForm.fromLst(testParaLst)
    #testCF.summary()
    testCF.computePrice()
    #print(testCF.computeSummary)
    print(testCF.price)

    
    print('montecarlo simulation')
    testMC = testCF.compare_mcSim(10000, 20)
    #testMC.trialstat()
    print(testMC.price)
else:
    print('imported succesfully')