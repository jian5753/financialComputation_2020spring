import math
from scipy.stats import norm
import numpy as np

class vanillaOption():
    def __init__(self, S0, r, q, sigma, T, K):
        self.S0 = S0
        self.r = r
        self.q = q
        self.sigma = sigma
        self.sigmaSq = math.pow(sigma, 2)
        self.T = T
        self.sqrtT = math.pow(T, 0.5)
        self.K = K

class vanillaEuro(vanillaOption):
    def __init__(self, S0, r, q, sigma, T, K):
        super().__init__(S0, r, q, sigma, T, K)

        self.d1 = (math.log(self.S0 / self.K) + (self.r - self.q + self.sigmaSq / 2) * self.T) / (self.sigma * self.sqrtT)
        self.d2 = self.d1 - self.sigma * self.sqrtT

class vanillaEuro_bs(vanillaEuro):
    def __init__(self, S0, r, q, sigma, T, K):
        super().__init__(S0, r, q, sigma, T, K)

        self.nd1 = norm.cdf(self.d1)
        self.nd2 = norm.cdf(self.d2)
        self.n_d1 = norm.cdf(-self.d1)
        self.n_d2 = norm.cdf(-self.d2)

        self.callPrice_1 = self.S0 * math.exp(-self.q * self.T) * self.nd1
        self.callPrice_2 = self.K * math.exp(-self.r * self.T) * self.nd2
        self.callPrice = self.callPrice_1 - self.callPrice_2

        self.putPrice_1 = self.K * math.exp(-self.r * self.T) * self.n_d2
        self.putPrice_2 = self.S0 * math.exp(-self.q * self.T) * self.n_d1
        self.putPrice = self.putPrice_1 - self.putPrice_2

class vanillaEuro_biTree(vanillaOption):
    def __init__(self, S0, r, q, sigma, T, K, n):
        super().__init__(S0, r, q, sigma, T, K)
        
        self.n = n
        self.delta_t = T / n
        self.sqrt_delta_t = math.pow(self.delta_t, 0.5)

        self.u = math.exp(self.sigma * self.sqrt_delta_t)
        self.d = 1 / self.u
        self.p = (math.exp((self.r -self.q)* self.delta_t) - self.d) / (self.u - self.d)

        self.payoffTree = np.zeros((n + 1, n + 1))
        self.price = self.payoffTree[0][0]

    def STij(self, generation, dCnt):
        uCnt = generation - dCnt
        return self.S0 * math.pow(self.u, uCnt) * math.pow(self.d, dCnt)
    
    def nodePayoff(self, St):
        return St

    def fillInPayoff(self):
        for i in range(self.n + 1):
            St = self.STij(self.n, i)
            self.payoffTree[i][self.n] = self.nodePayoff(St)

        for genCnt in range(self.n - 1, -1, -1):
            for nodeCnt in range(genCnt + 1):
                upNode = self.payoffTree[nodeCnt][genCnt + 1]
                downNode = self.payoffTree[nodeCnt + 1][genCnt + 1]
                exPayoff = self.p * upNode + (1 - self.p) * downNode
                self.payoffTree[nodeCnt][genCnt] = exPayoff * math.exp(-self.r * self.delta_t)

class vanillaEuroCall_mcSim(vanillaOption):
    def __init__(self, S0, r, q, sigma, T, K, priceCnt, trialCnt):
        super().__init__(S0, r, q, sigma, T, K)

        self.trialCnt = trialCnt
        self.priceCnt = priceCnt
        self.shape = (trialCnt, priceCnt)
        self.normSampleResult = np.zeros(self.shape)
        self.stSampleResult = np.zeros(self.shape)
        self.payoffResult = np.zeros(())

        self.trialSd = 1
        self.trustinterval = (-1, 1)
    
    def payoff(self, st):
        return max(st - self.K, 0) + 0.0

    payoff_v = np.vectorize(payoff)

    def mcSim(self):
        mu = math.log(self.S0) + self.T * (self.r - self.q - math.pow(self.sigma, 2)/2)
        sd = self.sigma * math.pow(self.T, 0.5)
        self.normSampleResult = np.random.normal(loc= mu, scale= sd, size=self.shape)
        self.stSampleResult = np.exp(self.normSampleResult)
        self.payoffResult = self.payoff_v(self, self.stSampleResult)

    def trialstat(self):
        self.mcSim()
        means = np.mean(self.payoffResult, axis=1)
        discountedMean = means * math.exp((-1) * self.r * self.T)
        self.expectedPayoff = np.mean(means)
        self.price = np.mean(discountedMean)
        self.trialSd = np.std(discountedMean)
        self.trustinterval = (self.price - 2 * self.trialSd, self.price + 2 * self.trialSd)

class vanillaEuroPut_mcSim(vanillaEuroCall_mcSim):
    def __init__(self, S0, r, q, sigma, T, K, priceCnt, trialCnt):
        super().__init__(S0, r, q, sigma, T, K, priceCnt, trialCnt)
    
    def payoff(self, st):
        return max(self.K - st, 0) + 0.0

    payoff_v = np.vectorize(payoff)

class vanillaEuro_svBiTree(vanillaEuro_biTree):
    def __init__(self, S0, r, q, sigma, T, K, n):
        super().__init__(S0, r, q, sigma, T, K, n)

        self.payoffTree = np.zeros(n + 1)

    def fillInPayoff(self):
        for i in range(self.n + 1):
            St = self.STij(self.n, i)
            self.payoffTree[i] = self.nodePayoff(St)

        for genCnt in range(self.n - 1, -1, -1):
            for nodeCnt in range(genCnt + 1):
                upNode = self.payoffTree[nodeCnt]
                downNode = self.payoffTree[nodeCnt + 1]
                exPayoff = self.p * upNode + (1 - self.p) * downNode
                self.payoffTree[nodeCnt] = exPayoff * math.exp(-self.r * self.delta_t)

class vanillaEuroCall_biTree(vanillaEuro_biTree):
    def __init__(self, S0, r, q, sigma, T, K, n):
        super().__init__(S0, r, q, sigma, T, K, n)

    def nodePayoff(self, St):
        return max(St - self.K, 0.0) + 0.0

    
class vanillaEuroPut_biTree(vanillaEuro_biTree):
    def __init__(self, S0, r, q, sigma, T, K, n):
        super().__init__(S0, r, q, sigma, T, K, n)

    def nodePayoff(self, St):
        return max(self.K - St, 0.0) +0.0

class vanillaEuroCall_svBiTree(vanillaEuro_svBiTree):
    def __init__(self, S0, r, q, sigma, T, K, n):
        super().__init__(S0, r, q, sigma, T, K, n)

        self.payoffTree = np.zeros(n + 1)

    def nodePayoff(self, St):
        return max(St-self.K, 0.0) + 0.0

class vanillaEuroPut_svBiTree(vanillaEuro_svBiTree):
    def __init__(self, S0, r, q, sigma, T, K, n):
        super().__init__(S0, r, q, sigma, T, K, n)

        self.payoffTree = np.zeros(n + 1)

    def nodePayoff(self, St):
        return max(self.K - St, 0.0) +0.0

class vanillaAmrc(vanillaEuro_biTree):
    def __init__(self, S0, r, q, sigma, T, K, n):
        super().__init__(S0, r, q, sigma, T, K, n)

        self.earlyExCheckBox = np.zeros((n + 1, n + 1))

    def fillInPayoff(self):
        for i in range(self.n + 1):
            St = self.STij(self.n, i)
            self.payoffTree[i][self.n] = self.nodePayoff(St)

        for genCnt in range(self.n - 1, -1, -1):
            for nodeCnt in range(genCnt + 1):
                upNode = self.payoffTree[nodeCnt][genCnt + 1]
                downNode = self.payoffTree[nodeCnt + 1][genCnt + 1]
                exPayoff = self.p * upNode + (1 - self.p) * downNode
                exPayoff = exPayoff * math.exp(-self.r * self.delta_t)

                currentS = self.STij(genCnt, nodeCnt)
                earlyExPayoff = self.nodePayoff(currentS)

                #print(earlyExPayoff, exPayoff)
                if earlyExPayoff >= exPayoff:
                    self.earlyExCheckBox[nodeCnt][genCnt] = 1
                    exPayoff = earlyExPayoff

                self.payoffTree[nodeCnt][genCnt] = exPayoff

class vanillaAmrc_call(vanillaAmrc):
    def __init__(self, S0, r, q, sigma, T, K, n):
        super().__init__(S0, r, q, sigma, T, K, n)

    def nodePayoff(self, St):
        #print('hello')
        return max(St - self.K, 0.0) +0.0

class vanillaAmrc_put(vanillaAmrc):
    def __init__(self, S0, r, q, sigma, T, K, n):
        super().__init__(S0, r, q, sigma, T, K, n)

    def nodePayoff(self, St):
        return max(self.K - St, 0.0) + 0.0

class vanillaAmrcCall_svBiTree(vanillaAmrc_call):
    def __init__(self, S0, r, q, sigma, T, K, n):
        super().__init__(S0, r, q, sigma, T, K, n)

        self.payoffTree = np.zeros(n + 1)

    def fillInPayoff(self):
        for i in range(self.n + 1):
            St = self.STij(self.n, i)
            self.payoffTree[i] = self.nodePayoff(St)

        for genCnt in range(self.n - 1, -1, -1):
            for nodeCnt in range(genCnt + 1):
                upNode = self.payoffTree[nodeCnt]
                downNode = self.payoffTree[nodeCnt + 1]
                exPayoff = self.p * upNode + (1 - self.p) * downNode
                exPayoff = exPayoff * math.exp(-self.r * self.delta_t)

                currentS = self.STij(genCnt, nodeCnt)
                earlyExPayoff = self.nodePayoff(currentS)

                #print(earlyExPayoff, exPayoff)
                if earlyExPayoff >= exPayoff :
                    self.earlyExCheckBox[nodeCnt][genCnt] = 1
                    exPayoff = earlyExPayoff

                self.payoffTree[nodeCnt] = exPayoff
                
class vanillaAmrcPut_svBiTree(vanillaAmrcCall_svBiTree):
    def __init__(self, S0, r, q, sigma, T, K, n):
        super().__init__(S0, r, q, sigma, T, K, n)

    def nodePayoff(self, St):
        return max(self.K - St, 0) + 0.0

def cexpf(n, m, p, payoff):
    step = min(m, n-m)
    total = 0

    for i in range(step):
        currentN = n - i
        currentM = step - i
        total += (math.log(currentN) - math.log(currentM))
    
    for i in range(m):
        total += math.log(p)
    
    for i in range(n-m):
        total += math.log(1-p)
    

    toReturn = math.exp(total) * payoff
    #print(math.exp(total), payoff, toReturn)
    return toReturn


class combiEuro_call(vanillaEuroCall_svBiTree):
    def __init__(self, S0, r, q, sigma, T, K, n):
        super().__init__(S0, r, q, sigma, T, K, n)

        self.price = 0
    
    def compute(self):
        for i in range(self.n):
            currentSt = self.STij(self.n, i)
            expf = cexpf(self.n, self.n - i, self.p, self.nodePayoff(currentSt))
            self.price += expf
        self.price = self.price * math.exp(-self.r * self.T)

class combiEuro_put(vanillaEuroPut_svBiTree):
    def __init__(self, S0, r, q, sigma, T, K, n):
        super().__init__(S0, r, q, sigma, T, K, n)

        self.price = 0

    def compute(self):
        for i in range(self.n + 1):
            #print(i, end=' ')
            currentSt = self.STij(self.n, i)
            expf = cexpf(self.n, self.n - i, self.p, self.nodePayoff(currentSt))
            self.price += expf
        self.price = self.price * math.exp(-self.r * self.T)