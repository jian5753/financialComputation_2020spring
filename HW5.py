import numpy as np
ERROR_TOLERENCE = 1e-7
def sequentialSearch(target, pool):
    for idx, number in enumerate(pool):
        if number <= target + ERROR_TOLERENCE:
            return idx, number

def binarySearch(target, pool):
    headIdx = 0
    tailIdx = len(pool) - 1
    if pool[tailIdx] + ERROR_TOLERENCE >= target and pool[tailIdx] - ERROR_TOLERENCE <= target:
        return tailIdx, pool[tailIdx]
    while True:
        middleIdx = int((headIdx + tailIdx) / 2)
        middleValue = pool[middleIdx]
        #print(headIdx, middleIdx, tailIdx, ":", middleValue)
        if middleValue == target:
            return middleIdx, middleValue
        elif tailIdx - headIdx <= 1:
            return tailIdx, pool[tailIdx]
        elif target > middleValue:
            tailIdx = middleIdx
        elif target < middleValue:
            headIdx = middleIdx

def LIsearch(target, pool):
    headIdx = 0
    tailIdx = len(pool) - 1
    headValue = pool[headIdx]
    tailValue = pool[tailIdx]
    if target <= pool[tailIdx]:
        return tailIdx, pool[tailIdx]
    elif target >= pool[headIdx]:
        return headIdx, pool[headIdx]
    else:
        while True:
            preInt = headIdx + (tailIdx - headIdx) * (headValue - target) / (headValue - tailValue)
            tempIdx = int(preInt)
            tempValue = pool[tempIdx]
            #print(headIdx, preInt, tempIdx, tailIdx, "||", headValue, target, tempValue, tailValue)
            # 判斷 tempIdx 再目標前面還是後面 
            if tempValue == target:
                # 終止條件: 找到了
                return tempIdx, target
            elif tempValue >= target:
                headIdx = tempIdx + 1
                #print(headIdx, tempIdx, tailIdx)
            elif tempValue <= target:
                tailIdx = tempIdx - 1
                #print(headIdx, tempIdx, tailIdx)

            headValue = pool[headIdx]
            tailValue = pool[tailIdx]
            # 終止條件: 穿越
            if headValue < target:
                return headIdx, headValue
            if tailValue > target:
                return tempIdx, tempValue
            
class asianNode():
    def __init__(self, St, A_max, A_min, M, bonus1 = False):
        self.St = St
        self.A_max = A_max
        self.A_min = A_min
        self.M = M
        
        #print(A_max, A_min, M, step)
        if A_max == A_min:
            self.AvgLst = np.ones(1) * self.A_max
        else:
            if bonus1:
                logAmax = np.log(A_max)
                logAmin = np.log(A_min)
                #print(logAmax - logAmin, end = '\r')
                temp = np.arange(logAmax, logAmin, (logAmin - logAmax)/M)
                self.AvgLst = np.exp(temp)
                self.AvgLst = np.append(self.AvgLst, self.A_min)
            else:
                self.step = (A_max - A_min) / M
                self.AvgLst = np.arange(A_max, A_min, -self.step)
                self.AvgLst = np.append(self.AvgLst, self.A_min)

        self.uAvgLst = np.array(self.AvgLst)
        self.dAvgLst = np.array(self.AvgLst)
        self.uValueLst = np.zeros_like(self.AvgLst)
        self.dValueLst = np.zeros_like(self.AvgLst) 
        self.ValueLst = np.zeros_like(self.AvgLst)
    
    def getCallValue(self, midAvg, searchMethod):
        if searchMethod == 'binary':
            lowerIdx, lowerAvg = binarySearch(midAvg, self.AvgLst)
        elif searchMethod == 'sequential':
            try:
                lowerIdx, lowerAvg = sequentialSearch(midAvg, self.AvgLst)
            except TypeError:
                print(midAvg, self.AvgLst[-1], midAvg - self.AvgLst[-1])
                raise TypeError
        elif searchMethod == 'LI':
            lowerIdx, lowerAvg = LIsearch(midAvg, self.AvgLst)

        if lowerIdx == 0:
            # midAvg == A_max
            midValue = self.ValueLst[0]
        elif midAvg <= self.AvgLst[-1]:
            # midAvg == A_min 但因為計算誤差所以有可能 midAvg < A_min   
            midValue = self.ValueLst[-1]
        else:
            higherAvg = self.AvgLst[lowerIdx - 1]
            higherValue = self.ValueLst[lowerIdx - 1]
            lowerValue = self.ValueLst[lowerIdx]
            midValue = lowerValue + (higherValue - lowerValue) * (midAvg - lowerAvg) / (higherAvg - lowerAvg)
        return midValue
        

class asianOption():
    def __init__(self, St, K, r, q, sigma, t, T, initAvg):
        self.St = St
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.t = t
        self.T = T
        self.remain_t = self.T - self.t
        self.initAvg = initAvg        

class asianOptionBiTree(asianOption):
    def __init__(self, St, K, r, q, sigma, t, T, initAvg, n, M):
        super().__init__(St, K, r, q, sigma, t, T, initAvg)

        self.n = n
        self.M = M
        self.delta_remain_t = self.remain_t / n
        self.sqrt_delta_remain_t = np.power(self.delta_remain_t, 0.5)
        self.deltaDiscountFac = np.exp(-r * self.delta_remain_t)
        
        self.u = np.exp(self.sigma * self.sqrt_delta_remain_t)
        self.d = 1 / self.u
        self.p = (np.exp((self.r -self.q)* self.delta_remain_t) - self.d) / (self.u - self.d)

        self.w1 = self.n * self.t / self.remain_t + 1
        self.w2 = self.n

        self.tree = np.empty((n+1,n+1), dtype = object)

    def STij(self, generation, dCnt):
        uCnt = generation - dCnt
        if uCnt >= dCnt:
            uCnt = uCnt - dCnt
            dCnt = 0
        else:
            dCnt = dCnt - uCnt
            uCnt = 0
        return self.St * np.power(self.u, uCnt) * np.power(self.d, dCnt)

    def compAmax(self, genCnt, dCnt):
        # without considering stock prices before time t
        uPowi_j = np.power(self.u, genCnt - dCnt)
        dPowj = np.power(self.d, dCnt)
        part2_1 = self.St * self.u * (1 - uPowi_j) / (1 - self.u)
        part2_2 = self.St * uPowi_j * self.d * (1 - dPowj) / (1 - self.d)

        part2 = (part2_1 + part2_2) / genCnt # S0 excluded        
        ans = (self.initAvg * self.w1 + part2 * genCnt) / (self.w1 + genCnt)

        return ans

    def compAmin(self, genCnt, dCnt):
        uPowi_j = np.power(self.u, genCnt - dCnt)
        dPowj = np.power(self.d, dCnt)
        part2_1 = self.St * self.d * (1 - dPowj) / (1 - self.d)
        part2_2 = self.St * dPowj * self.u * (1 - uPowi_j) / (1 - self.u)

        part2 = (part2_1 + part2_2) / genCnt # S0 excluded
        ans = (self.initAvg * self.w1 + part2 * genCnt) / (self.w1 + genCnt)

        return ans

    def grow(self, bonus1):
        self.tree[0][0] = asianNode(self.St, self.St, self.St, self.M)
        for genCnt in range(1, self.n + 1):
            for dCnt in range(0, genCnt + 1):
                Amax = self.compAmax(genCnt, dCnt)
                Amin = self.compAmin(genCnt, dCnt)
                self.tree[genCnt][dCnt] = asianNode(
                    self.STij(genCnt, dCnt),
                    Amax, Amin,
                    self.M,
                    bonus1
                )

        for dCnt in range(0, self.n + 1):
            currentNode = self.tree[self.n][dCnt]
            for idx, avgPrice in enumerate(currentNode.AvgLst):
                currentNode.ValueLst[idx] = max(avgPrice - self.K, 0)

    def comp_uAvg(self, avgPrice, currentGen, currentSt):
        currentW1 = currentGen + self.w1
        return(avgPrice * currentW1 + currentSt * self.u) / (currentW1 + 1)

    def comp_dAvg(self, avgPrice, currentGen, currentSt):
        currentW1 = currentGen + self.w1
        return(avgPrice * currentW1 + currentSt * self.d) / (currentW1 + 1)

    def backward_induction(self, currentGen, currentDcnt, AM = False, searchMethod = 'binary'):
        currentNode = self.tree[currentGen][currentDcnt]
        uChild = self.tree[currentGen + 1][currentDcnt]
        dChild = self.tree[currentGen + 1][currentDcnt + 1]

        #backward induction 
        for idx, avgPrice in enumerate(currentNode.AvgLst):
            uAvg = self.comp_uAvg(avgPrice, currentGen, currentNode.St)
            dAvg = self.comp_dAvg(avgPrice, currentGen, currentNode.St)
            currentNode.uAvgLst[idx] = uAvg
            currentNode.dAvgLst[idx] = dAvg
            print(f'{currentGen}, {currentDcnt}, {idx}', end ='\r')
            uValue = uChild.getCallValue(uAvg, searchMethod)
            dValue = dChild.getCallValue(dAvg, searchMethod)
            currentNode.uValueLst[idx] = uValue
            currentNode.dValueLst[idx] = dValue
            holdingValue = (self.p * uValue + (1 - self.p) * dValue) * self.deltaDiscountFac
            
            if AM:
                earlyExValue = max(avgPrice - self.K, 0)
                if earlyExValue > holdingValue:
                    holdingValue = earlyExValue
            currentNode.ValueLst[idx] = holdingValue

    def backwardInduction(self, AM, searchMethod):
        for genCnt in range(self.n - 1, -1, -1):
            for dCnt in range(0, genCnt + 1):
                #print(genCnt, dCnt, end = '\r')
                self.backward_induction(genCnt, dCnt, AM, searchMethod)

    

class asianOptionMCsim(asianOption):
    def __init__(self, St, K, r, q, sigma, t, T, initAvg, n, simCnt, repeatCnt):
        super().__init__(St, K, r, q, sigma, t, T, initAvg)

        self.n = n
        self.delta_remain_t = self.remain_t / n
        self.sqrt_delta_remain_t = np.power(self.delta_remain_t, 0.5)
        self.discountFac = np.exp(-r * self.remain_t)

        self.simCnt = simCnt
        self.repeatCnt = repeatCnt

    def sim(self):
        # (n + 1) x simCnt zero matrix
        totalSimCnt = self.simCnt * self.repeatCnt
        self.results = np.zeros((self.n + 1, totalSimCnt))
        self.results[0] = np.ones(totalSimCnt) * np.log(self.St)

        # sampling
        self.step_mu = (self.r - self.q - np.power(self.sigma, 2) / 2) * self.delta_remain_t
        self.step_std = self.sigma * self.sqrt_delta_remain_t

        for i in range(1, self.n+1):
            temp = np.random.normal(self.step_mu, self.step_std, totalSimCnt)
            self.results[i] = self.results[i - 1] + temp

        # computation
        self.STPath = np.exp(self.results[1:,:])
        self.STAvg_post = self.STPath.mean(axis = 0)
        self.w1 = self.n * self.t / self.remain_t + 1
        self.w2 = self.n
        self.STAvg = (self.STAvg_post * self.w2 + np.ones(totalSimCnt) * self.initAvg * self.w1) / (self.w1 + self.w2)
        self.expectedPayoff = ((self.STAvg - self.K) * ((self.STAvg - self.K) > 0)).reshape(self.simCnt, self.repeatCnt).mean(axis = 0)

        ##############################
        self.dsctExPayoff = self.expectedPayoff * self.discountFac
        self.meanPrice = self.dsctExPayoff.mean()
        self.priceStd = self.dsctExPayoff.std()
        self.interval = [
            self.meanPrice - 2 * self.priceStd,
            self.meanPrice,
            self.meanPrice + 2 * self.priceStd
            ]