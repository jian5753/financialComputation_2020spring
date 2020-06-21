import numpy as np
import sys

class lookbackNode():
    def __init__(self, St):
        self.St = St
        self.S_max_tLst = []
        self.payoffDict = dict()

    def insertS_max_t(self, toInsert):
        if len(self.S_max_tLst) == 0:
            self.S_max_tLst.append(toInsert)
        else:
            for i, S_max in enumerate(self.S_max_tLst):
                if S_max < toInsert:
                    print('S_max < toInsert')
                    self.S_max_tLst.insert(i, toInsert)
                    return None
                if S_max == toInsert:
                    return None
            self.S_max_tLst.append(toInsert)

    def dictionary(self):
        for s_max_t in self.S_max_tLst:
            self.payoffDict[s_max_t] = 0

class lookbackOption():
    def __init__(self, St, r, q, sigma, t, T, S_max_t):
        self.St = St
        self.r = r
        self.q = q
        self.sigma = sigma
        self.t = t
        self.T = T
        self.remain_t = self.T - self.t
        self.S_max_t = S_max_t        
                
class biTree(lookbackOption):
    def __init__(self, St, r, q, sigma, t, T, S_max_t, n):
        super().__init__(St, r, q, sigma, t, T, S_max_t)
        
        self.n = n
        self.delta_remain_t = self.remain_t / n
        self.sqrt_delta_remain_t = np.power(self.delta_remain_t, 0.5)
        self.deltaDiscountFac = np.exp(-r * self.delta_remain_t)
        

        self.u = np.exp(self.sigma * self.sqrt_delta_remain_t)
        self.d = 1 / self.u
        self.p = (np.exp((self.r -self.q)* self.delta_remain_t) - self.d) / (self.u - self.d)

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

    def forward_tracking(self, currentGen, currentDcnt):
        #print("##############", currentGen, currentDcnt, "##########")

        # dParent is the parrent with lower St then current St,
        # uParent is the parrent with higher St then cuurent one.
        # print(f'\r{currentGen}, {currentDcnt}', end = '\r')
        currentNode = self.tree[currentGen][currentDcnt]
        #print(currentNode, currentNode.St)
        uParent = None
        dParent = None

        #forward tracking 
        if currentDcnt != 0: # has uParrent 
            uParent = self.tree[currentGen - 1][currentDcnt-1]
            for s_max_u in uParent.S_max_tLst:
                if s_max_u >= currentNode.St:
                    currentNode.insertS_max_t(s_max_u)
                else:
                    currentNode.insertS_max_t(currentNode.St)
                    break

        if currentDcnt != currentGen: # has dParrent
            dParent = self.tree[currentGen - 1][currentDcnt]
            for s_max_d in dParent.S_max_tLst:
                if s_max_d >= currentNode.St:
                    currentNode.insertS_max_t(s_max_d)
                else:
                    currentNode.insertS_max_t(currentNode.St)
                    break
        
        #print(currentNode.S_max_tLst)
        #print('-----------------------------------------')

    def fast_forward_tracking(self, currentGen, currentDcnt):
        currentNode = self.tree[currentGen, currentDcnt]
        currentS = currentNode.St

        genCntSup = currentGen - currentDcnt
        genCntInf = max(self.generateCnt_star, currentGen - currentDcnt * 2)

        if currentS < self.S_max_t:
            currentNode.S_max_tLst.append(self.S_max_t)

        for i in range(genCntSup, genCntInf - 1, -1):
            currentNode.S_max_tLst.append(self.tree[i][0].St)

    def backwardInduction(self, currentGen, currentDcnt, mode = 'am'):
        #print(f'{currentGen}, {currentDcnt}', end = '\r ')
        #print("## backward induction", currentGen, currentDcnt, "##")

        # dChild is the child node with lower St than current St,
        # uChild is the child n
        # ode with higher St than cuurent one.

        currentNode = self.tree[currentGen][currentDcnt]
        #print(currentNode, currentNode.St)
        uChild = self.tree[currentGen + 1][currentDcnt]
        #print(uChild.payoffDict)
        dChild = self.tree[currentGen + 1][currentDcnt + 1]

        #backward induction 
        for s_max_t in currentNode.payoffDict:
            try:
                uValue = uChild.payoffDict[s_max_t]
            except KeyError:
                try:
                    uValue = uChild.payoffDict[self.STij(currentGen + 1, currentDcnt)]
                    #uValue = uChild.payoffDict[uChild.S_max_tLst[-1]]
                except KeyError:
                    print(currentGen, currentDcnt)
                    raise KeyError
                except IndexError:
                    print(currentGen, currentDcnt)
                    print(uChild.S_max_tLst)
            dValue = dChild.payoffDict[s_max_t]
            expectedPayoff = (uValue * self.p + dValue * (1 - self.p)) * self.deltaDiscountFac
            if mode == 'am':
                earlyExPayoff = s_max_t - currentNode.St
                #print('early exercise:',s_max_t, currentNode.St, earlyExPayoff)
                expectedPayoff = max(expectedPayoff, earlyExPayoff)
            currentNode.payoffDict[s_max_t] = expectedPayoff
        
        #print('-----------------------------------------')        

    def compute(self, am=False, bonus1=False):
        #find generateCnt_star
        for i in range(0, self.n+1):
            if self.STij(i, 0) >= self.S_max_t:
                self.generateCnt_star = i
                break

        for genCnt in range(self.n + 1):
            for dCnt in range(0, genCnt + 1):
                self.tree[genCnt][dCnt] = lookbackNode(self.STij(genCnt, dCnt))

        self.tree[0][0].insertS_max_t(self.S_max_t)

        #print("forward_tracking...")
        if bonus1:
            for genCnt in range(1, self.n + 1):
                for dCnt in range(0, genCnt + 1):
                    self.fast_forward_tracking(genCnt, dCnt)
        else:
            for genCnt in range(1, self.n + 1):
                for dCnt in range(0, genCnt + 1):
                    self.forward_tracking(genCnt, dCnt)

        # compute the payoff of each node in last generation
        for endNode in self.tree[-1]:
            endNode.dictionary()
            for s_max_t in endNode.payoffDict:
                endNode.payoffDict[s_max_t] = max(s_max_t - endNode.St, 0)

        # backward induction
        #print("backward_induction...")
        if am:
            mode = 'am'
        else:
            mode = 'eu'

        for genCnt in range(self.n - 1, -1, -1):
            for dCnt in range(0, genCnt + 1):
                self.tree[genCnt][dCnt].dictionary()
                self.backwardInduction(genCnt, dCnt, mode = mode)
        self.backwardInduction(0, 0, mode = mode)

class lookBackMCsim(lookbackOption):
    def __init__(self, St, r, q, sigma, t, T, S_max_t, n, simCnt, repeatCnt):
        super().__init__(St, r, q, sigma, t, T, S_max_t)

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
        self.ln_s_max_T = self.results.max(axis= 0)
        self.ln_s_max_T = np.array([self.ln_s_max_T, np.ones(totalSimCnt) * np.log(self.S_max_t)]).max(axis = 0)
        self.s_max_T = np.exp(self.ln_s_max_T)
        self.ST = np.exp(self.results[-1])
        self.expectedPayoff = (self.s_max_T - self.ST).reshape(self.simCnt, self.repeatCnt).mean(axis = 0)
        self.dsctExPayoff = self.expectedPayoff * self.discountFac
        self.meanPrice = self.dsctExPayoff.mean()
        self.priceStd = self.dsctExPayoff.std()
        self.interval = [
            self.meanPrice - 2 * self.priceStd,
            self.meanPrice,
            self.meanPrice + 2 * self.priceStd
            ]
        

if __name__ == "__main__":
    pass
else:
    print('0611')