import numpy
import HW4

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