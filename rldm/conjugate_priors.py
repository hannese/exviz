import numpy as np
import runstats
from scipy.stats import t, dirichlet, multinomial, multivariate_normal, invwishart

class ConjugateDirichlet:
    def __init__(self, d, alphas_0=None):

        if alphas_0 is None:
            self.alphas_0 = np.ones(d)
        else:
            self.alphas_0 = alphas_0

        self.d = d
        self.alphas = np.zeros(self.d)
        self.N = 0

    def update(self, x):

        self.alphas[x] += 1.0
        self.N += 1

    def reset(self):

        self.alphas = np.zeros(self.d)
        self.N = 0

    def prior(self):
        return dirichlet(self.alphas_0)

    def posterior(self):
        return dirichlet(self.alphas_0 + self.alphas)

    def marginal(self):
        return multinomial(n=self.N, p=(self.alphas_0 + self.alphas) / (self.alphas_0 + self.N))

    def prior_sample(self):
        return dirichlet.rvs(self.alphas_0)

    def posterior_sample(self):
        return dirichlet.rvs(self.alphas_0 + self.alphas)

class ConjugateNormalInverseGamma:

    def __init__(self, mu_0 = 0.0, alpha_0=1.0, beta_0=1.0, kappa_0=1.0):

        self.N = 0
        self.mu_0 = mu_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.kappa_0 = kappa_0

        self.mu = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.kappa = 0.0
        self.running_statistics = runstats.Statistics()

    def update(self, x):

        self.N += 1
        self.running_statistics.push(x)

        mean = self.running_statistics.mean()
        self.mu = (self.kappa_0 * self.mu_0 + self.N * mean) / (self.kappa_0 + self.N + 0.)
        self.kappa = self.kappa_0 + self.N
        self.alpha = self.alpha_0 + self.N / 2.

        if len(self.running_statistics) < 2:
            squared_diff = 0.
        else:
            squared_diff = self.running_statistics.variance()
        self.beta = self.beta_0 + 0.5 * squared_diff * self.N + \
                    (self.kappa_0 * self.N * (mean - self.mu_0) ** 2.) / (2. * self.kappa_0 + self.N)

    def reset(self):

        self.N = 0
        self.mu = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.kappa = 0.0
        self.running_statistics = runstats.Statistics()

    def prior(self):
        pass

    def posterior(self):
        pass

    def marginal(self):

        t_df = 2.0 * self.alpha
        t_mu = self.mu
        t_scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        return t(t_df, t_mu, t_scale)

    def prior_sample(self):
        gamma_sample = np.random.gamma(self.alpha_0, self.beta_0)
        normal_sample = np.random.normal(self.mu_0,
                                         2 * np.sqrt(gamma_sample / (self.kappa_0 * self.kappa_0)))
        return normal_sample

    def posterior_sample(self):
        gamma_sample = np.random.gamma(self.alpha, self.beta)
        normal_sample = np.random.normal(self.mu,
                                         2 * np.sqrt(gamma_sample / (self.kappa * self.kappa)))
        return normal_sample

'''
Bayesian Linear Regression (Minka '98)
https://tminka.github.io/papers/minka-linear.pdf
Class for Multivariate BRL

input               x: (m x 1)
output              y: (d x 1)
coefficient matrix: A: (d x m)
vector noise        e: (d x 1)
noise prior         V: (d x d)
                  Sxx: (m x m)
                  Syx: (d x m)
                  Syy: (d x d)
                 Sy_x: (d x d)

p(y|x, A, V)        ~ N(Ax, V)

'''
class BayesianMultivariateRegression:
    def __init__(self, d, m, S0, N0, a):

        self.d = d
        self.m = m
        self.S0 = S0
        self.N0 = N0
        self.a = a

        assert(S0.shape[0] == d)
        assert(S0.shape[1] == d)

        self.N = 0.0
        self.M = np.zeros((d, m))
        self.A = self.M
        self.K = np.eye((m)) * a
        self.Sxx = self.K
        self.inv_Sxx = np.eye(m)
        self.Syx = np.matmul(self.M, self.K)
        self.Syy = np.matmul(self.Syx, self.M.T)
        self.Sy_x = np.zeros((d, d))

    # y ~N(Sx, V)
    def update(self, x, y):

        assert(y.shape[0] == self.d)
        assert(x.shape[0] == self.m)

        self.N += 1.0
        self.Sxx += np.outer(x, x) # Sxx = X*X'
        self.Syx += np.outer(y, x) # Syx = Y*X'
        self.Syy += np.outer(y, y) # Syy = Y*Y'
        self.inv_Sxx = np.linalg.inv(self.Sxx)
        self.M = np.matmul(self.Syx, self.inv_Sxx)
        self.Sy_x = self.Syy - np.matmul(self.M, self.Syx.T) # //Sy|x = Syy - Syx*Sxx^{-1}*Syx'

    def expected_A(self):
        return self.M

    def expected_V(self):
        return (self.Sy_x + self.N0 * np.eye(self.d)) / (self.N - self.N0)

    # Returns mu, cov
    def sample_model(self):

        df = self.N + self.N0
        scale = self.Sy_x + self.S0

        if self.N + self.N0 < self.Sy_x.shape[0]:
            df = self.Sy_x.shape[0]

        VV = invwishart.rvs(df=df, scale=scale)
        cov = np.kron(self.inv_Sxx, VV)
        MM = multivariate_normal.rvs(mean=self.M.flatten(), cov=cov).reshape(self.d, self.m)

        return MM, VV

    # Return the marginal posterior prob of y
    def posterior(self, x, y):
        xx = self.inv_Sxx * x
        c = 1.0 + np.product(x, xx)
        V = (self.Sy_x + self.S0) * c
        mean = self.M * x
        df = self.N + self.N0 + 1.0
        return t.pdf(x=y, df=df, loc=mean, scale=np.linalg.inv(V))

    def reset(self):

        self.N = 0.0
        self.M = np.empty((self.d, self.m))
        self.A = self.M
        self.K = np.eye(self.m) * self.a
        self.Sxx = self.K
        self.Syx = self.M * self.K
        self.Syy = self.Syx * self.M.T


def _dirichlet(seed = 42, data_points=10000):

    data_points = data_points
    oracle_vector = np.array([0.2, 0.1, 0.6, 0.1])

    dr = ConjugateDirichlet(d = len(oracle_vector))
    np.random.seed(seed)

    data = np.random.choice(range(len(oracle_vector)), p=oracle_vector, size=data_points)
    data2 = []

    for d in data:
        dr.update(d)

    for d in data:
        d2 = dr.posterior_sample()
        data2.append(d2)

    marginal = dr.marginal()

    print("Testing ConjugateDirichlet ...")
    print("Seed: ", seed, "data_points: ", data_points)
    print("Actual vector: ", np.around(oracle_vector, 2), " \t\tSampled mean: ", np.around(np.mean(data2, axis=0), 2), " \t\tMarginal mean: ", np.around(dr.posterior().mean(), 2))

def _normalgamma(seed = 42, data_points=10000):

    oracle_mean = 2.0
    oracle_std = 5.0
    data_points = data_points

    ng = ConjugateNormalInverseGamma(mu_0=0., alpha_0=1.0, beta_0=1.0, kappa_0=1.0)
    np.random.seed(seed)

    data = np.random.normal(oracle_mean, oracle_std, size=data_points)
    data2 = []
    for d in data:
        ng.update(d)

    for d in data:
        d2 = ng.posterior_sample()
        data2.append(d2)

    dist = ng.marginal()

    print("Testing ConjugateNormalInverseGamma ...")
    print("Seed: ", seed, "data_points: ", data_points)
    print("Actual mean: ", np.around(oracle_mean, 2), " \t\tSampled mean: ", np.around(np.mean(data2), 2), " \t\tMarginal mean: ", np.around(dist.mean(), 2))
    print("Actual std: ", np.around(oracle_std, 2), " \t\tSampled std: ", np.around(np.std(data2), 2), " \t\tMarginal std: ", np.around(dist.std(), 2))


'''
    Linear model is Y' = AX' + b + e ~ E
'''
def _bayesianmultivariateregression(seed = 42, data_points = 10000):

    np.random.seed(seed)

    A = np.array([[1.0, 2.0],
                 [3.0, 4.0]])

    b = np.array([0.0, 0.0]).reshape(-1, 1)
    E = invwishart.rvs(2+1, np.eye(2))
    d = 2


    data = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=(data_points))
    Y = (A @ data.T + b + np.random.multivariate_normal(np.zeros(d), E, size=(data_points)).T).T

    expected_A_LS = Y.T @ np.linalg.pinv(data.T)

    pS = np.eye(d)
    blr = BayesianMultivariateRegression(d, d, pS, d+1, 0.001)

    for i in range(data_points):
        blr.update(data[i, :], Y[i, :])

    expected_A_BLR = blr.M.flatten()
    expected_V_BLR = (blr.Sy_x + blr.N0 * np.eye(d)) / (blr.N - blr.N0)

    print("Testing BayesianMultivariateRegression ...")
    print("Seed: ", seed, "data_points: ", data_points)
    print("Actual A: ", np.around(A.flatten(), 2), " \t\tLS A: ", np.around(expected_A_LS.flatten(), 2), " \t\tBMR A: ", np.around(expected_A_BLR.flatten(), 2))
    print("Actual V: ", np.around(E.flatten(), 2), " \t\tLS V: N/A", " \t\tBMR V: ", np.around(expected_V_BLR.flatten(), 2))




def main():

    evals = [1, 10, 100, 1000, 10000]
    seed = 42

    for e in evals:

        _normalgamma(seed, e)
        print("")

    for e in evals:

        _dirichlet(seed, e)
        print("")

    for e in evals:

        _bayesianmultivariateregression(seed, e)
        print("")

if __name__ == "__main__":
    main()
