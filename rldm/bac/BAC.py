import numpy as np
import runstats

class TabularBAC:
    # Priors
    class TransitionPriorDirichlet:
        def __init__(self, num_states, num_actions, alpha_0=None):
            if alpha_0 is None:
                alpha_0 = 1.0
            assert (alpha_0 > 0)

            self.num_states = num_states
            self.num_actions = num_actions
            self.alphas = np.zeros((num_states, num_actions, num_states)) + alpha_0

        # Update with information of one transition (s, a) -> s'
        def update(self, s, a, s_):
            self.alphas[s][a][s_] += 1.0

        # Sample a model of size (num_states, num_actions, num_states)
        def sample_matrix(self):
            T = np.zeros((self.num_states, self.num_actions, self.num_states))
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    T[s][a] = np.random.dirichlet(alpha=self.alphas[s][a])
            return T

        def get_transition(self, s, a):
            return np.random.choice(self.num_states, p=np.random.dirichlet(alpha=self.alphas[s][a]))

        def get_mean(self):
            T = np.zeros((self.num_states, self.num_actions, self.num_states))
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    T[s][a] = self.alphas[s, a] / np.sum(self.alphas[s, a])
            return T

    class RewardPriorNG:
        def __init__(self, num_states, num_actions, mu_0=0., kappa_0=1., alpha_0=1., beta_0=1.):
            self.running_statistics = []
            self.num_states = num_states
            self.num_actions = num_actions
            self.mu_0 = mu_0
            self.kappa_0 = kappa_0
            self.alpha_0 = alpha_0
            self.beta_0 = beta_0

            self.mu_sa = None
            self.kappa_sa = None
            self.alpha_sa = None
            self.beta_sa = None

            for s in range(num_states):
                inner_row = []
                for a in range(num_actions):
                    inner_row.append(runstats.Statistics())
                self.running_statistics.append(inner_row)

        # Update with information of one reward (s, a) -> r
        def update(self, s, a, r):
            self.running_statistics[s][a].push(r)
            self.mu_sa = None
            self.kappa_sa = None
            self.alpha_sa = None
            self.beta_sa = None

        # Sample a rewrad matrix of size (num_states, num_actions)
        def sample_matrix(self):
            R = np.zeros((self.num_states, self.num_actions))

            if self.mu_sa is None:

                self.mu_sa = np.zeros((self.num_states, self.num_actions))
                self.kappa_sa = np.zeros((self.num_states, self.num_actions))
                self.alpha_sa = np.zeros((self.num_states, self.num_actions))
                self.beta_sa = np.zeros((self.num_states, self.num_actions))

                for s in range(self.num_states):
                    for a in range(self.num_actions):
                        n = len(self.running_statistics[s][a])
                        mean_sa = self.running_statistics[s][a].mean()
                        self.mu_sa[s][a] = (self.kappa_0 * self.mu_0 + n * mean_sa) / (self.kappa_0 + n + 0.)
                        self.kappa_sa[s][a] = self.kappa_0 + n
                        self.alpha_sa[s][a] = self.alpha_0 + n / 2.
                        if len(self.running_statistics[s][a]) < 2:
                            squared_diff = 0.
                        else:
                            squared_diff = self.running_statistics[s][a].variance()
                        self.beta_sa[s][a] = self.beta_0 + 0.5 * squared_diff * n + (self.kappa_0 * n * (
                            mean_sa - self.mu_0) ** 2.) / (2. * self.kappa_0 + n)

            for s in range(self.num_states):
                for a in range(self.num_actions):
                    gamma_sample = np.random.gamma(self.alpha_sa[s][a], self.beta_sa[s][a])
                    normal_sample = np.random.normal(self.mu_sa[s][a],
                                                     np.sqrt(1. / (self.kappa_sa[s][a] * gamma_sample)))
                    R[s][a] = normal_sample
            return R

        def get_reward(self, state, action):
            if self.mu_sa is None:

                self.mu_sa = np.zeros((self.num_states, self.num_actions))
                self.kappa_sa = np.zeros((self.num_states, self.num_actions))
                self.alpha_sa = np.zeros((self.num_states, self.num_actions))
                self.beta_sa = np.zeros((self.num_states, self.num_actions))

                for s in range(self.num_states):
                    for a in range(self.num_actions):
                        n = len(self.running_statistics[s][a])
                        mean_sa = self.running_statistics[s][a].mean()
                        self.mu_sa[s][a] = (self.kappa_0 * self.mu_0 + n * mean_sa) / (self.kappa_0 + n + 0.)
                        self.kappa_sa[s][a] = self.kappa_0 + n
                        self.alpha_sa[s][a] = self.alpha_0 + n / 2.
                        if len(self.running_statistics[s][a]) < 2:
                            squared_diff = 0.
                        else:
                            squared_diff = self.running_statistics[s][a].variance()
                        self.beta_sa[s][a] = self.beta_0 + 0.5 * squared_diff * n + (self.kappa_0 * n * (
                            mean_sa - self.mu_0) ** 2.) / (2. * self.kappa_0 + n)
            s = state
            a = action
            gamma_sample = np.random.gamma(self.alpha_sa[s][a], self.beta_sa[s][a])
            normal_sample = np.random.normal(self.mu_sa[s][a], np.sqrt(1. / (self.kappa_sa[s][a] * gamma_sample)))
            return normal_sample

        def get_mean(self):
            if self.mu_sa is None:

                self.mu_sa = np.zeros((self.num_states, self.num_actions))
                self.kappa_sa = np.zeros((self.num_states, self.num_actions))
                self.alpha_sa = np.zeros((self.num_states, self.num_actions))
                self.beta_sa = np.zeros((self.num_states, self.num_actions))

                for s in range(self.num_states):
                    for a in range(self.num_actions):
                        n = len(self.running_statistics[s][a])
                        mean_sa = self.running_statistics[s][a].mean()
                        self.mu_sa[s][a] = (self.kappa_0 * self.mu_0 + n * mean_sa) / (self.kappa_0 + n + 0.)
                        self.kappa_sa[s][a] = self.kappa_0 + n
                        self.alpha_sa[s][a] = self.alpha_0 + n / 2.
                        if len(self.running_statistics[s][a]) < 2:
                            squared_diff = 0.
                        else:
                            squared_diff = self.running_statistics[s][a].variance()
                        self.beta_sa[s][a] = self.beta_0 + 0.5 * squared_diff * n + (self.kappa_0 * n * (
                            mean_sa - self.mu_0) ** 2.) / (2. * self.kappa_0 + n)

            return self.mu_sa

    # Helper functions
    def sample_models(self, R_params, T_params, N):
        Rs = []
        Ts = []
        for i in range(N):
            Rs.append(R_params.sample_matrix())
            Ts.append(T_params.sample_matrix())

        return Rs, Ts

    def policy_evaluation(self, R, T, policy, gamma):

        V = np.linalg.inv(np.eye(R.shape[0]) - gamma * np.einsum('ijk, ij->ik', T, policy)) \
            @ np.einsum('ij,ij->i', R, policy).reshape(-1, 1)
        Q = R + gamma * T @ V.flatten()

        return V, Q

    def __init__(self, num_states, num_actions, gamma, alpha, num_samples=100, num_iterations=1000):
        self.reward_prior = self.RewardPriorNG(num_states, num_actions)
        self.transition_prior = self.TransitionPriorDirichlet(num_states, num_actions)
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.num_samples = num_samples
        self.w = np.ones((num_states, num_actions))
        self.PI = np.ones((num_states, num_actions)) / num_actions
        self.reset_weights = False

    def get_action(self, state):
        return np.random.choice(self.num_actions, p=self.PI[state])

    def update(self, s, a, r, s_, done):
        self.reward_prior.update(s, a, r)
        self.transition_prior.update(s, a, s_)

    def update_policy(self):
        if self.reset_weights:
            self.w = np.ones((self.num_states, self.num_actions))

        for s in range(self.num_states):
            self.PI[s] = np.exp(self.w[s]) / np.sum(np.exp(self.w[s]))

        for t in range(self.num_iterations):
            VV = np.zeros((self.num_samples, self.num_states))
            QQ = np.zeros((self.num_samples, self.num_states, self.num_actions))

            Rs, Ts = self.sample_models(self.reward_prior, self.transition_prior, self.num_samples)
            for j in range(self.num_samples):
                V, Q = self.policy_evaluation(Rs[j], Ts[j], self.PI, self.gamma)
                VV[j] = V.flatten()
                QQ[j] = Q

            dTheta = np.zeros((self.num_states, self.num_actions))

            for s in range(self.num_states):
                for a in range(self.num_actions):
                    score = 1 - np.exp(self.w[s][a]) / np.sum(np.exp(self.w[s]))
                    dTheta[s][a] = self.alpha * score * (np.mean(QQ[:, s, a])-np.mean(VV[:, s]))
            self.w += dTheta

            for s in range(self.num_states):
                self.PI[s] = np.exp(self.w[s]) / np.sum(np.exp(self.w[s]))

# Epistemic risk-sensitive BAC
class TabularEPBAC(TabularBAC):
    def __init__(self, num_states, num_actions, gamma, alpha, beta, num_samples=100, num_iterations=1000):
        self.reward_prior = self.RewardPriorNG(num_states, num_actions)
        self.transition_prior = self.TransitionPriorDirichlet(num_states, num_actions)
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.num_iterations = num_iterations
        self.num_samples = num_samples
        self.w = np.ones((num_states, num_actions))
        self.PI = np.ones((num_states, num_actions)) / num_actions
        self.reset_weights = False

    def update_policy(self):
        if self.reset_weights:
            self.w = np.ones((self.num_states, self.num_actions))

        for s in range(self.num_states):
            self.PI[s] = np.exp(self.w[s]) / np.sum(np.exp(self.w[s]))

        for t in range(self.num_iterations):
            VV = np.zeros((3, self.num_samples, self.num_states))
            QQ = np.zeros((3, self.num_samples, self.num_states, self.num_actions))

            for i in range(3):
                Rs, Ts = self.sample_models(self.reward_prior, self.transition_prior, self.num_samples)
                for j in range(self.num_samples):
                    V, Q = self.policy_evaluation(Rs[j], Ts[j], self.PI, self.gamma)
                    VV[i, j] = V.flatten()
                    QQ[i, j] = Q

            dTheta = np.zeros((self.num_states, self.num_actions))

            for s in range(self.num_states):
                for a in range(self.num_actions):
                    score = 1 - np.exp(self.w[s][a]) / np.sum(np.exp(self.w[s]))
                    num = np.mean(np.exp(
                        self.beta * (QQ[0, :, s, a] - np.min(QQ[0, :, s, a])) /
                        (np.max(QQ[0, :, s, a]) - np.min(QQ[0, :, s, a]))))
                    den = np.mean(
                        np.exp(self.beta * (QQ[2, :, s, a] - np.min(QQ[2, :, s, a])) /
                               (np.max(QQ[2, :, s, a]) - np.min(QQ[2, :, s, a]))))
                    dTheta[s][a] = self.alpha * score * num \
                                   * (np.mean(QQ[1, :, s, a])-np.mean(VV[1, :, s])) / den
            self.w += dTheta

            for s in range(self.num_states):
                self.PI[s] = np.exp(self.w[s]) / np.sum(np.exp(self.w[s]))

import gym
def main():
    env_list = ['NChain-v0', 'FrozenLake-v0', 'Roulette-v0']

    gamma = 0.99
    alpha = 0.1
    num_samples = 100
    num_iterations = 1000
    max_episodes = 1000
    num_updates = 5
    beta = -1.0

    returns = np.zeros((len(env_list), max_episodes // num_updates, num_updates))

    for i in range(len(env_list)):
        env = gym.make(env_list[i])
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        BAC = TabularEPBAC(num_states, num_actions, gamma, alpha, beta, num_samples, num_iterations)
        s = env.reset()
        episodic_reward = 0
        for t in range(max_episodes):
            if t % (max_episodes // num_updates) == 0:
                BAC.update_policy()
            while True:
                a = BAC.get_action(s)
                s_, r, done, _ = env.step(a)
                BAC.update(s, a, r, s_, done)
                episodic_reward += r
                if done:
                    s = env.reset()
                    returns[i, t % (max_episodes // num_updates), t // (max_episodes // num_updates)] = episodic_reward
                    episodic_reward = 0
                    break
                else:
                    s = s_
if __name__ == "__main__":
    main()