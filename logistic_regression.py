import math
import numpy as np
import newton_raphson
import matplotlib.pyplot as plt


class LogisticRegression:

    iterations_newton_raphson = 5 # number of iterations in applications of the Newton-Raphson metod

    def __init__(self, predictors, responses, group_size):
        self.predictors = predictors  # 1d-array holding predictors (log-concentrations)
        self.responses = responses  # 1d-array, i'th entry holds number of successes for i'th predictor
        self.group_size = group_size

    # data
    def get_num_of_obs(self):
        pass

    def get_frequencies(self):
        freqs = []
        for i in range(len(self.predictors)):
            freqs.append(self.responses[i] / self.group_size)

        return freqs

    def get_s(self):
        return sum(self.responses)

    def get_sp(self):
        return np.inner(self.responses, self.predictors)

    # likelihood
    def log_likelihood(self, alpha, beta):
        return self.group_size * sum(math.log(1 + math.exp(alpha + beta * t)) for t in self.predictors) - alpha * self.get_s() - beta * self.get_sp()

    def score(self, alpha, beta):
        return np.array([self.group_size * sum(math.exp(alpha + beta * t) / (1 + math.exp(alpha + beta * t)) for t in self.predictors) - self.get_s(),
                         self.group_size * sum(t * math.exp(alpha + beta * t) / (1 + math.exp(alpha + beta * t)) for t in self.predictors) - self.get_sp()])

    def information(self, alpha, beta):
        terms = [(math.exp(alpha + beta * self.predictors[i]) / (1 + math.exp(alpha + beta * self.predictors[i]))) * (1 / (1 + math.exp(alpha + beta * self.predictors[i]))) for i in range(len(self.predictors))]
        return np.array([[self.group_size * sum(terms), self.group_size * sum(self.predictors[i] * terms[i] for i in range(len(self.predictors)))],
                        [self.group_size * sum(self.predictors[i] * terms[i] for i in range(len(self.predictors))), self.group_size * sum(self.predictors[i] ** 2 * terms[i] for i in range(len(self.predictors)))]])

    def get_initials(self):
        p1, p2 = 0.25, 0.75
        i = 0
        while LogisticRegression.logit(self.get_frequencies()[i]) < LogisticRegression.logit(p1):
            i += 1
        t1 = self.predictors[i]

        while LogisticRegression.logit(self.get_frequencies()[i]) < LogisticRegression.logit(p2):
            i += 1
        t2 = self.predictors[i]

        a = np.array([[1, t1], [1, t2]])
        v = np.array([LogisticRegression.logit(p1), LogisticRegression.logit(p2)])
        return np.linalg.inv(a).dot(v)

    def get_initial_alpha(self):
        return self.get_initials()[0]

    def get_initial_beta(self):
        return self.get_initials()[1]

    def get_estimates(self):
        # run newton raphson on score-function
        return newton_raphson.method_twodim(self.score, self.information, self.get_initial_alpha(), self.get_initial_beta(), LogisticRegression.iterations_newton_raphson)

    def get_alpha_hat(self):
        # return estimate for alpha
        return self.get_estimates()[0]

    def get_beta_hat(self):
        # return estimate for beta
        return self.get_estimates()[1]

    def get_gamma_hat(self):
        # return gamma = - alpha/beta = LD50
        return -self.get_alpha_hat() / self.get_beta_hat()

    def get_estimated_variance_matrix(self):
        return np.linalg.inv(self.information(self.get_alpha_hat(), self.get_beta_hat()))

    def regression_curve(self, t):
        return math.exp(self.get_alpha_hat() + self.get_beta_hat() * t) / (1 + math.exp(self.get_alpha_hat() + self.get_beta_hat() * t))

    def graph(self):
        t = np.linspace(self.predictors[0], self.predictors[-1], 100)
        curve = []
        for s in t:
            curve.append(self.regression_curve(s))

        fig, ax = plt.subplots()
        ax.plot(t, curve)
        ax.scatter(self.predictors, self.get_frequencies())
        plt.show()
        # want data in plot also

    @classmethod
    def set_iterations_newton_raphson(cls, n):
        cls.iterations_newton_raphson = n

    @staticmethod
    def logit(p):
        if 0 < p < 1:
            return math.log(p / (1 - p))
        elif p <= 0:
            return np.NINF
        else:
            return np.inf
