from math import exp, log
import numpy as np
import newton_raphson
import matplotlib.pyplot as plt


class LogisticRegression:

    iterations_newton_raphson = 5  # Number of iterations in applications of the Newton-Raphson method

    def __init__(self, predictors, responses, group_sizes):
        self.predictors = predictors  # Array holding predictors.
        self.responses = responses  # Array, i'th entry holds number of successes for i'th predictor.
        self.group_sizes = group_sizes  # Array, i'th entry holds number of observations for i'th predictor.

    # Sample statistics
    def get_frequencies(self):  # Returns array holding the success frequencies for each predictor.
        freqs = []
        for i in range(len(self.predictors)):
            freqs.append(self.responses[i] / self.group_sizes[i])

        return freqs

    def get_s(self):
        return sum(self.responses)

    def get_sp(self):
        return np.inner(self.responses, self.predictors)

    # Log-likelihood function and its derivatives
    def log_likelihood(self, alpha, beta):  # Log-likehood function given data with parameters alpha and beta.
        t = self.predictors
        s = self.get_s()
        sp = self.get_sp()
        gs = self.group_sizes
        return sum(gs[i] * log(1 + exp(alpha + beta * t[i])) for i in range(len(t))) - alpha * s - beta * sp

    def score(self, alpha, beta):   # Score function (gradient of the log-likehood function) given data.
        t = self.predictors
        s = self.get_s()
        sp = self.get_sp()
        gs = self.group_sizes
        return np.array([sum(gs[i] * exp(alpha + beta * t[i]) / (1 + exp(alpha + beta * t[i])) for i in range(len(t))) - s,
                        sum(gs[i] * t[i] * exp(alpha + beta * t[i]) / (1 + exp(alpha + beta * t[i])) for i in range(len(t))) - sp])

    def information(self, alpha, beta): # Information function (Hessian of the log-likehood function) given data.
        t = self.predictors
        gs = self.group_sizes
        terms = [exp(alpha + beta * t[i]) / (1 + exp(alpha + beta * t[i])) ** 2 for i in range(len(t))]
        return np.array([[sum(gs[i] * terms[i] for i in range(len(t))),  sum(gs[i] * t[i] * terms[i] for i in range(len(t)))],
                        [sum(gs[i] * t[i] * terms[i] for i in range(len(t))), sum(gs[i] * t[i] ** 2 * terms[i] for i in range(len(t)))]])

    # finding initials guesses for Newton-Raphson
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

    # Calculating estimates for alpha, beta and gamma = LD50
    def get_estimates(self):
        # run newton raphson on score-function
        return newton_raphson.method_twodim(self.score, self.information, self.get_initial_alpha(), self.get_initial_beta(), LogisticRegression.iterations_newton_raphson)

    def get_alpha_hat(self):
        return self.get_estimates()[0]

    def get_beta_hat(self):
        return self.get_estimates()[1]

    def get_gamma_hat(self):
        return -self.get_alpha_hat() / self.get_beta_hat()

    def get_estimated_variance_matrix(self):
        return np.linalg.inv(self.information(self.get_alpha_hat(), self.get_beta_hat()))

    # visualisation
    def regression_curve(self, t):
        return exp(self.get_alpha_hat() + self.get_beta_hat() * t) / (1 + exp(self.get_alpha_hat() + self.get_beta_hat() * t))

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
            return log(p / (1 - p))
        elif p <= 0:
            return np.NINF
        else:
            return np.inf
