# Logistic-Regression-with-Python

We are given a number of observations with values 0 or 1, each of which is associated with some predictor.
We assume that the probability p of some observation being 1 is given by logit(p) = alpha + beta * t for
unknown alpha and beta where t is the associated predictor and logit(p) = log(p / (1 - p)). Finding alpha and
beta given data is called LOGISTIC REGRESSION.

For example, imagine that a number of flies is exposed to a substance containing some toxic, where the
concentration of the toxic in the substance is varying. We could then try to model the probability of a fly
dying given that it has been a certain concentration of the toxic using the logistic regression model. (Usually,
one uses the log of concentration as the predictor in this case.)

An instance of the LogisticRegression class represents some data suitable for logistic regression and has
methods for performing logistic regression and visualising the results. The parameters are estimated using
maximum likehood estimation. The maximisation is performed using the Newton-Raphson method.

Reference: Ernst Hansen: Introduktion til Matematisk Statistik, Københavns Universitet

Erik Lange, 10-06-2021
