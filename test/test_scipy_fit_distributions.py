import numpy as np
import scipy.stats
import matplotlib.pyplot as plt  # matplotlib must be installed to plot

rng = np.random.default_rng()
dist = scipy.stats.poisson
mu = 100
data = dist.rvs(mu, size=1000, random_state=rng)

bounds = [(data.min(), data.max())]
res = scipy.stats.fit(dist, data, bounds)

res.params


res.plot()
plt.show()