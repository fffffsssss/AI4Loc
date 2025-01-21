import torch

# rosenbrock function (a = 1, b = 100)
# https://en.wikipedia.org/wiki/Rosenbrock_function
def f(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

# random initial iterate. Every re-run of this code will yield a different
# initial iterate
d = 30
xk = torch.rand(2,1) * d - d/2

# same initial iterate, different optimizers
xk1 = xk.clone()
xk1.requires_grad = True
xk2 = xk.clone()
xk2.requires_grad = True

# LBFGS parameters
max_iter = 1000
gnorm_tol = 1e-5
tolerance_change = 1e-5
# history_size must match max_iter to keep all search directions. Avoids the
# .pop() method internally
history_size = max_iter

# max_iter steps, no external loop
optimizer1 = torch.optim.LBFGS([xk1],
                               max_iter = max_iter,
                               history_size = history_size,
                               tolerance_grad = gnorm_tol,
                               tolerance_change = tolerance_change,
                               line_search_fn = "strong_wolfe")

def closure1():
    optimizer1.zero_grad()
    y = f(xk1)
    y.backward()
    return y

# single step, yes external loop
optimizer2 = torch.optim.LBFGS([xk2],
                               max_iter = 1,
                               history_size = history_size,
                               tolerance_grad = gnorm_tol,
                               tolerance_change = tolerance_change,
                               line_search_fn = "strong_wolfe")

def closure2():
    optimizer2.zero_grad()
    y = f(xk2)
    y.backward()
    return y

# comparison
# optimizer1
optimizer1.step(closure1)

"""
For reference, here are some of the available keys for the state of the optimizer:
optimizer1.state[optimizer1._params[0]][<key string>]

Key strings and definitions
"d" : search direction at current iteration
"t" : step size at current iteration
"old_dirs" : differences in successive gradients up to history_size (y vectors)
"old_stps" : differences in successive iterates up to history_size (s vectors)
"ro" : 1 / (y.T @ s) at current iterate
"n_iter" : number of iterations so far
"""

# repeat the same for optimizer2 using the same number of iterations
num_iter = optimizer1.state[optimizer1._params[0]]['n_iter']

# optimizer2
for _ in range(num_iter):
    optimizer2.step(closure2)

# compare recorded differences in successive gradients (y vector) and
# recorded differences in successive steps (s vector)
y1 = optimizer1.state[optimizer1._params[0]]['old_dirs']
y2 = optimizer2.state[optimizer2._params[0]]['old_dirs']
s1 = optimizer1.state[optimizer1._params[0]]['old_stps']
s2 = optimizer2.state[optimizer2._params[0]]['old_stps']

ys_equal = all([torch.all(z1 == z2) for z1,z2 in zip(y1,y2)])
ss_equal = all([torch.all(z1 == z2) for z1,z2 in zip(s1,s2)])

print("y vectors equal? {}\ns vectors equal? {}".format(ys_equal,ss_equal))