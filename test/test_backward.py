import torch
import time

import ailoc.common


def manual_grad_test():
    # prepare data
    x_data = torch.rand(1000)
    y_data = 2 * x_data ** 2 + 3 + torch.randn(1000)

    # optimize using aotograd
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    t0 = time.time()
    optimzer = torch.optim.Adam([a, b], lr=0.01)
    for i in range(10000):
        y_model = a * x_data ** 2 + b
        loss = torch.mean((y_model - y_data) ** 2)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        print(loss.item())
    time1 = time.time() - t0

    # optimize using manual grad
    c = torch.tensor(1.0, requires_grad=False)
    d = torch.tensor(1.0, requires_grad=False)
    t0 = time.time()
    optimzer2 = torch.optim.Adam([c, d], lr=0.01)
    for i in range(10000):
        y_model = c * x_data ** 2 + d
        loss = torch.mean((y_model - y_data) ** 2)
        # loss.backward()
        optimzer2.zero_grad()
        c.grad = torch.mean(2 * (y_model - y_data) * x_data ** 2)
        d.grad = torch.mean(2 * (y_model - y_data))
        optimzer2.step()
        print(loss.item())
    time2 = time.time() - t0

    # print results
    print(a, b, c, d, time1, time2)


def crop_grad_test():
    a = torch.randn(size=(100, 128), requires_grad=True)
    b = a**2 + 1
    c = b[10:20, 10:20]
    loss = c.sum()
    loss.backward()
    d= a.grad
    print(a.grad)


if __name__ == '__main__':
    # manual_grad_test()
    crop_grad_test()

