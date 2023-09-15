import time

import torch
import numpy as np
import torch.distributions as dist
import torch.nn as nn

import ailoc.common


def count_loss(p_pred, p_gt):
    """
    Computes the loss for the number of emitters. The loss is the negative log likelihood of the
    gaussian distribution.
    """

    # t0 = time.time()
    # for i in range(10000):
    #     num_em_mean = p_pred.sum([-2, -1])
    #     num_em_var = (p_pred - p_pred ** 2).sum([-2, -1])
    #     gauss_num = dist.Normal(num_em_mean, torch.sqrt(num_em_var))
    #     num_em = p_gt.sum([-2, -1])
    #     nll = -gauss_num.log_prob(num_em) * num_em
    # print(time.time()-t0)

    # direct version, should be faster
    num_em_mean = p_pred.sum([-2, -1])
    num_em_var = (p_pred - p_pred ** 2).sum([-2, -1])
    num_em = p_gt.sum([-2, -1])
    nll_direct = (1/2 * (num_em-num_em_mean)**2 / num_em_var + 1/2 * torch.log(2 * np.pi * num_em_var)) * num_em

    return nll_direct


def loc_loss(p_pred, xyzph_pred, xyzph_sig_pred, xyzph_array_gt, mask_array_gt):
    """
    Computes the loss for the localization. The loss is the negative log likelihood of the gaussian mixture model.
    """

    cur_batch_size = p_pred.shape[0]
    h, w = p_pred.shape[-2:]

    p_normed = p_pred / (p_pred.sum([-2, -1]).view(-1, 1, 1))

    pix_inds = tuple((p_pred + 1).nonzero().transpose(1, 0))

    xyzph_mu = xyzph_pred[pix_inds[0], :, pix_inds[1], pix_inds[2]]
    xyzph_mu[:, 0] += ailoc.common.gpu(pix_inds[2]) + 0.5
    xyzph_mu[:, 1] += ailoc.common.gpu(pix_inds[1]) + 0.5
    xyzph_mu = xyzph_mu.reshape(cur_batch_size, 1, -1, 4)
    xyzph_sig = xyzph_sig_pred[pix_inds[0], :, pix_inds[1], pix_inds[2]].reshape(cur_batch_size, 1, -1, 4)
    xyzph_gt = xyzph_array_gt.reshape(cur_batch_size, -1, 1, 4).expand(-1, -1, h*w, -1)

    # direct version, should be faster
    numerator = -1/2 * (xyzph_gt - xyzph_mu) ** 2
    denominator = xyzph_sig ** 2
    log_gauss_4d = (numerator / denominator).sum(3) - 1 / 2 * (torch.log(2 * np.pi * denominator[:, :, :, 0]) +
                                                               torch.log(2 * np.pi * denominator[:, :, :, 1]) +
                                                               torch.log(2 * np.pi * denominator[:, :, :, 2]) +
                                                               torch.log(2 * np.pi * denominator[:, :, :, 3]))
    gmm_coefs = p_normed.reshape(cur_batch_size, 1, h*w)
    gmm_coefs_log = torch.log(gmm_coefs)
    gmm_coefs_logmax = torch.log_softmax(gmm_coefs_log, dim=2)
    gmm_log_direct = torch.sum(torch.logsumexp(log_gauss_4d + gmm_coefs_logmax, dim=2) * mask_array_gt, dim=-1)

    # # use the torch.distributions to compute the log likelihood
    # t0 = time.time()
    # for i in range(10000):
    #     mix = dist.Categorical(p_normed[pix_inds].reshape(cur_batch_size, -1))
    #     comp = dist.Independent(dist.Normal(xyzph_mu[:, 0], xyzph_sig[:, 0]), 1)
    #     gmm = dist.mixture_same_family.MixtureSameFamily(mix, comp)
    #     gmm_log = gmm.log_prob(xyzph_array_gt.transpose(0, 1)).transpose(0, 1)
    #     gmm_log = (gmm_log * mask_array_gt).sum(-1)
    # print(time.time()-t0)

    return -gmm_log_direct


def sample_loss(p_pred, p_gt):
    loss_ce = -(p_gt * torch.log(p_pred) + (1-p_gt) * torch.log(1-p_pred)).sum([-2, -1])
    return loss_ce


def bg_loss(bg_pred, bg_gt):
    loss_bg = nn.MSELoss(reduction='none')(bg_pred, bg_gt)
    return loss_bg.sum([-2, -1])


def compute_log_p_x_given_h(data, model):
    loss = - model + data + data*torch.log(model/data)
    return loss.sum([-2, -1])


def compute_log_q_h_given_x(mu, sig, delta, data):
    num_sample = delta.shape[1]
    gauss = torch.distributions.Normal(mu[:, None].expand(-1, num_sample, -1, -1, -1),
                                       sig[:, None].expand(-1, num_sample, -1, -1, -1))
    loss = gauss.log_prob(data.permute([1, 2, 0, 3, 4])).sum(2)*delta
    return loss.sum([-2, -1])


if __name__ == '__main__':
    xyzph_pred = torch.rand(16, 4, 64, 64, requires_grad=True)
    xyzph_sig_pred = torch.rand(16, 4, 64, 64)
    delta_map_sample = torch.distributions.Bernoulli(torch.ones(16, 40, 64, 64)*0.5).sample()
    xyzph_map_sample = torch.rand(16, 40, 4, 64, 64)
    loss = compute_log_q_h_given_x(xyzph_pred, xyzph_sig_pred, delta_map_sample, xyzph_map_sample)
    print(loss.detach().numpy())
