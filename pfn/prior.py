from functools import partial
import torch
import numpy as np
import math
from scipy import stats

betacdf = stats.beta.cdf

def M3(rng, x, minmax_samp=True, up=False):
    beta = rng.lognormal(-1, 0.5)
    c = rng.lognormal(-2, 1)
    gamma = rng.lognormal(0, 1)
    y = beta*(1/x + gamma)**c

    y = (y - y[-1]) / (y[0] - y[-1])
    if up: y *= -1
    if minmax_samp:
        y0 = rng.uniform(0.2, 1.2)
        # y0 = 1
        y1 = rng.uniform(0, y0)
        y = y * (y0 - y1) + y1
    return y

def M4(rng, x, proc=1000, minmax_samp=True, up=False):
    alpha = rng.lognormal(0, 0.5)
    beta = rng.lognormal(-1, 0.5)
    c = -rng.lognormal(0, 0.5)
    if alpha < 1e-5:
        y = beta*(x**c)
    else:
        f = (beta*(x**c)).reshape(-1, 1)
        b = 1
        w = np.linspace(0, b, proc).reshape(1, -1)
        loss = np.abs(w - f * (b - w)**alpha)
        indx_min = np.argmin(loss, axis=1)
        y = w[0, indx_min]

    y = (y - y[-1]) / (y[0] - y[-1])
    if up: y *= -1
    if minmax_samp:
        y0 = rng.uniform(0.2, 1.2)
        # y0 = 1
        y1 = rng.uniform(0, y0)
        y = y * (y0 - y1) + y1
    return y

def upward(rng, x):
    beta = math.exp(rng.uniform(0.5, 1))
    gamma = rng.lognormal(0, 0.1)

    y = betacdf(x, beta, beta)**gamma
    return y

def sample_nobreak(
        rng,
        X,
        context_size,
        var_lnloc,
        var_lnscale,
        seq_len,
    ):
    while True:
        f = M3 if rng.uniform(0, 1) < 0.5 else M4
        Y = f(rng, X)

        # add noise
        var = np.exp(
            rng.normal(loc=var_lnloc, scale=var_lnscale)
        )
        Y += rng.normal(np.zeros_like(Y), var)

        # Constraint: lower bound
        if (Y < 0).sum() > 0:
            continue

        # constrain: Nan
        if np.isnan(Y).sum() > 0:
            continue
        break

    return Y

def sample_downward(
        rng,
        X,
        context_size,
        var_lnloc,
        var_lnscale,
        seq_len,
    ):

    f_components = {
        "M3": M3,
        "M4": M4,
    }

    cutoff = X[context_size]
    reject = {'cutoff':0, 'break_idx':0, 'lb':0, 'nan':0}
    while True:
        # print(reject)
        n_brk = rng.randint(1, 3)
        priors = rng.choice(["M3", "M4"], size=n_brk+1, replace=True)

        piece_lens = rng.uniform(min(max(cutoff-0.05, 0), 0.2), 0.4, size=n_brk+1)
        brks = [piece_lens[:i+1].prod() for i in range(n_brk)]
        brks = np.sort(np.array(brks))

        # Constraint: the last break should be earlier than cutoff
        brks_normalizer = brks[-1] / min(brks[-1], cutoff-0.05)
        brks = np.append(brks / brks_normalizer, 1.)

        brks_indices = np.searchsorted(X, brks)
        brks_indices[-1] = seq_len
        
        # Constraint: the first break index should be larger than 0
        if (brks_indices == 0).sum() > 0:
            brks_indices[brks_indices == 0] = 1

        # first piece
        Y = f_components[priors[0]](rng, X)[:brks_indices[0]]
        for i in range(1, n_brk+1):
            start_idx, end_idx = brks_indices[i-1], brks_indices[i]
            y = f_components[priors[i]](rng, X)[start_idx-1:end_idx]
            y = y - y[0] + Y[-1]
            Y = np.concatenate([Y, y[1:]])

        # add noise
        var = np.exp(
            rng.normal(loc=var_lnloc, scale=var_lnscale)
        )
        Y += rng.normal(np.zeros_like(Y), var)

        # Constraint: lower bound
        if (Y < 0).sum() > 0:
            reject['lb'] += 1
            continue

        # constrain: Nan
        if np.isnan(Y).sum() > 0:
            reject['nan'] += 1
            continue

        break

    return Y

def sample_upward(
        rng,
        X,
        context_size,
        var_lnloc,
        var_lnscale,
        seq_len,
    ):

    f_components = {
        "M3": M3,
        "M4": M4,
    }

    cutoff = X[context_size]
    n_brk = 2
    reject = {'cutoff':0, 'break_idx':0, 'lb':0, 'ub':0, 'nan':0}
    while True:
        # print(reject)
        priors = rng.choice(["M3", "M4"], size=n_brk+1, replace=True)
        piece_lens = rng.uniform(min(max(cutoff-0.05, 0), 0.2), 0.4, size=n_brk+1)
        brks = [piece_lens[:i+1].prod() for i in range(n_brk)]
        brks = np.sort(np.array(brks))

        # Constraint: the first break should be earlier than cutoff
        brks_normalizer = max(brks[0], cutoff-0.05) / brks[0]
        brks_normalizer = brks[0] / min(brks[0], cutoff-0.05)
        brks = np.append(brks / brks_normalizer, 1.)

        brks_indices = np.searchsorted(X, brks)
        brks_indices[-1] = seq_len

        # Constraint: the break index should be larger than 0
        if (brks_indices == 0).sum() > 0:
            brks_indices[brks_indices == 0] = 1

        # first piece
        Y = f_components[priors[0]](rng, X)[:brks_indices[0]]

        # middle piece
        start_idx, end_idx = brks_indices[0], brks_indices[1]
        x = X[start_idx-1:end_idx]
        delta_x = x[-1] - x[0]
        slope = rng.uniform(0.01, 0.7)
        delta_y = slope * delta_x
        x = (x-x[0])/x[-1]
        y = upward(rng, x)
        y = y * delta_y
        y = y - y[0] + Y[-1]
        Y = np.concatenate([Y, y[1:]])

        # last piece
        start_idx, end_idx = brks_indices[1], brks_indices[2]
        y = f_components[priors[2]](rng, X)[start_idx-1:end_idx]
        y = y - y[0] + Y[-1]
        Y = np.concatenate([Y, y[1:]])

        # add noisese
        var = np.exp(
            rng.normal(loc=var_lnloc, scale=var_lnscale)
        )
        Y += rng.normal(np.zeros_like(Y), var)

        # Constraint: lower & upper bound
        if ((Y < 0).sum() > 0):
            reject['lb'] += 1
            continue

        if ((Y > Y[0]).sum() > 0):
            reject['ub'] += 1
            continue

        # constrain: Nan
        if np.isnan(Y).sum() > 0:
            reject['nan'] += 1
            continue

        break

    return Y

def sample_from_prior(
        rng,
        seq_len,
        context_size,
        var_lnloc,
        var_lnscale,
        epsilon=1e-2
    ):
    X = np.concatenate([np.random.uniform(epsilon, 1, seq_len-2), [epsilon, 1]])
    X = np.sort(X)
    p = rng.uniform(0, 1)
    if p < 0.7:
        Y = sample_nobreak(rng, X, context_size, var_lnloc, var_lnscale, seq_len)
    elif p < 0.9:
        Y = sample_downward(rng, X, context_size, var_lnloc, var_lnscale, seq_len)
    else:
        Y = sample_upward(rng, X, context_size, var_lnloc, var_lnscale, seq_len)

    return X, Y
    
def create_get_batch_func(
        prior,
        var_lnloc,
        var_lnscale,
    ):

    return partial(
        get_batch,
        prior=prior,
        var_lnloc=var_lnloc,
        var_lnscale=var_lnscale,
    )

# function producing batches for PFN training
def get_batch(
    batch_size,
    prior,
    var_lnloc,
    var_lnscale,
    seq_len=None,
    device="cpu",
    **_,
):
    if seq_len is None:
        seq_len = np.random.choice(np.logspace(1.7, 2.7, num=1000).astype(int))
    x = np.empty((batch_size, seq_len), dtype=float)
    y_noisy = np.empty((batch_size, seq_len), dtype=float)
    context_size = max(2, np.random.randint(int(seq_len*0.1), int(seq_len*0.9)+1))

    for i in range(batch_size):
        x[i], y_noisy[i] = prior(
            np.random,
            seq_len=seq_len,
            context_size=context_size,
            var_lnloc=var_lnloc, 
            var_lnscale=var_lnscale,
            )
    
    x = torch.from_numpy(x).unsqueeze(-1).to(device)
    y_noisy = torch.from_numpy(y_noisy).to(device)

    # changes
    x = x.float()
    y_noisy = y_noisy.float()

    return x, y_noisy, context_size