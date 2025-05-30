import numpy as np
import torch
import pandas as pd
import pickle

def get_bench_data(dir, cutoff=-1):
    # test data
    df_vision = pd.read_csv(f'{dir}/benchmark.vision.csv')
    df_lang = pd.read_csv(f'{dir}/benchmark.lang.csv')
    df_all = pd.concat([df_vision, df_lang])

    # IC
    domain = 'IC'
    df = df_all[df_all['Domain'] == domain]
    models = set(df['Model'])
    models = sorted(list(models))
    downstream_groups = {
        'Birds': ['bird_5', 'bird_10', 'bird_25'],
        'CIFAR100': ['c_5', 'c_10', 'c_25'],
        'Caltech101': ['cal_5', 'cal_10', 'cal_25'],
        'ImageNet': ['inet_5', 'inet_10', 'inet_25'],
    }
    IC_data = []
    for group in downstream_groups:
        for downstream in downstream_groups[group]:
            for model in models:
                key = (domain, downstream, model)

                df_subset1 = df[(df['Model'] == model) & (df['Task'] == downstream) & (df['Training'] == 1)]
                xc = np.array(df_subset1['Seen Examples'])
                yc = np.array(df_subset1['Loss'])

                df_subset0 = df[(df['Model'] == model) & (df['Task'] == downstream) & (df['Training'] == 0)]
                xt = np.array(df_subset0['Seen Examples'])
                yt = np.array(df_subset0['Loss'])

                xc, yc = zip(*sorted(zip(xc, yc)))
                xt, yt = zip(*sorted(zip(xt, yt)))

                xc, yc, xt, yt = torch.tensor(xc).float(), torch.tensor(yc).float(), torch.tensor(xt).float(), torch.tensor(yt).float()

                if cutoff != -1:
                    n_context = max(2, int(len(xc) * cutoff))
                    x, y = torch.cat((xc, xt)), torch.cat((yc, yt))
                    xc, yc = x[:n_context], y[:n_context]
                    xt, yt = x[n_context:], y[n_context:]

                IC_data.append((key, xc, yc, xt, yt))

    # NMT
    domain = 'NMT'
    df = df_all[df_all['Domain'] == domain]
    models = set(df['Model'])
    models = sorted(list(models))
    NMT_data = []
    for model in models:
        key = (domain, 'upstream', model)

        df_subset1 = df[(df['Model'] == model) & (df['Training'] == 1)]
        xc = np.array(df_subset1['Seen Examples'])
        yc = np.array(df_subset1['Loss'])

        df_subset0 = df[(df['Model'] == model) & (df['Training'] == 0)]
        xt = np.array(df_subset0['Seen Examples'])
        yt = np.array(df_subset0['Loss'])

        xc, yc = zip(*sorted(zip(xc, yc)))
        xt, yt = zip(*sorted(zip(xt, yt)))

        xc, yc, xt, yt = torch.tensor(xc).float(), torch.tensor(yc).float(), torch.tensor(xt).float(), torch.tensor(yt).float()

        if cutoff != -1:
            n_context = max(2, int(len(xc) * cutoff))
            x, y = torch.cat((xc, xt)), torch.cat((yc, yt))
            xc, yc = x[:n_context], y[:n_context]
            xt, yt = x[n_context:], y[n_context:]

        NMT_data.append((key, xc, yc, xt, yt))

    # LM
    domain = 'LM'
    df = df_all[df_all['Domain'] == domain]
    models = set(df['Model'])
    models = sorted(models, key=lambda x: float(x), reverse=True)
    LM_data = []
    for model in models:
        key = (domain, 'upstream', model)

        df_subset1 = df[(df['Model'] == model) & (df['Training'] == 1)]
        xc = np.array(df_subset1['Seen Examples'])
        yc = np.array(df_subset1['Loss'])

        df_subset0 = df[(df['Model'] == model) & (df['Training'] == 0)]
        xt = np.array(df_subset0['Seen Examples'])
        yt = np.array(df_subset0['Loss'])

        xc, yc = zip(*sorted(zip(xc, yc)))
        xt, yt = zip(*sorted(zip(xt, yt)))

        xc, yc, xt, yt = torch.tensor(xc).float(), torch.tensor(yc).float(), torch.tensor(xt).float(), torch.tensor(yt).float()

        if cutoff != -1:
            n_context = max(2, int(len(xc) * cutoff))
            x, y = torch.cat((xc, xt)), torch.cat((yc, yt))
            xc, yc = x[:n_context], y[:n_context]
            xt, yt = x[n_context:], y[n_context:]
            
        LM_data.append((key, xc, yc, xt, yt))

    # BB
    domain = 'BB'
    df = df_all[df_all['Domain'] == domain]
    tasks = set(df['Task'])
    tasks = sorted(list(tasks))
    BB_data = []
    for task in tasks:
        key = (domain, task, '2.62e+08 Param')

        df_subset1 = df[(df['Task'] == task) & (df['Training'] == 1)]
        xc = np.array(df_subset1['Seen Examples'])
        yc = np.array(df_subset1['Loss'])

        df_subset0 = df[(df['Task'] == task) & (df['Training'] == 0)]
        xt = np.array(df_subset0['Seen Examples'])
        yt = np.array(df_subset0['Loss'])
        
        xc, yc = zip(*sorted(zip(xc, yc)))
        xt, yt = zip(*sorted(zip(xt, yt)))

        xc, yc, xt, yt = torch.tensor(xc).float(), torch.tensor(yc).float(), torch.tensor(xt).float(), torch.tensor(yt).float()

        if cutoff != -1:
            n_context = max(2, int(len(xc) * cutoff))
            x, y = torch.cat((xc, xt)), torch.cat((yc, yt))
            xc, yc = x[:n_context], y[:n_context]
            xt, yt = x[n_context:], y[n_context:]
            
        BB_data.append((key, xc, yc, xt, yt))

    return IC_data, NMT_data, LM_data, BB_data

def get_DD_data(dir, cutoff=-1):
    if cutoff == -1:
        cutoff = 0.5
    else:
        cutoff = 0.5 * cutoff

    with open(f'{dir}/dd_data.pkl', 'rb') as f:
        dd_raw = pickle.load(f)
    labels = {}
    data = []
    for task, x_label, y_label, x, y in dd_raw:
        key = ('DD', task, '-')
        n_context = max(3, min(int(cutoff * len(x) + 0.5), len(x) - 1))
        xc = torch.tensor(x[:n_context]).float()
        yc = torch.tensor(y[:n_context]).float()
        xt = torch.tensor(x[n_context:]).float()
        yt = torch.tensor(y[n_context:]).float()
        data.append((key, xc, yc, xt, yt))
        labels[task] = (x_label, y_label)
    data = sorted(data, key=lambda x: x[0][1])

    return data, labels

def get_nano_data(dir, cutoff=-1):
    if cutoff == -1:
        n_context = 6
    else:
        n_context = max(2, int(6*cutoff))
    n_emb = [6, 12, 24, 48, 96, 192, 384]
    data = []

    # min val loss
    nano_raw = np.load(f'{dir}/min_val_loss_over_n_embd.npy')
    
    for i in range(nano_raw.shape[0]):
        key = ('Nano', f'minval_hyp{i}', '-')
        mv_xc = torch.tensor(n_emb[:n_context]).float()
        mv_yc = torch.tensor(nano_raw[i][:n_context]).float()
        mv_xt = torch.tensor(n_emb[n_context:]).float()
        mv_yt = torch.tensor(nano_raw[i][n_context:]).float()
        data.append((key, mv_xc, mv_yc, mv_xt, mv_yt))

    # final val loss
    nano_raw = np.load(f'{dir}/final_val_loss_over_n_embd.npy')
    for i in range(nano_raw.shape[0]):
        key = ('Nano', f'finalval_hyp{i}', '-')
        fv_xc = torch.tensor(n_emb[:n_context]).float()
        fv_yc = torch.tensor(nano_raw[i][:n_context]).float()
        fv_xt = torch.tensor(n_emb[n_context:]).float()
        fv_yt = torch.tensor(nano_raw[i][n_context:]).float()
        data.append((key, fv_xc, fv_yc, fv_xt, fv_yt))

    return data

def get_colpret_data(dir, cutoff=-1):
    if cutoff == -1:
        cutoff = 0.5
    else:
        cutoff = 0.5 * cutoff

    colpret_raw = np.load(f'{dir}/colpret.npz')
    colpret_raw = dict(sorted(colpret_raw.items()))
    data = []

    for model, values in colpret_raw.items():
        x = values[0] # the number of seen tokens
        y = values[1] # losses
        key = ('ColPreT', '-', model)
        n_context = max(3, min(int(cutoff * len(x) + 0.5), len(x) - 1))
        xc = torch.tensor(x[:n_context]).float()
        yc = torch.tensor(y[:n_context]).float()
        xt = torch.tensor(x[n_context:]).float()
        yt = torch.tensor(y[n_context:]).float()
        data.append((key, xc, yc, xt, yt))

    return data

if __name__ == '__main__':
    IC_data, NMT_data, LM_data, BB_data = get_bench_data('/workspace/scaling-law-2025/data')
    DD_data, DD_labels = get_DD_data('/workspace/scaling-law-2025/data')
    nano_data = get_nano_data('/workspace/scaling-law-2025/data')
    colpret_data = get_colpret_data('/workspace/scaling-law-2025/data')

    # check the data
    print(f'IC_data[0]: {IC_data[0]}')
    print(f'NMT_data[0]: {NMT_data[0]}')
    print(f'LM_data[0]: {LM_data[0]}')
    print(f'BB_data[0]: {BB_data[0]}')
    print(f'DD_data[0]: {DD_data[0]}')
    print(f'nano_data[0]: {nano_data[0]}')
    print(f'colpret_data[0]: {colpret_data[0]}')