import time, json, numpy as np, torch, data, baselines, train, model
torch.set_num_threads(2)
D = data.load('test'); n = 16
y = torch.tensor(D['y'][:n]); u0 = torch.tensor(D['u0'][:n].astype(np.float32)); psf = torch.tensor(D['psf'][:n])
out = {}
def tm(f, reps=1):
    t = time.time()
    for _ in range(reps): f()
    return (time.time() - t) / reps / n
with torch.no_grad():
    out['svd'] = tm(lambda: [baselines.svd_filter(y[i:i+1], 6) for i in range(n)])
    out['asvd'] = tm(lambda: [baselines.asvd_filter(y[i:i+1], 0.13) for i in range(n)])
    out['rpca'] = tm(lambda: [baselines.rpca(y[i:i+1], 2.0) for i in range(n)])
    out['registration'] = tm(lambda: [model.init_motion(y[i:i+1] / y[i:i+1].abs().pow(2).mean().sqrt()) for i in range(n)])
    out['mcsvd'] = tm(lambda: [baselines.mc_svd(y[i:i+1], u0[i:i+1], 4) for i in range(n)])
    out['jpa'] = tm(lambda: [baselines.jpa(y[i:i+1], u0[i:i+1], 0.003, 1.0) for i in range(n)])
    for v in ['durpca', 'cnn', 'full', 'K3', 'K9']:
        net = train.build_model(v); net.load_state_dict(torch.load(f'runs/{v}_s0.pt', weights_only=False)['state']); net.eval()
        out[v] = tm(lambda: [net(y[i:i+1], psf_true=psf[i:i+1], u0=u0[i:i+1]) for i in range(n)])
        out[v + '_params'] = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(json.dumps(out, indent=1)); json.dump(out, open('results/timing.json', 'w'), indent=1)
