"""Statistics of the learned weight map lambda_K on the test set (all seeds)."""
import numpy as np, json, data
from scipy.ndimage import binary_dilation
D = data.load('test'); c = (slice(4, -4), slice(4, -4)); out = {'vessel': [], 'micro': [], 'span': []}
for s in range(5):
    lam = np.load(f'results/extra_test_full@{s}.npz')['lam']
    for i in range(len(lam)):
        l = lam[i][c]; V = D['vessel'][i][c]; M = D['micro'][i][c]; B = ~binary_dilation(V, iterations=1)
        out['vessel'].append(np.median(l[B]) / np.median(l[V])); out['micro'].append(np.median(l[B]) / np.median(l[M]))
        out['span'].append(np.percentile(l, 95) / np.percentile(l, 5))
res = {k: dict(geo_mean=float(np.exp(np.mean(np.log(v)))), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75))) for k, v in out.items()}
print(res); json.dump(res, open('results/lambda_stats.json', 'w'), indent=1)
