import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from experiments.utils.experiments import load_config, get_sigma, deep_mdt_experiment
from experiments.mvmat.load_data import get_data


def _ami(z, labels, k, seed=0):
    pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(z)
    return adjusted_mutual_info_score(labels, pred)


def main() -> None:
    config, config_path = load_config()
    name = config['data']['name']
    out = os.path.join(config['output_dir'], f"mvmat_{name}_{config['mdt']['method']}")
    os.makedirs(out, exist_ok=True)
    os.system(f'cp {config_path} {out}')

    data = get_data(**config['data'])
    k = config['mdt']['n_components']
    sigmas = [get_sigma(v, config['mdt']['quantile']) for v in data['train']]

    results, encoder = deep_mdt_experiment(
        data=data,
        sigmas=sigmas,
        n_components=k,
        steps=config['mdt']['steps'],
        method=config['mdt']['method'],
        encoder_config=config['encoder'],
        knn=config['mdt'].get('knn'),
    )

    # Reference: classical MDT embedding (truncated SVD of W) on the train set.
    W = results.pop('_W')
    U, s, _ = np.linalg.svd(W, full_matrices=False)
    classical = U[:, 1:k + 1] * s[1:k + 1]

    amis = {
        'deep_train': _ami(results['train']['X_red'], data['train_color'], k),
        'deep_test_oos': _ami(results['test']['X_red'], data['test_color'], k),
        'classical_train (reference)': _ami(classical, data['train_color'], k),
    }
    print(f"\n=== {name} | method={config['mdt']['method']} | {k} clusters ===")
    for key, val in amis.items():
        print(f"  AMI {key:32s}: {val:.3f}")

    np.savez(os.path.join(out, 'embeddings.npz'),
             train=results['train']['X_red'], test=results['test']['X_red'],
             train_color=data['train_color'], test_color=data['test_color'])
    with open(os.path.join(out, 'ami.txt'), 'w') as f:
        for key, val in amis.items():
            f.write(f"{key}: {val:.4f}\n")
    print('done ->', out)


if __name__ == '__main__':
    main()
