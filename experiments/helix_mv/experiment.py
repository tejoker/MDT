import os
import sys
import numpy as np

from experiments.utils.experiments import load_config, get_sigma, deep_mdt_experiment
from experiments.helix_mv.load_data import get_data


def main() -> None:
    config, config_path = load_config()
    # Make the MDT package (mdt_operator / random_mdt / mdt_contrastive) importable.
    sys.path.insert(0, config['mdt_repo'])

    output_dir = os.path.join(config['output_dir'], 'helix_mv_' + config['mdt']['method'])
    os.makedirs(output_dir, exist_ok=True)
    os.system(f'cp {config_path} {output_dir}')

    data = get_data(**config['data'])
    sigmas = [get_sigma(v, config['mdt']['quantile']) for v in data['train']]

    results, encoder = deep_mdt_experiment(
        data=data,
        sigmas=sigmas,
        n_components=config['mdt']['n_components'],
        steps=config['mdt']['steps'],
        method=config['mdt']['method'],
        encoder_config=config['encoder'],
        knn=config['mdt'].get('knn'),
    )

    # Quick visual sanity check: 2D embedding colored by the helix parameter.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for sub in ('train', 'test'):
            z = results[sub]['X_red']
            plt.figure(figsize=(4, 4))
            plt.scatter(z[:, 0], z[:, 1], c=data[f'{sub}_color'], s=5, cmap='twilight')
            plt.title(f"MDT deep embedding ({sub})")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'embedding_{sub}.png'), dpi=150)
            plt.close()
    except Exception as e:
        print('plot skipped:', e)

    np.savez(
        os.path.join(output_dir, 'embeddings.npz'),
        train=results['train']['X_red'], test=results['test']['X_red'],
        train_color=data['train_color'], test_color=data['test_color'],
    )
    encoder.save(os.path.join(output_dir, 'mv_encoder.keras'))
    print('done ->', output_dir)


if __name__ == '__main__':
    main()
