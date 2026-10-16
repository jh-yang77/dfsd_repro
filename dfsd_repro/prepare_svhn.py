"""Recreate the archived SVHN evaluation subset from the official test file."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from dfsd_repro.config import package_path


def prepare(root: Path):
    manifest = json.loads(package_path('manifests', 'svhn_test_indices.json').read_text())
    source = root / 'test_32x32.mat'
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('SVHN source differs from the archived official test file; do not substitute another split.')
    data = loadmat(source)
    indices = np.asarray(manifest['indices'], dtype=np.int64)
    selected = {'X': data['X'][:, :, :, indices], 'y': data['y'][indices]}
    target = root / 'selected_test_32x32.mat'
    if target.exists():
        saved = loadmat(target)
        if not all(np.array_equal(saved[k], selected[k]) for k in selected):
            raise ValueError('Existing selected_test_32x32.mat differs; move it aside explicitly before regenerating.')
    else:
        savemat(target, selected)
    print(f'Verified {len(indices)} images and labels: {target}')
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=package_path('ood_datasets', 'svhn'))
    prepare(p.parse_args().root)


if __name__ == '__main__':
    main()
