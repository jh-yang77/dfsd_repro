"""Prepare or verify OOD evaluation folders using archived relative-path manifests."""
import argparse
import json
from pathlib import Path
from dfsd_repro.config import package_path


def prepare(source_root, target_root):
    for file in sorted(package_path('manifests').glob('*_files.json')):
        manifest=json.loads(file.read_text())
        name=manifest['dataset_root']
        for relative in manifest['paths']:
            source=source_root/name/relative
            target=target_root/name/relative
            if not source.is_file():raise FileNotFoundError(source)
            if source.resolve()==target.resolve():continue
            if target.exists() or target.is_symlink():
                raise FileExistsError(f'Refusing to replace existing file: {target}')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(source.resolve())
        print(f"{name}: {manifest['count']} selected files")
    verify(target_root)


def verify(root):
    extensions={'.jpg','.jpeg','.png','.bmp','.webp'}
    for file in sorted(package_path('manifests').glob('*_files.json')):
        manifest=json.loads(file.read_text())
        folder=root/manifest['dataset_root']
        expected=set(manifest['paths'])
        actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in extensions}
        if actual!=expected:
            raise ValueError(f"{folder}: {len(expected-actual)} missing and {len(actual-expected)} extra image paths")
        print(f"Verified {folder}: {len(actual)} images")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,help='Downloaded datasets with the directory layout documented in README')
    p.add_argument('--target-root',type=Path,default=package_path('ood_datasets'))
    args=p.parse_args()
    if args.source_root:prepare(args.source_root,args.target_root)
    else:verify(args.target_root)


if __name__=='__main__':main()
