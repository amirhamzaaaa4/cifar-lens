"""Download a published experiment (zip of an artifacts/<name> folder).

Used for hosted demos, where trained weights live in a GitHub Release or on the
Hugging Face Hub rather than in Git:

    python -m lens.fetch https://github.com/<user>/cifar-lens/releases/download/v1.0/resnet20.zip

The archive must contain results.json and model.keras (optionally inside one
top-level folder). The model checksum is verified against results.json.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

REQUIRED = ('results.json', 'model.keras', 'errors.npz')
MAX_BYTES = 200 * 1024 * 1024


def extract_experiment(archive_bytes, destination):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'{destination} already contains files')
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = [n for n in archive.namelist() if not n.endswith('/')]
        if len(names) > 1000 or sum(item.file_size for item in archive.infolist()) > MAX_BYTES:
            raise ValueError('Expanded archive exceeds limits')
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
                raise ValueError(f'Unsafe path in archive: {name}')
        records = [PurePosixPath(n).parent for n in names if PurePosixPath(n).name == 'results.json']
        if len(records) != 1:
            raise ValueError('Archive must contain exactly one experiment')
        prefix = '' if str(records[0]) == '.' else str(records[0]) + '/'
        if prefix and any(not name.startswith(prefix) for name in names):
            raise ValueError('Files must belong to the experiment directory')
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            for name in names:
                relative = name[len(prefix):]
                if not relative:
                    continue
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
            missing = [f for f in REQUIRED if not (staging / f).exists()]
            if missing:
                raise ValueError(f'Archive is missing {", ".join(missing)}')
            results = json.loads((staging / 'results.json').read_text())
            digest = hashlib.sha256((staging / 'model.keras').read_bytes()).hexdigest()
            if digest != results['model_sha256']:
                raise ValueError('model.keras checksum does not match results.json')
            destination.mkdir(parents=True, exist_ok=True)
            for item in staging.iterdir():
                shutil.move(str(item), destination / item.name)
    return destination


def fetch(url, destination):
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError('Archive exceeds 200 MB')
    return extract_experiment(data, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Download and verify a published experiment')
    parser.add_argument('url')
    parser.add_argument('--output', type=Path, default=Path('artifacts/published'))
    args = parser.parse_args(argv)
    print(f'Installed experiment at {fetch(args.url, args.output)}')


if __name__ == '__main__':
    main()
