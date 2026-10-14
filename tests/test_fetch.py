import hashlib
import io
import json
import zipfile
import pytest
from lens.fetch import extract_experiment


def archive(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        for name, data in files.items():
            z.writestr(name, data)
    return stream.getvalue()


def experiment(prefix='', model=b'weights', sha=None):
    return archive({prefix + 'model.keras': model, prefix + 'errors.npz': b'npz',
                    prefix + 'results.json': json.dumps({'model_sha256': sha or hashlib.sha256(model).hexdigest()}),
                    prefix + 'samples/cat_0.png': b'png'})


def test_extracts_flat_or_single_folder_archive(tmp_path):
    for prefix in ('', 'resnet20/', 'artifacts/resnet20/'):
        target = extract_experiment(experiment(prefix), tmp_path / prefix.strip('/') / 'out')
        assert (target / 'model.keras').read_bytes() == b'weights'
        assert (target / 'samples' / 'cat_0.png').exists()


def test_rejects_checksum_mismatch_and_unsafe_paths(tmp_path):
    with pytest.raises(ValueError):
        extract_experiment(experiment(sha='0' * 64), tmp_path / 'a')
    assert not (tmp_path / 'a').exists()
    with pytest.raises(ValueError):
        extract_experiment(archive({'../evil.txt': b'x'}), tmp_path / 'b')
    for name in ('..\\evil.txt', 'C:/evil.txt'):
        with pytest.raises(ValueError):
            extract_experiment(archive({name: b'x'}), tmp_path / 'c')
