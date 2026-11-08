"""Learning invariants, safe Fashion-MNIST loading, and reproducible partitions."""
from pathlib import Path
import gzip
import hashlib
import json
import struct
import urllib.request
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
RAW_NAMES = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
             't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
MD5 = ['8d4fb7e6c68d591d4c3dfef9ec88bf0d', '25c81989df183df01b3e8a0aad5dffbe',
       'bef4ecab320f06d8554ea6380940ec79', 'bb300cfdad3c16e7a12a480ee83cd310']

class SmallCNN(nn.Module):
    """Original supplied CNN: 12,810 parameters, no convolution padding."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3)
        self.c2 = nn.Conv2d(16, 32, 3)
        self.fc = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, dict):
        return {k: to_numpy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_numpy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_numpy(v) for v in value)
    return value


def to_torch(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {k: to_torch(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_torch(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_torch(v) for v in value)
    return value


def initial_state(seed):
    torch.manual_seed(seed)
    return to_numpy(SmallCNN().state_dict())


def fedavg(states, sizes):
    if not states or len(states) != len(sizes) or min(sizes) <= 0:
        raise ValueError('FedAvg requires nonempty states and positive sample counts')
    weights = np.asarray(sizes, dtype=np.float64) / sum(sizes)
    return {k: sum(np.float32(w) * state[k] for w, state in zip(weights, states))
            for k in states[0]}


def partition(labels, clients, alpha, seed):
    if clients < 1 or clients > len(labels) or alpha <= 0:
        raise ValueError('Require 1 <= clients <= examples and alpha > 0')
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        parts = [[] for _ in range(clients)]
        for cls in np.unique(labels):
            ids = rng.permutation(np.flatnonzero(labels == cls))
            probs = rng.dirichlet(np.full(clients, alpha))
            cuts = (np.cumsum(probs) * len(ids)).astype(int)[:-1]
            for part, chunk in zip(parts, np.split(ids, cuts)):
                part.extend(chunk.tolist())
        if all(parts):
            return [np.array(sorted(p), dtype=np.int64) for p in parts]
    raise ValueError('Could not find nonempty Dirichlet partition; increase alpha')


def batch_order(indices, seed, round_id, client, epoch):
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_id, client, epoch, 731]))
    return rng.permutation(indices)


def read_idx(path, image):
    with gzip.open(path, 'rb') as f:
        raw = f.read()
    magic, count = struct.unpack('>II', raw[:8])
    expected = 2051 if image else 2049
    if magic != expected:
        raise ValueError(f'Bad IDX magic in {path}')
    if image:
        rows, cols = struct.unpack('>II', raw[8:16])
        if (rows, cols) != (28, 28) or len(raw) != 16 + count * 784:
            raise ValueError('Invalid image dimensions or payload')
        return np.frombuffer(raw, np.uint8, offset=16).reshape(count, 1, 28, 28).copy()
    if len(raw) != 8 + count:
        raise ValueError('Invalid label payload')
    return np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)


def prepare_data(cfg, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    raw = cache
    sources = []
    for name, expected in zip(RAW_NAMES, MD5):
        p = raw / (name + '.gz')
        if not p.exists():
            p = cache / (name + '.gz')
            if not p.exists():
                url = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/' + name + '.gz'
                tmp = p.with_suffix('.download')
                urllib.request.urlretrieve(url, tmp)
                tmp.replace(p)
        if hashlib.md5(p.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Official Fashion-MNIST checksum mismatch: {p}')
        sources.append(p)
    meta = dict(subset=cfg['subset'], subset_seed=cfg['subset_seed'],
                normalization={'mean': 0.2860, 'std': 0.3530},
                source_sha256={p.name: sha(p) for p in sources})
    identity = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]
    dest = cache / f'fashion-{identity}.npz'
    if not dest.exists():
        x, y, xt, yt = [read_idx(p, i % 2 == 0) for i, p in enumerate(sources)]
        ids = np.random.default_rng(cfg['subset_seed']).permutation(len(y))[:cfg['subset']]
        np.savez(dest, x=x[ids], y=y[ids], xt=xt, yt=yt, subset_indices=ids)
    with np.load(dest, allow_pickle=False) as data:
        if len(data['y']) != cfg['subset'] or len(data['yt']) != 10000:
            raise ValueError('Dataset cache size mismatch')
    meta.update(sha256=sha(dest), file=dest.name, test_examples=10000)
    atomic_json(cache / f'fashion-{identity}.json', meta)
    return dest, meta
