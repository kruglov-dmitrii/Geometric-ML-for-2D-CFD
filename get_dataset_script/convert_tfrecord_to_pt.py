#!/usr/bin/env python3
import os
import json
from typing import List, Optional

import numpy as np
import torch
from tfrecord.torch.dataset import TFRecordDataset


def _parse_bytes(b: bytes, dtype: str) -> np.ndarray:
    if 'float' in dtype:
        return np.frombuffer(b, dtype=np.float32)
    return np.frombuffer(b, dtype=np.int32)


def _reshape_arr(arr: np.ndarray, shape: List[int]) -> np.ndarray:
    last = shape[-1]
    first = shape[0]
    if first is not None and first > 1:
        time = first
        nodes = int(arr.size // (time * last))
        return arr.reshape((time, nodes, last))
    else:
        nodes = int(arr.size // last)
        return arr.reshape((nodes, last))


def convert_tfrecord_to_pt(
    tfrecord_path: str,
    meta_path: str,
    out_dir: str,
    max_samples: Optional[int] = None,
) -> List[str]:
    """Convert TFRecord file into per-sample .pt files.

    Returns a list of saved filenames (relative to `out_dir`).
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(meta_path))
    features = meta['features']
    description = {k: 'byte' for k in features.keys()}
    ds = TFRecordDataset(tfrecord_path, index_path=None, description=description)

    index: List[str] = []
    for i, rec in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        sample = {}
        for k, v in rec.items():
            dtype = features[k]['dtype']
            arr = _parse_bytes(v, dtype)
            reshaped = _reshape_arr(arr, features[k]['shape'])
            sample[k] = torch.from_numpy(reshaped)
        out_path = os.path.join(out_dir, f'sample_{i:06d}.pt')
        torch.save(sample, out_path)
        index.append(os.path.basename(out_path))
        if (i + 1) % 50 == 0:
            print(f'Converted {i+1} samples')

    json.dump(index, open(os.path.join(out_dir, 'index.json'), 'w'))
    print('Conversion finished. samples=', len(index))
    return index


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('tfrecord', help='path to TFRecord')
    parser.add_argument('meta', help='path to meta.json')
    parser.add_argument('out', help='output directory for .pt files')
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()
    convert_tfrecord_to_pt(args.tfrecord, args.meta, args.out, args.max_samples)
