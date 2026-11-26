"""Average two packed Stage 2 checkpoints with identical integer assignments.

Every convolution uses the same code indices in both sources. Averaging the
per-group floating-point centroids therefore averages their decoded weights,
up to the final FP16 rounding. Direct biases and offsets use FP32 rounding.
The output directory contains the blended convolutional graph, the optimized
production graph, and a receipt with source digests and parity results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import numpy_helper

from .fast_stage2_graph import TokenGraph, verify
from .pack_weights import byte_lookup


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def tables(model):
    result = {t.name:t for t in model.graph.initializer}
    if len(result) != len(model.graph.initializer):
        raise ValueError('Duplicate initializer names')
    return result


def structure(model, allowed):
    clone = onnx.ModelProto()
    clone.CopyFrom(model)
    for tensor in clone.graph.initializer:
        if tensor.name in allowed:
            if not tensor.raw_data:
                raise ValueError('Expected inline floating-point parameter bytes')
            tensor.raw_data = bytes(len(tensor.raw_data))
    return sha(clone.SerializeToString())


def mean_parameter(first, second, dtype, fraction):
    a, b = numpy_helper.to_array(first), numpy_helper.to_array(second)
    if (a.dtype != dtype or b.dtype != dtype or a.shape != b.shape or
            not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('Expected matching finite floating-point parameters')
    exact = (1-fraction)*a.astype(np.float64) + fraction*b.astype(np.float64)
    rounded = exact.astype(dtype)
    return rounded, exact, np.count_nonzero(a != b), np.count_nonzero(a != rounded)


def symbol_counts(initializers, name, shape):
    prefix = name.removesuffix('centers')
    groups, width = numpy_helper.to_array(initializers[prefix+'groups']).tolist()
    dimensions = numpy_helper.to_array(initializers[prefix+'shape'])
    octets = numpy_helper.to_array(initializers[prefix+'bytes'])
    if (len(shape) != 2 or groups != shape[0] or shape[1] not in (4,16) or groups < 1 or width < 1 or
            dimensions.ndim != 1 or dimensions.dtype.kind not in 'iu' or (dimensions <= 0).any() or octets.dtype != np.uint8):
        raise ValueError('Invalid packed kernel dimensions or storage')
    size = int(np.prod(dimensions, dtype=np.int64))
    bits = shape[1].bit_length()-1
    if not (groups-1)*width < size <= groups*width or octets.size*(8//bits) != groups*width:
        raise ValueError('Packed assignments do not cover the original kernel')
    lookup = byte_lookup(bits)
    root = prefix.rsplit('/', 2)[0] + '/lookup'
    if root not in initializers or not np.array_equal(numpy_helper.to_array(initializers[root]), lookup):
        raise ValueError('Unrecognized packed byte ordering')
    codes = lookup[octets].reshape(groups,width)
    ids = (np.arange(groups)[:,None]*shape[1]+codes).ravel()[:size]
    return np.bincount(ids,minlength=groups*shape[1]).reshape(shape), size


def blend(first, second, fraction):
    if not 0 < fraction < 1:
        raise ValueError('Choose a blend weight strictly between zero and one')
    result = onnx.ModelProto()
    result.CopyFrom(first)
    left, right = tables(result), tables(second)
    centers = {k for k in left if re.fullmatch(r'_*wpack/[0-9]+/centers',k)}
    affines = {k for k in left if re.fullmatch(r'layers\..+\.(bias|offset)',k)}
    if (not centers or left.keys() != right.keys() or
            structure(first,centers|affines) != structure(second,centers|affines)):
        raise ValueError('Graph or integer assignments differ outside permitted floating-point parameters')
    direct = 0
    affine_error = 0.
    for name in sorted(affines):
        rounded, exact, different, _ = mean_parameter(left[name],right[name],np.float32,fraction)
        direct += int(different)
        affine_error = max(affine_error,float(np.abs(rounded.astype(np.float64)-exact).max()))
        left[name].raw_data = rounded.astype('<f4').tobytes()
    weights = symbols = changed = 0
    squared_error = squared_norm = maximum = 0.
    for name in sorted(centers):
        rounded, exact, _, modified = mean_parameter(left[name],right[name],np.float16,fraction)
        counts, n = symbol_counts(left,name,rounded.shape)
        err = rounded.astype(np.float64)-exact
        squared_error += float(np.sum(counts*err**2))
        squared_norm += float(np.sum(counts*exact**2))
        maximum = max(maximum,float(np.abs(err[counts>0]).max()))
        weights += n
        symbols += rounded.size
        changed += int(modified)
        left[name].raw_data = rounded.astype('<f2').tobytes()
    metrics = dict(coefficients=[1-fraction,fraction], allCodesIdentical=True, weights=weights, centers=symbols,
                   changedFromFirst=changed, differingDirectAffines=direct, maximumAffineRoundoff=affine_error,
                   relativeWeightRoundoff=(squared_error/squared_norm)**.5 if squared_norm else 0.,
                   maxWeightRoundoff=maximum)
    return result, metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('first','second','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--second-weight',type=float,default=.5)
    parser.add_argument('--max-bytes',type=int,default=1_480_935)
    parser.add_argument('--fixtures',type=Path,default=Path(__file__).with_name('parity.json'))
    parser.add_argument('--threads',type=int,default=1)
    a=parser.parse_args()
    if a.output.exists() or min(a.max_bytes,a.threads) < 1:
        parser.error('Select a fresh output directory, positive byte cap, and positive thread count')
    blobs = [p.read_bytes() for p in (a.first,a.second)]
    raw, metrics = blend(*(onnx.load_model_from_string(b) for b in blobs),a.second_weight)
    source = raw.SerializeToString()
    production = TokenGraph(raw).build().SerializeToString()
    if len(production)>a.max_bytes:
        raise ValueError('Production model exceeds the requested byte cap')
    fixtures=np.asarray([row['encoded'] for row in json.loads(a.fixtures.read_bytes())['cases']],np.float32)
    parity=verify(source,production,fixtures,a.threads)
    receipt=dict(format='uttt-fixed-codebook-mean-v1',sources=[sha(b) for b in blobs],rawSha256=sha(source),
                 sha256=sha(production),bytes=len(production),graphParity=parity,**metrics)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=a.output.parent) as tmp:
        folder=Path(tmp)/'result'
        folder.mkdir()
        (folder/'source.onnx').write_bytes(source)
        (folder/'model.onnx').write_bytes(production)
        (folder/'manifest.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        folder.rename(a.output)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
