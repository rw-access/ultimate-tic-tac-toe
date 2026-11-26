import unittest

import numpy as np
import onnx
from onnx import numpy_helper

from training.blend_packed import blend
from training.pack_weights import byte_lookup


def fixture(shift):
    model=onnx.ModelProto()
    data={'wpack/lookup':byte_lookup(2),'wpack/0/centers':np.array([[.2,.4,.65,.8]],np.float16)+np.float16(shift),
          'wpack/0/shape':np.array([3],np.int64),'wpack/0/groups':np.array([1,4],np.int64),
          'wpack/0/bytes':np.array([0b11100100],np.uint8),'layers.early.bias':np.array([.4+shift],np.float32)}
    for name,array in data.items():
        model.graph.initializer.append(numpy_helper.from_array(array,name))
    return model


class BlendPackedTests(unittest.TestCase):
    def test_decoded_arithmetic_mean_including_affines_and_excluding_padding(self):
        a,b=fixture(0),fixture(.05)
        original=a.SerializeToString()
        m,stats=blend(a,b,.3)
        arrays=lambda e:{t.name:numpy_helper.to_array(t) for t in e.graph.initializer}
        x,y,z=map(arrays,(a,b,m))
        expected=.7*x['wpack/0/centers'].astype(float)+.3*y['wpack/0/centers'].astype(float)
        np.testing.assert_array_equal(z['wpack/0/centers'],expected.astype(np.float16))
        np.testing.assert_array_equal(z['layers.early.bias'],(.7*x['layers.early.bias'].astype(float)+.3*y['layers.early.bias'].astype(float)).astype(np.float32))
        self.assertEqual(stats['weights'],3)
        self.assertEqual(a.SerializeToString(),original)
        np.testing.assert_array_equal(z['wpack/0/bytes'],x['wpack/0/bytes'])

    def test_incompatible_codes_and_layout_are_rejected(self):
        a,b=fixture(0),fixture(.1)
        b.graph.initializer[4].raw_data=bytes([0])
        with self.assertRaisesRegex(ValueError,'integer assignments'):
            blend(a,b,.5)
        a,b=fixture(0),fixture(.1)
        b.graph.name='another graph'
        with self.assertRaisesRegex(ValueError,'Graph'):
            blend(a,b,.5)
        with self.assertRaisesRegex(ValueError,'blend weight'):
            blend(a,a,1)
