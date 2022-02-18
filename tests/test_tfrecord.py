from enum import IntEnum

import numpy as np

from torch_types import ArrayField, BBoxField, ClassificationField, Instance
from torch_types.io import TFRecordReader, TFRecordWriter


TEST_FILE = "_test.tfrecord"


class _TestClassificationField(ClassificationField):
    class Classes(IntEnum):
        x = 0
        y = 1
        z = 2


class _TestInstance(Instance):
    array: ArrayField
    bbox: BBoxField
    target: _TestClassificationField


def test_roundtrip():
    array = np.ones((10, 5))
    bbox = np.array([[2, 3, 4, 5], [4, 5, 6, 7]])
    target = 1
    instance = _TestInstance(
        array=ArrayField(array),
        bbox=BBoxField(bbox),
        target=_TestClassificationField(clss=target),
    )

    writer = TFRecordWriter(TEST_FILE)
    writer.write(instance)
    writer.write(instance)
    writer.close()

    reader = TFRecordReader(TEST_FILE, _TestInstance)
    assert len(reader) == 2
    for rt_instance in reader:
        assert np.isclose(array, rt_instance.array.value).all()
        assert np.isclose(bbox, rt_instance.bbox.bboxes).all()
        assert rt_instance.target.clss == target
    reader.close()
