
from process_data import fetch_tfrecords
import tensorflow as tf

# Prints the keys of the TFRecords in the dataset
# Used to determine what keys the tfrecords_to_images function should parse
def inspect_tfrecord_keys():
    raw_dataset = fetch_tfrecords()
    for raw in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        for key in example.features.feature:
            print(key)
'''
image/format
image/width
image/filename
image/object/bbox/xmin
image/encoded
image/height
image/object/class/text
image/object/bbox/ymin
image/object/class/label
image/object/bbox/ymax
image/object/bbox/xmax
'''