# Copyright (c) 2019 Lightricks. All rights reserved.
import math
import os
import os.path as osp
import sys
import tensorflow as tf

from allegroai import DataView, IterationOrder, Task, Logger
from object_detection.utils import dataset_util

flags = tf.compat.v1.flags
flags.DEFINE_string("dataset_name", "", "Name of allegro dataset.")
flags.DEFINE_integer("num_shards", 4, "Number of output TFRecords.")
flags.DEFINE_string("output_dir", "", "Path to output TFRecords directory.")
flags.DEFINE_string(
    "split",
    None,
    "Optional. Dataset split, i.e., train, val, test. Each "
    "split will be saved in a different version.",
)
FLAGS = flags.FLAGS


def str_to_list(x, dtype):
    return list(map(dtype, x.split("|")))


def create_tf_example(frame):

    local_image_path = frame.get_local_source()

    # Image height
    height = int(frame.height)

    # Image width
    width = int(frame.width)

    # Filename of the image. Empty if image is not from file
    filename = frame.metadata["filename"].encode()

    # Encoded image bytes
    with tf.io.gfile.GFile(local_image_path, "rb") as fid:
        encoded_image_data = fid.read()

    # b'jpeg' or b'png'
    image_format = frame.metadata["image_format"].encode()

    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    classes_text, classes = [], []
    for annotation in frame.annotations:
        xmin, ymin, bbox_width, bbox_height = annotation.bounding_box_xywh

        xmins.append(float(xmin) / width)
        ymins.append(float(ymin) / height)
        xmaxs.append(float(xmin + bbox_width) / width)
        ymaxs.append(float(ymin + bbox_height) / height)

        classes_text.append(annotation.labels[0].encode())
        classes.append(int(annotation.metadata["class_id"][0]))

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(filename),
                "image/source_id": dataset_util.bytes_feature(filename),
                "image/encoded": dataset_util.bytes_feature(encoded_image_data),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )
    return tf_example


def main(_):

    # Get dataset frames.
    dataview = DataView(iteration_order=IterationOrder.random)
    dataview.add_query(
        dataset_name=FLAGS.dataset_name, version_name=FLAGS.split,
    )
    frames = dataview.to_list()
    dataview.prefetch_files()

    num_images = len(frames)
    num_per_shard = int(math.ceil(num_images / FLAGS.num_shards))

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    for shard_id in range(FLAGS.num_shards):
        output_filename = osp.join(
            FLAGS.output_dir, "%s-%05d-of-%05d.tfrecord" % (FLAGS.split, shard_id, FLAGS.num_shards)
        )
        with tf.compat.v1.python_io.TFRecordWriter(output_filename) as writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write(
                    "\r>> Converting image %d/%d shard %d" % (i + 1, num_images, shard_id)
                )
                frame = frames[i]
                tf_example = create_tf_example(frame)
                writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
    tf.compat.v1.app.run()
