# Copyright (c) 2019 Lightricks. All rights reserved.
import os.path as osp
import pandas as pd
import tensorflow as tf

from object_detection.utils import dataset_util


flags = tf.compat.v1.flags
flags.DEFINE_string("output_path", "", "Path to output TFRecord")
flags.DEFINE_string("annotation_file", "", "Path to CSV annotation file.")
flags.DEFINE_string("images_path", "", "Path to dataset images.")

FLAGS = flags.FLAGS


def str_to_list(x, dtype):
    return list(map(dtype, x.split("|")))


def create_tf_example(example, images_path):

    # Image height
    height = int(example["Height"])

    # Image width
    width = int(example["Width"])

    # Filename of the image. Empty if image is not from file
    filename = example["Filename"]

    # Encoded image bytes
    with tf.io.gfile.GFile(osp.join(images_path, filename), "rb") as fid:
        encoded_image_data = fid.read()

    # b'jpeg' or b'png'
    image_format = example["Format"]

    # List of normalized left x coordinates in bounding box (1 per box)
    xmins = [x / width for x in str_to_list(example["BBox/xmin"], float)]

    # List of normalized right x coordinates in bounding box (1 per box)
    xmaxs = [x / width for x in str_to_list(example["BBox/xmax"], float)]

    # List of normalized top y coordinates in bounding box (1 per box)
    ymins = [x / height for x in str_to_list(example["BBox/ymin"], float)]

    # List of normalized bottom y coordinates in bounding box (1 per box)
    ymaxs = [x / height for x in str_to_list(example["BBox/ymax"], float)]

    # List of string class name of bounding box (1 per box)
    classes_text = str_to_list(example["ClassName"], str)

    # List of integer class id of bounding box (1 per box)
    classes = str_to_list(example["ClassID"], int)

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
    writer = tf.compat.v1.python_io.TFRecordWriter(FLAGS.output_path)
    bbox_data = pd.read_csv(FLAGS.annotation_file, dtype=str)
    data_values = bbox_data.values.tolist()
    data_keys = bbox_data.columns.to_list()

    for example in data_values:
        tf_example = create_tf_example(dict(zip(data_keys, example)), FLAGS.images_path)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == "__main__":
    tf.compat.v1.app.run()
