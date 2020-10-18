# Copyright (c) 2019 Lightricks. All rights reserved.
from allegroai import DatasetVersion, SingleFrame, FrameGroup
import os
import os.path as osp
import pandas as pd
import progressbar
import tensorflow as tf


flags = tf.compat.v1.flags
flags.DEFINE_string("annotation_file", "", "Path to CSV annotation file.")
flags.DEFINE_string("images_path", "", "Path to dataset images (can be S3 path).")
flags.DEFINE_string(
    "split",
    "",
    "Optional. Dataset split, i.e., train, val, test. Each "
    "split will be saved in a different version.",
)
flags.DEFINE_string("dataset_name", "", "Name of allegro dataset.")
FLAGS = flags.FLAGS


def str_to_list(x, dtype):
    return list(map(dtype, x.split("|")))


def create_frame(example, images_path):

    frame = SingleFrame(source=osp.join(images_path, example["Filename"]))

    # Image height
    frame.height = int(example["Height"])

    # Image width
    frame.width = int(example["Width"])

    # Add the frame metadata: format ('jpeg' or 'png') and filename.
    frame.metadata = {"image_format": example["Format"], "filename": example["Filename"]}

    # Annotations:
    # List of normalized left x coordinates in bounding box (1 per box)
    xmins = [x for x in str_to_list(example["BBox/xmin"], int)]

    # List of normalized right x coordinates in bounding box (1 per box)
    xmaxs = [x for x in str_to_list(example["BBox/xmax"], int)]

    # List of normalized top y coordinates in bounding box (1 per box)
    ymins = [x for x in str_to_list(example["BBox/ymin"], int)]

    # List of normalized bottom y coordinates in bounding box (1 per box)
    ymaxs = [x for x in str_to_list(example["BBox/ymax"], int)]

    # List of string class name of bounding box (1 per box)
    classes_text = [s.encode() for s in str_to_list(example["ClassName"], str)]

    # List of integer class id of bounding box (1 per box)
    classes_id = str_to_list(example["ClassID"], int)

    for xmin, xmax, ymin, ymax, class_name, class_id in zip(
        xmins, xmaxs, ymins, ymaxs, classes_text, classes_id
    ):
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin

        frame.add_annotation(
            box2d_xywh=(xmin, ymin, bbox_width, bbox_height),
            labels=[class_name,],
            metadata={"bbox_size": int(bbox_width * bbox_height), "class_id": [class_id,]},
        )

    return frame


def main(_):

    bbox_data = pd.read_csv(FLAGS.annotation_file, dtype=str)
    data_values = bbox_data.values.tolist()
    data_keys = bbox_data.columns.to_list()

    frames = []
    for example in progressbar.progressbar(data_values):
        frames.append(create_frame(dict(zip(data_keys, example)), FLAGS.images_path))

    # Create the dataset if it doesn't exist, and does nothing if it does.
    DatasetVersion.create_new_dataset(FLAGS.dataset_name)
    if FLAGS.split:
        dataset = DatasetVersion.create_version(
            dataset_name=FLAGS.dataset_name, version_name=FLAGS.split
        )
    else:
        dataset = DatasetVersion.get_current(dataset_name=FLAGS.dataset_name)

    dataset.add_frames(frames)


if __name__ == "__main__":
    tf.compat.v1.app.run()
