# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts ADE20K data to TFRecord file format with Example protos."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import random
import sys
import build_data
from six.moves import range
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_multi_string(
    'train_image_folder',
    None,
    'Folder containing trainng images')
tf.app.flags.DEFINE_multi_string(
    'train_image_label_folder',
    None,
    'Folder containing masks for training images')

tf.app.flags.DEFINE_multi_string(
    'val_image_folder',
    None,
    'Folder containing validation images')

tf.app.flags.DEFINE_multi_string(
    'val_image_label_folder',
    None,
    'Folder containing masks for validation')

tf.app.flags.DEFINE_multi_string(
    'test_image_folder',
    None,
    'Folder containing validation images')

tf.app.flags.DEFINE_multi_string(
    'test_image_label_folder',
    None,
    'Folder containing masks for validation')


tf.app.flags.DEFINE_string(
    'output_dir', './tfrecord',
    'Path to save converted tfrecord of Tensorflow example')

tf.app.flags.DEFINE_string(
    'basename', None,
    'Basename to add to tfrecord.')

tf.app.flags.DEFINE_integer('num_shards', 4, 'Number of output tfrecords.')


def _convert_dataset(dataset_split, dataset_dir_list, dataset_label_dir_list, num_shards):
  """Converts a generic image mask dataset into into tfrecord format.

  Args:
    dataset_split: Dataset split (e.g., train, val, test).
    dataset_dir: Dir in which the dataset locates.
    dataset_label_dir: Dir in which the annotations locates.

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  image_formats = ["jpg", "png", "JPEG", "jpeg"]
  all_data_files = []
  for i in range(len(dataset_dir_list)):
    dataset_dir = dataset_dir_list[i]
    img_names = []
    for image_format in image_formats:
      img_names.extend(tf.gfile.Glob(os.path.join(dataset_dir, '*.' + image_format)))

    if dataset_label_dir_list:
      dataset_label_dir = dataset_label_dir_list[i]
      seg_names = []
      new_img_names = []
      for image_file in img_names:
        # get the filename without the extension
        basename = os.path.basename(image_file).split('.')[0]
        # cover its corresponding *_seg.png
        seg = os.path.join(dataset_label_dir, basename+'.png')
        if os.path.isfile(seg):
            seg_names.append(seg)
            new_img_names.append(image_file)
        else:
            print("Couldn't find pair for %s in %s:" % (image_file, dataset_label_dir))
      img_names = new_img_names
      all_data_files += list(zip(img_names, seg_names))
    else:
      all_data_files += img_names

  if dataset_split == 'train':
    random.shuffle(all_data_files)

  num_images = len(all_data_files)
  num_per_shard = int(math.ceil(num_images / num_shards))

  image_reader = build_data.ImageReader(image_format, channels=3)
  if dataset_label_dir_list:
    label_reader = build_data.ImageReader('png', channels=1)
    img_names, seg_names = zip(*all_data_files)
  else:
    img_names = all_data_files
  print("Found %d data files" % num_images)
  for shard_id in range(num_shards):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%s%05d-of-%05d.tfrecord' %
        (dataset_split, FLAGS.basename + '-' if FLAGS.basename else '', shard_id, num_shards))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_images, shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = img_names[i]
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        seg_data = None
        if dataset_label_dir:
            # Read the semantic segmentation annotation.
            seg_filename = seg_names[i]
            seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
            seg_height, seg_width = label_reader.read_image_dims(seg_data)
            if height != seg_height or width != seg_width:
              print('Shape mismatched between image and label. image_name=%s, '
                                 'seg_name=%s, image.shape=(%d, %d), seg.shape=(%d,'
                                 '%d)' % (image_filename, seg_filename, height, width,
                                          seg_height, seg_width))
              continue
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, img_names[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  tf.gfile.MakeDirs(FLAGS.output_dir)

  if FLAGS.train_image_folder and FLAGS.train_image_label_folder:
    print("FLAGS.train_image_folder:", FLAGS.train_image_folder,
          "FLAGS.train_image_label_folder:", FLAGS.train_image_label_folder)
    _convert_dataset(
          'train',
          FLAGS.train_image_folder,
          FLAGS.train_image_label_folder,
          num_shards=FLAGS.num_shards,
      )

  if FLAGS.val_image_folder and FLAGS.val_image_label_folder:
    _convert_dataset('val', FLAGS.val_image_folder, FLAGS.val_image_label_folder,
                     num_shards=FLAGS.num_shards)

  if FLAGS.test_image_folder:
      _convert_dataset('test', FLAGS.test_image_folder, FLAGS.test_image_label_folder,
                       num_shards=FLAGS.num_shards)


if __name__ == '__main__':
    tf.app.run()
