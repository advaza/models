# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Evaluation script for the DeepLab model.

See model.py for more details and usage.
"""
import ntpath

import numpy as np

# import six
import tensorflow as tf

# from tensorflow.python.platform import tf_logging as logging

from tensorflow.contrib import metrics as contrib_metrics

# from tensorflow.contrib import training as contrib_training

from tensorflow.python.training import (
    monitored_session,
    # session_run_hook,
    # basic_session_run_hooks,
    # training_util,
)
from deeplab import common
from deeplab import model
from deeplab.datasets import data_generator
from pathlib import Path


# from tensorflow.python.summary import summary
from sklearn.metrics import confusion_matrix

flags = tf.app.flags
FLAGS = flags.FLAGS
import yaml

from skimage.io import imsave
import os

flags.DEFINE_string("master", "", "BNS name of the tensorflow server")

# Settings for log directories.

flags.DEFINE_string("eval_logdir", None, "Where to write the event logs.")

flags.DEFINE_string("checkpoint_path", None, "Directory of model checkpoints.")

# Settings for evaluating the model.

flags.DEFINE_integer("eval_batch_size", 1, "The number of images in each batch during evaluation.")

flags.DEFINE_list("eval_crop_size", "513,513", "Image crop size [height, width] for evaluation.")

flags.DEFINE_integer("eval_interval_secs", 60 * 5, "How often (in seconds) to run evaluation.")

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer("atrous_rates", None, "Atrous rates for atrous spatial pyramid pooling.")

flags.DEFINE_integer("output_stride", 16, "The ratio of input to output spatial resolution.")

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float("eval_scales", [1.0], "The scales to resize images for evaluation.")

# Change to True for adding flipped images during test.
flags.DEFINE_bool("add_flipped_images", False, "Add flipped images for evaluation or not.")

flags.DEFINE_integer(
    "quantize_delay_step", -1, "Steps to start quantized training. If < 0, will not quantize model."
)

# Dataset settings.

flags.DEFINE_string("dataset", "pascal_voc_seg", "Name of the segmentation dataset.")

flags.DEFINE_string("eval_split", "val", "Which split of the dataset used for evaluation")

flags.DEFINE_string("dataset_dir", None, "Where the dataset reside.")


flags.DEFINE_integer(
    "max_number_of_evaluations",
    0,
    "Maximum number of eval iterations. Will loop " "indefinitely upon nonpositive values.",
)

# extra - save data
flags.DEFINE_string(
    "save_path", None, "path for saving images of predictions, labels and weights, default None"
)
flags.DEFINE_bool("save_predictions", False, "save predictions if true, default False")
flags.DEFINE_bool("save_labels", False, "save labels if true, default False")
flags.DEFINE_bool("save_weights", False, "save weights if true, default False")
flags.DEFINE_bool("save_images", False, "save original images if true, default False")

# dataset name
flags.DEFINE_string("dataset_name", "", "dataset name for log file and yaml file naming")


def assert_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def divide_no_nan(a, b):
    np.seterr(divide="ignore")
    return np.where(b != 0, a / b, np.zeros_like(a))


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset = data_generator.Dataset(
        dataset_name=FLAGS.dataset,
        split_name=FLAGS.eval_split,
        dataset_dir=FLAGS.dataset_dir,
        batch_size=FLAGS.eval_batch_size,
        crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
        min_resize_value=FLAGS.min_resize_value,
        max_resize_value=FLAGS.max_resize_value,
        resize_factor=FLAGS.resize_factor,
        model_variant=FLAGS.model_variant,
        num_readers=2,
        is_training=False,
        should_shuffle=False,
        should_repeat=False,
    )

    tf.gfile.MakeDirs(FLAGS.eval_logdir)
    tf.logging.info("Evaluating on %s set", FLAGS.eval_split)

    with tf.Graph().as_default():
        samples = dataset.get_one_shot_iterator().get_next()

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
            crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride,
        )

        # Set shape in order for tf.contrib.tfprof.model_analyzer to work properly.
        samples[common.IMAGE].set_shape(
            [FLAGS.eval_batch_size, int(FLAGS.eval_crop_size[0]), int(FLAGS.eval_crop_size[1]), 3]
        )
        if tuple(FLAGS.eval_scales) == (1.0,):
            tf.logging.info("Performing single-scale test.")
            predictions = model.predict_labels(
                samples[common.IMAGE], model_options, image_pyramid=FLAGS.image_pyramid
            )
        else:
            tf.logging.info("Performing multi-scale test.")
            if FLAGS.quantize_delay_step >= 0:
                raise ValueError("Quantize mode is not supported with multi-scale test.")

            predictions = model.predict_labels_multi_scale(
                samples[common.IMAGE],
                model_options=model_options,
                eval_scales=FLAGS.eval_scales,
                add_flipped_images=FLAGS.add_flipped_images,
            )
        predictions = predictions[common.OUTPUT_TYPE]
        predictions = tf.reshape(predictions, shape=[FLAGS.eval_batch_size, -1])
        labels = tf.reshape(samples[common.LABEL], shape=[FLAGS.eval_batch_size, -1])
        weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))

        # Set ignore_label regions to label 0, because metrics.mean_iou requires
        # range of labels = [0, dataset.num_classes). Note the ignore_label regions
        # are not evaluated since the corresponding regions contain weights = 0.
        labels = tf.where(tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

        predictions_tag = "miou"
        for eval_scale in FLAGS.eval_scales:
            predictions_tag += "_" + str(eval_scale)
        if FLAGS.add_flipped_images:
            predictions_tag += "_flipped"

        session_creator = monitored_session.ChiefSessionCreator(
            checkpoint_filename_with_path=FLAGS.checkpoint_path
        )

        # create the folders for saving the data:
        if FLAGS.save_path:
            pred_path = os.path.join(FLAGS.save_path, "predictions")
            labels_path = os.path.join(FLAGS.save_path, "labels")
            weights_path = os.path.join(FLAGS.save_path, "weights")
            images_path = os.path.join(FLAGS.save_path, "original_images")

        # assert paths exist
        if FLAGS.save_predictions:
            assert_path(pred_path)
        if FLAGS.save_labels:
            assert_path(labels_path)
        if FLAGS.save_weights:
            assert_path(weights_path)
        if FLAGS.save_images:
            assert_path(images_path)

        # open the results yaml - if exists
        yaml_file_path = os.path.join(FLAGS.eval_logdir, "%s_results.yaml" % FLAGS.dataset_name)

        if os.path.exists(yaml_file_path):
            yaml_file = open(yaml_file_path)
            yaml_data = yaml.load(yaml_file)
            yaml_file.close()
        else:
            yaml_data = None

        im_size = FLAGS.eval_crop_size[0], FLAGS.eval_crop_size[1]
        batch_index = 0
        with open(yaml_file_path, "a+") as yaml_file:
            with monitored_session.MonitoredSession(session_creator=session_creator) as session:

                while not session.should_stop():
                    (
                        image_name_batch,
                        predictions_batch,
                        labels_batch,
                        weights_batch,
                    ) = session.run([samples[common.IMAGE_NAME], predictions, labels, weights,])

                    for image_index in range(FLAGS.eval_batch_size):

                        image_path = image_name_batch[image_index]
                        pred = predictions_batch[image_index]
                        label = labels_batch[image_index]
                        sample_weights = weights_batch[image_index]

                        # dictionary for image data to save in yaml file
                        image_data_dict = {}

                        image_path = image_path.decode("utf-8")
                        yaml_key = "(%d, %d)" % (batch_index, image_index)

                        # check if already calculated for this image
                        if yaml_data and yaml_key in yaml_data:
                            print("%s already exists\n" % image_path)
                        else:
                            image_data_dict["path"] = image_path

                            # save images
                            if FLAGS.save_path:
                                # n_im = image.reshape(im_size)
                                n_w = sample_weights.reshape(im_size)
                                n_label = label.reshape(im_size) * n_w
                                n_pred = pred.reshape(im_size) * n_w

                                base_name = Path(os.path.basename(image_path)).stem
                                new_name = base_name + ".png"

                                if FLAGS.save_predictions:
                                    imsave(os.path.join(pred_path, new_name), n_pred)
                                if FLAGS.save_labels:
                                    imsave(os.path.join(labels_path, new_name), n_label)
                                if FLAGS.save_weights:
                                    imsave(os.path.join(weights_path, new_name), n_w)
                                # if FLAGS.save_images:
                                #     imsave(os.path.join(image_path, "original_images", new_name), n_im)

                            classes = np.arange(dataset.num_of_classes)

                            confus_mat = confusion_matrix(label, pred, sample_weight=sample_weights)
                            true_positives = np.diag(confus_mat)
                            sum_over_row = np.sum(confus_mat, axis=0)
                            sum_over_col = np.sum(confus_mat, axis=1)
                            # sum_over_row + sum_over_col =
                            #     2 * true_positives + false_positives + false_negatives.
                            denominator = sum_over_row + sum_over_col - true_positives
                            num_valid_entries = np.sum((denominator != 0))
                            iou = divide_no_nan(true_positives, denominator)
                            mean_iou = divide_no_nan(np.sum(iou), num_valid_entries)
                            image_data_dict["mean_iou"] = float(mean_iou)
                            image_data_dict["confusion_mat"] = confus_mat
                            for c in classes:
                                image_data_dict["class_%s_iou" % c] = float(iou[c])

                            yaml.dump({yaml_key: image_data_dict}, yaml_file)

                    batch_index += 1
        yaml_file.close()


if __name__ == "__main__":
    flags.mark_flag_as_required("checkpoint_path")
    flags.mark_flag_as_required("eval_logdir")
    flags.mark_flag_as_required("dataset_dir")
    tf.app.run()
