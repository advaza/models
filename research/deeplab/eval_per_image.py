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
import six
import tensorflow as tf

# from tensorflow.python.platform import tf_logging as logging

from tensorflow.contrib import metrics as contrib_metrics

from tensorflow.contrib import training as contrib_training

from tensorflow.python.training import (
    monitored_session,
    session_run_hook,
    basic_session_run_hooks,
    training_util,
)
from deeplab import common
from deeplab import model
from deeplab.datasets import data_generator
from pathlib import Path


from tensorflow.python.summary import summary
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
    "num_classes", 0, "Number of classes in the dataset, not including the background."
)


flags.DEFINE_integer(
    "max_number_of_evaluations",
    0,
    "Maximum number of eval iterations. Will loop " "indefinitely upon nonpositive values.",
)

# extra
flags.DEFINE_string("save_path", None, "path for saving images of predictions, labels and weights")
flags.DEFINE_bool("save_predictions", True, "save predictions if true, default True")
flags.DEFINE_bool("save_labels", True, "save labels if true, default True")
flags.DEFINE_bool("save_weights", True, "save weights if true, default True")
flags.DEFINE_bool("save_images", True, "save original images if true, default True")


def assert_path(path):
    if os.path.exists(path):
        os.mkdir(path)


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
        num_classes=FLAGS.num_classes,
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
        predictions = tf.reshape(predictions, shape=[-1])
        labels = tf.reshape(samples[common.LABEL], shape=[-1])
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

        # Define the evaluation metric.
        metric_map = {}
        num_classes = dataset.num_of_classes
        metric_map["eval/%s_overall" % predictions_tag] = tf.metrics.mean_iou(
            labels=labels, predictions=predictions, num_classes=num_classes, weights=weights
        )
        # IoU for each class.
        one_hot_predictions = tf.one_hot(predictions, num_classes)
        one_hot_predictions = tf.reshape(one_hot_predictions, [-1, num_classes])
        one_hot_labels = tf.one_hot(labels, num_classes)
        one_hot_labels = tf.reshape(one_hot_labels, [-1, num_classes])
        for c in range(num_classes):
            predictions_tag_c = "%s_class_%d" % (predictions_tag, c)
            tp, tp_op = tf.metrics.true_positives(
                labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c], weights=weights
            )
            fp, fp_op = tf.metrics.false_positives(
                labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c], weights=weights
            )
            fn, fn_op = tf.metrics.false_negatives(
                labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c], weights=weights
            )
            tp_fp_fn_op = tf.group(tp_op, fp_op, fn_op)
            iou = tf.where(tf.greater(tp + fn, 0.0), tp / (tp + fn + fp), tf.constant(np.NaN))
            metric_map["eval/%s" % predictions_tag_c] = (iou, tp_fp_fn_op)

        (metrics_to_values, metrics_to_updates) = contrib_metrics.aggregate_metric_map(metric_map)

        session_creator = monitored_session.ChiefSessionCreator(
            checkpoint_filename_with_path=FLAGS.checkpoint_path
        )

        # create the folders for saving the data:
        pred_path = os.path.join(FLAGS.save_path, "predictions")
        labels_path = os.path.join(FLAGS.save_path, "labels")
        weights_path = os.path.join(FLAGS.save_path, "weights")
        images_path = os.path.join(FLAGS.save_path, "original_images")

        if FLAGS.save_predictions:
            assert_path(pred_path)
        if FLAGS.save_labels:
            assert_path(labels_path)
        if FLAGS.save_weights:
            assert_path(weights_path)
        if FLAGS.save_images:
            assert_path(images_path)

        with monitored_session.MonitoredSession(session_creator=session_creator) as session:

            while not session.should_stop():
                (
                    metrics_updates_batch,
                    metrics_results_batch,
                    image_name_batch,
                    images_batch,
                    predictions_batch,
                    labels_batch,
                    weights_batch,
                ) = session.run(
                    [
                        metrics_to_updates,
                        metrics_to_values,
                        samples[common.IMAGE_NAME],
                        samples[common.IMAGE],
                        predictions,
                        labels,
                        weights,
                    ]
                )

                # creates log file (info about process and calculations)
                # and yaml file (output information)

                yaml_file_path = "/cnvrg/output/mhp_results.yaml"
                if os.path.exists(yaml_file_path):
                    yaml_file = open(yaml_file_path)
                    yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
                    yaml_file.close()
                else:
                    yaml_data = None

                logfile = open("/cnvrg/output/mhp_log.txt", "a+")
                with open("/cnvrg/output/mhp_results.yaml", "a+") as yaml_file:
                    image_data_dict = {}

                    # get original image path
                    for image_path in image_name_batch:
                        image_path = image_path.decode("utf-8")
                        logfile.writelines(["original image path:%s" % image_path, "\n"])

                    image_name = ntpath.basename(image_path)

                    # check if already calculated for this image
                    if yaml_data and image_name in yaml_data:
                        logfile.write("%s already exists\n" % image_name)
                    else:
                        logfile.writelines(["image name:%s" % image_name, "\n"])
                        image_data_dict["path"] = image_path

                        image = images_batch
                        label = labels_batch
                        w = np.array(weights_batch, dtype=np.int32)
                        pred = predictions_batch

                        # save images
                        if FLAGS.save_path:
                            n_im = image.reshape((513, 513))
                            n_w = w.reshape((513, 513))
                            n_label = label.reshape((513, 513)) * n_w
                            n_pred = pred.reshape((513, 513)) * n_w
                            base_name = Path(image_name).stem
                            new_name = base_name + ".png"

                            if FLAGS.save_predictions:
                                imsave(os.path.join(pred_path, new_name), n_pred)
                            if FLAGS.save_labels:
                                imsave(os.path.join(labels_path, new_name), n_label)
                            if FLAGS.save_weights:
                                imsave(os.path.join(weights_path, "weights", new_name), n_w)
                            if FLAGS.save_images:
                                imsave(os.path.join(image_path, "original_images", new_name), n_im)

                        classes = np.arange(FLAGS.num_classes) + 1
                        all_iou = []  # for calculation mean iou for image

                        # calc for each class
                        for c in classes:
                            if c not in pred:
                                logfile.writelines(
                                    ["class %s" % str(c), " not in prediction ", "\n"]
                                )
                                if c in label:
                                    logfile.writelines(["class %s" % str(c), " in label ", "\n"])
                                else:
                                    logfile.writelines(
                                        ["class %s" % str(c), " also not in label ", "\n"]
                                    )

                            c_label = 1 * (label == c)
                            c_pred = 1 * (pred == c)
                            c_m = confusion_matrix(c_label, c_pred, sample_weight=weights).ravel()

                            if len(c_m) < 4:  # only if all 0 or all 1
                                tp = tn = fp = fn = 0
                                if np.all(c_pred * c_label):  # if all 1
                                    tp = c_m[0]
                                else:  # all is 0
                                    tn = c_m[0]

                                c_m = tn, fp, fn, tp
                            else:
                                tn, fp, fn, tp = c_m

                            # calc iou for this class
                            iou = np.NaN
                            if tp + fn + fp > 0.0:
                                iou = tp / (tp + fn + fp)

                            if not np.isnan(iou):
                                all_iou.append(iou)

                            # confusion matrix in %
                            sum_cm = np.sum(c_m)
                            if sum_cm > 0:
                                c_m_p = c_m / sum_cm
                            else:
                                c_m_p = 0, 0, 0, 0
                            tn_p, fp_p, fn_p, tp_p = c_m_p

                            lines = [
                                "class " + str(c) + " iou: " + str(iou) + "\n",
                                "tn=" + str(tn_p) + "\n",
                                "fp=" + str(fp_p) + "\n",
                                "fn=" + str(fn_p) + "\n",
                                "tp=" + str(tp_p) + "\n",
                            ]
                            logfile.writelines(lines)

                            c_m_data = [float(tn_p), float(fp_p), float(fn_p), float(tp_p)]
                            image_data_dict["confusion_mat_" + str(c)] = c_m_data
                            image_data_dict["class_" + str(c) + "_iou"] = float(iou)

                        # mean iou calc
                        mean_iou = np.mean(all_iou)
                        image_data_dict["mean_iou"] = float(mean_iou)
                        lines = ["mean_iou: " + str(mean_iou) + "\n"]

                        # add the calc (mean for all images so far) from original eval code to log file
                        for m, v in metrics_results_batch.items():
                            lines.append(str(m) + ": " + str(v) + "\n")
                        logfile.writelines(lines)

                        # update yaml file with image info
                        yaml.dump({image_name: image_data_dict}, yaml_file)

                    yaml_file.close()
                    logfile.close()

                # #             print(weights_batch.shape)
                # #             print("w=", np.unique(weights_batch))
                #             idx = np.where(weights_batch == 1)
                #             print("idx=", idx)
                #             vals_to_ignore = np.unique(labels_batch[idx])
                #             print(vals_to_ignore)
                #             print(len(np.where(weights_batch != 1)[0]))

        # summary_ops = []
        # for metric_name, metric_value in six.iteritems(metrics_to_values):
        #   op = tf.summary.scalar(metric_name, metric_value)
        #   op = tf.Print(op, [metric_value], metric_name)
        #   summary_ops.append(op)
        #
        # summary_op = tf.summary.merge(summary_ops)
        # summary_hook = contrib_training.SummaryAtEndHook(
        #     log_dir=FLAGS.eval_logdir, summary_op=summary_op)
        # hooks = [summary_hook]
        #
        # num_eval_iters = None
        # if FLAGS.max_number_of_evaluations > 0:
        #   num_eval_iters = FLAGS.max_number_of_evaluations
        #
        # if FLAGS.quantize_delay_step >= 0:
        #   contrib_quantize.create_eval_graph()
        #
        # contrib_tfprof.model_analyzer.print_model_analysis(
        #     tf.get_default_graph(),
        #     tfprof_options=contrib_tfprof.model_analyzer
        #     .TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        # contrib_tfprof.model_analyzer.print_model_analysis(
        #     tf.get_default_graph(),
        #     tfprof_options=contrib_tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
        # contrib_training.evaluate_repeatedly(
        #     checkpoint_dir=FLAGS.checkpoint_dir,
        #     master=FLAGS.master,
        #     eval_ops=list(metrics_to_updates.values()),
        #     max_number_of_evaluations=num_eval_iters,
        #     hooks=hooks,
        #     eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == "__main__":
    flags.mark_flag_as_required("checkpoint_path")
    flags.mark_flag_as_required("eval_logdir")
    flags.mark_flag_as_required("dataset_dir")
    tf.app.run()
