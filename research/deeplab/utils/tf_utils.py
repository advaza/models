# Copyright (c) 2017 Lightricks. All rights reserved.
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf


def run_queue_runner_session(tensors_to_evaluate, process_values_function, num_steps=None):

    # Create a session for reading the data
    session = tf.Session()

    # Start input enqueue threads.
    coordinator = tf.train.Coordinator()
    threads = None
    try:
        # Initialize variables.
        session.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

        iter = 0
        while not coordinator.should_stop() and (num_steps is None or iter < num_steps):
            tensors_values = session.run(tensors_to_evaluate)
            process_values_function(tensors_values)
            iter += 1

    except Exception as e:
        coordinator.request_stop(e)

    finally:
        # When done, ask the threads to stop.
        coordinator.request_stop()
        # Wait for threads to finish.
        coordinator.join(threads)


def process_features_from_tfrecord(tfrecord_file, parser, selected_features=None,
                                   return_as_dict=False, shuffle=True, num_epochs=None):

    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(
        [tfrecord_file],
        shuffle=shuffle,
        num_epochs=num_epochs
    )
    _, serialized_example = reader.read(filename_queue)

    return parser.parse(
        serialized_example,
        selected_features=selected_features,
        return_as_dict=return_as_dict,
    )


def inspect_ckpt(ckpt_file, get_name_to_weight=True, get_name_to_shape=False):

    ret_val = []

    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()

    if get_name_to_shape:
        ret_val.append(var_to_shape_map)

    if get_name_to_weight:
        name_to_weight = {tensor_name: reader.get_tensor(tensor_name)
                          for tensor_name in var_to_shape_map}
        ret_val.append(name_to_weight)

    if len(ret_val) == 1:
        return ret_val[0]

    return tuple(ret_val)


def inspect_tfrecord(tfrecord_file, id_feature, parser, features=None, selected_ids=None):
    """
    Usage Example:

    ...
    from example_features import ExampleFeatures
    from example_utils import ExampleParser

    data = inspect_tfrecord(
      tfrecord_file=tfrecord_path,
      id_feature="image/filename",
      parser=ExampleParser(ExampleFeatures())
    )
    ...

    :param tfrecord_file:
    :param id_feature:
    :param parser:
    :param features:
    :param selected_ids:
    :return:
    """
    features_values = {}

    def save_tensors_to_dict(evaluated_features):

        # Either save data of all IDs or save the data of selected IDs.
        if not selected_ids or evaluated_features[id_feature] in selected_ids:
            features_values[evaluated_features[id_feature]] = evaluated_features

    selected_features = [id_feature] + features if features else None
    features_tensors = process_features_from_tfrecord(
        tfrecord_file=tfrecord_file,
        selected_features=selected_features,
        parser=parser,
        shuffle=False,
        return_as_dict=True,
        num_epochs=1
    )

    run_queue_runner_session(
        tensors_to_evaluate=features_tensors,
        process_values_function=save_tensors_to_dict,
    )

    return features_values
