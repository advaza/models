# Copyright (c) 2017 Lightricks. All rights reserved.

import numpy as np
import tensorflow as tf


class ParseConfigFeatures(object):

    @staticmethod
    def _feature(dtype, variable_len=False, array_shape=None, default_value=None):

        if variable_len:
            return tf.VarLenFeature(dtype=dtype)

        elif array_shape:
            return tf.FixedLenFeature(array_shape, dtype, default_value=default_value)

        return tf.FixedLenFeature([], dtype)

    @staticmethod
    def float32_feature(variable_len=False, array_shape=None, default_value=None):
        return ParseConfigFeatures._feature(tf.float32, variable_len, array_shape, default_value)

    @staticmethod
    def int64_feature(variable_len=False, array_shape=None, default_value=None):
        return ParseConfigFeatures._feature(tf.int64, variable_len, array_shape, default_value)

    @staticmethod
    def string_feature(variable_len=False, array_shape=None, default_value=None):
        return ParseConfigFeatures._feature(tf.string, variable_len, array_shape, default_value)


class ExampleParser(object):

    def __init__(self, example_features):
        self.example_features = example_features

    def _process_features_to_parse(self, features_map, selected_features):

        features_to_parse = {}
        process_features = set(selected_features)
        while process_features:

            feature = process_features.pop()
            tensor_or_features = features_map[feature]

            if isinstance(tensor_or_features, dict):
                process_features.update(set(tensor_or_features.keys()))

            else:
                features_to_parse[feature] = tensor_or_features

        return features_to_parse

    def parse(self, example_serialized, selected_features=None, return_as_dict=False):

        features_map = self.example_features.features_map
        post_parsing_process = self.example_features.post_parsing_process

        single_feature = False
        if isinstance(selected_features, str):
            selected_features = [selected_features]
            single_feature = True

        elif selected_features is None:
            selected_features = features_map.keys() | post_parsing_process.keys()
            return_as_dict = True

        features_to_parse = self._process_features_to_parse(features_map, selected_features)

        features = tf.parse_single_example(example_serialized, features=features_to_parse)

        post_parsing_features = set(selected_features) & post_parsing_process.keys()
        for feature_name in post_parsing_features:
            post_parsing_process[feature_name](features)

        if single_feature:
            return features[selected_features[0]]

        elif return_as_dict:
            return {feature_name: features[feature_name] for feature_name in
                    set(selected_features) & features.keys()}

        else:
            return [features[feature_name] for feature_name in selected_features]


class ExampleFeatures(object):

    @staticmethod
    def _to_list(value):

        if not hasattr(value, '__iter__'):
            value = [value]

        value = np.asarray(value)
        value = list(value.flatten())
        return value

    @staticmethod
    def bytes_feature(value):

        bytes_list = ExampleFeatures._to_list(value)
        bytes_list = [s.encode("utf-8") if isinstance(s, str) else s for s in bytes_list]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes_list))

    @staticmethod
    def float_feature(value):

        float_list = ExampleFeatures._to_list(value)
        return tf.train.Feature(float_list=tf.train.FloatList(value=float_list))

    @staticmethod
    def int64_feature(value):

        int_list = ExampleFeatures._to_list(value)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=int_list))


class ExampleProducer(object):

    def __init__(self):
        self.features = {}

    def add_bytes_feature(self, bytes_value, feature_name):
        self.features[feature_name] = ExampleFeatures.bytes_feature(bytes_value)

    def add_float_feature(self, float_value, feature_name):
        self.features[feature_name] = ExampleFeatures.float_feature(float_value)

    def add_int_feature(self, int_value, feature_name):
        self.features[feature_name] = ExampleFeatures.int64_feature(int_value)

    def produce_example(self):
        example_object = tf.train.Example(features=tf.train.Features(feature=self.features))
        return example_object.SerializeToString()


def build_example_from_data_dict(features_data):

  example_producer = ExampleProducer()

  for feature_name, data in features_data.items():

    if isinstance(data, str) or isinstance(data, bytes) or \
            (isinstance(data, np.ndarray) and data.dtype == np.uint8):

      example_producer.add_bytes_feature(data, feature_name=feature_name)

    elif isinstance(data, np.int64) or isinstance(data, int) or \
            (isinstance(data, np.ndarray) and data.dtype == np.int64):

      example_producer.add_int_feature(data, feature_name=feature_name)

    elif isinstance(data, np.float32) or isinstance(data, float) or \
            (isinstance(data, np.ndarray) and data.dtype == np.float32):

      example_producer.add_float_feature(data, feature_name=feature_name)

    else:

      raise TypeError("Type {} is not supported yet!".format(type(data)))

  return example_producer.produce_example()
