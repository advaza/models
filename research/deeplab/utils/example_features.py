# Copyright (c) 2017 Lightricks. All rights reserved.

from example_utils import ParseConfigFeatures


class ExampleFeatures(object):
    """Parses an Example proto buffer containing a training example of a training image.

    The output of the build_image_data.py script is a dataset
    containing serialized Example protocol buffers. Each Example proto buffer contains
    the following fields:

    :param example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    :return:
     image_buffer: Tensor tf.string containing the contents of a JPEG file.
    """

    def __init__(self):

        self.features_map = {
            "image/encoded": ParseConfigFeatures.string_feature(default_value=''),
            "image/filename": ParseConfigFeatures.string_feature(default_value=''),
            "image/format": ParseConfigFeatures.string_feature(default_value='jpeg'),
            "image/height": ParseConfigFeatures.int64_feature(default_value=0),
            "image/width": ParseConfigFeatures.int64_feature(default_value=0),
            "image/segmentation/class/encoded":
                ParseConfigFeatures.string_feature(default_value=''),
            "image/segmentation/class/format":
                ParseConfigFeatures.string_feature(default_value='png'),
        }

        self.post_parsing_process = {}
