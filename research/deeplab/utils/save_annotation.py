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
"""Saves an annotation as one png image.

This script saves an annotation as one png image, and has the option to add
colormap to the png image for better visualization.
"""

import numpy as np
import PIL.Image as img
import tensorflow as tf

from deeplab.utils import get_dataset_colormap


def save_annotation(label,
                    save_dir,
                    filename,
                    add_colormap=True,
                    colormap_type=get_dataset_colormap.get_pascal_name(),
                    original_image=None):
  """Saves the given label to image on disk.

  Args:
    label: The numpy array to be saved. The data will be converted
      to uint8 and saved as png image.
    save_dir: The directory to which the results will be saved.
    filename: The image filename.
    add_colormap: Add color map to the label or not.
    colormap_type: Colormap type for visualization.
  """
  # Add colormap for visualizing the prediction.
  if add_colormap:
    colored_label = get_dataset_colormap.label_to_color_image(
        label, colormap_type)
  else:
    colored_label = label

  array_image = colored_label.astype(dtype=np.uint8)
  if original_image is not None:
    array_image = (array_image + 0.5*original_image).clip(0, 255).astype(np.uint8)
  pil_image = img.fromarray(array_image)
  with tf.gfile.Open('%s/%s.jpg' % (save_dir, filename), mode='w') as f:
    pil_image.save(f, 'JPEG')


def save_np_array(np_array,
                  save_dir,
                  filename):
  """Saves Numpy array np_array to file: save_dir/filename.npy.

  Args:
    np_array: np.ndarray, Numpy array to save.
    save_dir: srt, directory to save the array file.
    filename: str, filename to name the file.
  """
  np.save('%s/%s.npy' % (save_dir, filename), np_array)
