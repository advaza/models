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
"""Saves an annotation as one png image.

This script saves an annotation as one png image, and has the option to add
colormap to the png image for better visualization.
"""
import cv2
import numpy as np
import PIL.Image as img
import tensorflow as tf

from deeplab.utils import get_dataset_colormap
from matplotlib import gridspec
from matplotlib import pyplot as plt


def save_annotation(label,
                    save_dir,
                    filename,
                    add_colormap=True,
                    normalize_to_unit_values=False,
                    scale_values=False,
                    colormap_type=get_dataset_colormap.get_pascal_name(),
                    add_to_image=None):
  """Saves the given label to image on disk.

  Args:
    label: The numpy array to be saved. The data will be converted
      to uint8 and saved as png image.
    save_dir: String, the directory to which the results will be saved.
    filename: String, the image filename.
    add_colormap: Boolean, add color map to the label or not.
    normalize_to_unit_values: Boolean, normalize the input values to [0, 1].
    scale_values: Boolean, scale the input values to [0, 255] for visualization.
    colormap_type: String, colormap type for visualization.
  """
  # Add colormap for visualizing the prediction.
  if add_colormap:
    colored_label = get_dataset_colormap.label_to_color_image(
        label, colormap_type)
  else:
    colored_label = label
    if normalize_to_unit_values:
      min_value = np.amin(colored_label)
      max_value = np.amax(colored_label)
      range_value = max_value - min_value
      if range_value != 0:
        colored_label = (colored_label - min_value) / range_value

    if scale_values:
      colored_label = 255. * colored_label

  pil_image = img.fromarray(colored_label.astype(dtype=np.uint8))
  with tf.gfile.Open('%s/%s.png' % (save_dir, filename), mode='w') as f:
    pil_image.save(f, 'PNG')


def vis_segmentation(image,
                     seg_map,
                     logits=None,
                     save_dir=None,
                     filename=None,
                     colormap_type=get_dataset_colormap.get_pascal_name(),
                     label_names=None,
                     seg_bg_color=(0,0,0),
                     max_image_dim=1024):
  """Visualizes input image, segmentation map and overlay view."""
  # creates figure with the 3 images in original size

  height, width = image.shape[:2]
  if max(height, width) > max_image_dim:
    scale = max_image_dim / max(height, width)
    new_shape = (int(width*scale), int(height*scale))
    image = cv2.resize(image,
                       new_shape,
                       interpolation=cv2.INTER_AREA)
    seg_map = cv2.resize(seg_map,
                         new_shape,
                         interpolation=cv2.INTER_NEAREST)
    if logits is not None:
      logits = np.stack([
        cv2.resize(logits[:,:,i], new_shape, interpolation=cv2.INTER_LINEAR)
         for i in range(logits.shape[-1])], axis=2)

  dpi = plt.rcParams['figure.dpi']
  fig_size = width * 3 / float(dpi), height / float(dpi)
  #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=fig_size, sharex=True, sharey=True, squeeze=True,
  #                       gridspec_kw={'wspace': 0, 'hspace': 0})
  #title = "image: ", name
  #fig.suptitle(title, fontsize=20)
  #ax[0].imshow(img)
  #ax[1].imshow(mask)
  #ax[2].imshow(masked_img)

  colormap = get_dataset_colormap.create_label_colormap(colormap_type)
  image = image.astype(dtype=np.uint8)

  plt.figure(figsize=fig_size)
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])


  # plt.subplot(grid_spec[1])
  # seg_image = get_dataset_colormap.label_to_color_image(
  #     seg_map, colormap_type).astype(np.uint8)
  # plt.imshow(seg_image)
  # plt.axis('off')
  # plt.title('segmentation map')

  soft_seg_mix = np.full_like(image, fill_value=seg_bg_color, dtype=np.float32)
  soft_overlay = image.astype(np.float32)
  overlay = np.copy(image).astype(np.float32)
  num_classes = logits.shape[2]
  logits[logits < 0.01] = 0
  a = None
  for i in range(1, num_classes):
    color_image = np.full_like(image, fill_value=colormap[i])
    soft_seg_mix += color_image * np.stack([logits[:, :, i]]*3, axis=-1)
    a = color_image * np.stack([logits[:, :, i]]*3, axis=-1)
    overlay[seg_map==i] = (0.3 * overlay[seg_map==i] + 0.7 * color_image[seg_map==i])

  soft_seg_mix = soft_seg_mix.astype(np.uint8).astype(np.float32)
  inds = np.any(soft_seg_mix>20, 2)
  soft_overlay[inds] = 0.3 * soft_overlay[inds] + 0.7 * soft_seg_mix[inds]
  soft_overlay = soft_overlay.astype(np.uint8)
  overlay = overlay.astype(np.uint8)

  plt.subplot(grid_spec[0])
  plt.imshow(a)
  plt.axis('off')
  plt.title('soft segmentation')

  plt.subplot(grid_spec[1])
  plt.imshow(soft_overlay)
  plt.axis('off')
  plt.title('soft segmentation overlay')

  plt.subplot(grid_spec[2])
  plt.imshow(overlay)
  plt.axis('off')
  plt.title('segmentation overlay')

  # plt.subplot(grid_spec[2])
  # plt.imshow(image)
  # plt.imshow(seg_image, alpha=0.7)
  # plt.axis('off')
  # plt.title('segmentation overlay')

  if label_names is not None:
    full_color_map = np.arange(len(label_names)).reshape(len(label_names), 1)
    full_color_map = get_dataset_colormap.label_to_color_image(full_color_map, colormap_type)
    unique_labels = sorted(list(set(np.unique(seg_map)) - {0}))
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
      full_color_map[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), label_names[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)

  plt.grid('off')
  with tf.gfile.Open('%s/%s.png' % (save_dir, filename), mode='w') as f:
    plt.savefig(f)
  cv2.imwrite('%s/%s.png' % (save_dir, filename + 'debug'), cv2.cvtColor(a,cv2.COLOR_RGB2BGR))
