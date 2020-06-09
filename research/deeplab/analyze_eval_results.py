import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread
import os
from pathlib import Path

LUT = np.random.randint(255, size=(256, 3))
LUT[0] = [0, 0, 0]
LUT[127] = [0, 255, 0]
LUT[255] = [255, 0, 0]


def three_groups(iuo_arr, names, percent):
    """
    split data to three groups: by median,
    third group is percent images from all images that has worst score.
    :param iuo_arr: array of iou per image
    :param names: array of image names respective to iuo_arr
    :param percent: the percent of images for group 3
    :return: 3 groups of images names according to split
    """
    median = np.median(iuo_arr)
    group_1 = np.where(iuo_arr >= median)

    half_iou = iuo_arr[iuo_arr < median]
    half_iou_sorted = np.sort(half_iou)
    percent /= 50  # because using half of list
    num = int(len(half_iou_sorted) * percent)
    partition = half_iou_sorted[num]
    group_3 = np.where(iuo_arr < partition)

    group_2 = np.where((iuo_arr < median) & (iuo_arr >= partition))

    g1_names = names[group_1]
    g2_names = names[group_2]
    g3_names = names[group_3]

    return g1_names, g2_names, g3_names


def sort_images_by_values(values_arr, names):
    opposite_order_values = 1 - values_arr
    sorted_idx = np.argsort(opposite_order_values)
    sorted_values = values_arr[sorted_idx]
    sorted_names = names[sorted_idx]

    new_dict = {}
    for i, data in enumerate(zip(sorted_names, sorted_values)):
        name, val = data
        name = str(name)
        val = str(val)
        new_dict[i] = {"name": name, "value": val}

    return new_dict


def load_eval_yaml(path, num_classes):
    """
    load yaml file that is output of evaluation and process it
    :param path: path to yaml file
    :param num_classes: number of classes, including background
    :return: m: array of mean iuo per image
            array of iou per class per image
            name: array of images names respectfully
    """
    class_arr = []

    for c in range(num_classes):
        class_arr.append([])

    with open(path) as yaml_file:

        yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

        names = []
        m_iou = []
        for image, data in yaml_data.items():
            names.append(image)
            mean_iou = data["mean_iou"]
            m_iou.append(mean_iou)
            for c, arr in enumerate(class_arr):
                cat = "class_" + str(c) + "_iou"
                if cat in data.keys():
                    c_data = data[cat]
                    if c_data != "not in image" and c_data != "not in prediction":
                        c_data = float(c_data)
                        if not np.isnan(c_data):
                            arr.append(c_data)

        for c, arr in enumerate(class_arr):
            class_arr[c] = np.array(arr)

        m = np.array(m_iou)
        names = np.array(names)

        return m, class_arr, names


def display_image(
    image_path,
    results_images_path,
    labels_dir="labels",
    predictions_dir="predictions",
    weights_dir="weights",
    output_dir=None,
    class_id=None,
):
    image = imread(image_path)

    image_name = Path(image_path).stem
    new_name = image_name + ".png"

    label = imread(os.path.join(results_images_path, labels_dir, new_name))
    pred = imread(os.path.join(results_images_path, predictions_dir, new_name))
    weights = imread(os.path.join(results_images_path, weights_dir, new_name))

    # if the class id is given, the class gets the value 1 and background 0
    # converts the label and prediction to RGB images (3 channels)
    if class_id != None:
        new_label = np.zeros_like(label)
        new_label = np.dstack((new_label, new_label, new_label))
        new_label[label == class_id] = LUT[127]

        new_pred = np.zeros_like(pred)
        new_pred = np.dstack((new_pred, new_pred, new_pred))
        new_pred[pred == class_id] = LUT[127]

    else:
        new_label = LUT[label]
        new_pred = LUT[pred]

    # resize image according to the model resize
    idx = np.where(weights)
    min_x, max_x = np.min(idx[0]), np.max(idx[0])
    min_y, max_y = np.min(idx[1]), np.max(idx[1])
    len_x = max_x - min_x
    len_y = max_y - min_y
    new_shape = (len_y, len_x)
    image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)

    # creates new image according to weights
    new_image = np.zeros_like(new_pred)
    new_image[min_x:max_x, min_y:max_y] = image

    # create a new image combining the mask and original image
    masked_label = cv2.addWeighted(new_image, 0.3, new_label, 0.7, 0)
    masked_pred = cv2.addWeighted(new_image, 0.3, new_pred, 0.7, 0)

    # background remains the same color
    masked_label[label == 0] = new_image[label == 0]
    masked_pred[pred == 0] = new_image[pred == 0]

    # creates figure with the 6 images in original size
    dpi = mpl.rcParams["figure.dpi"]
    width, height = pred.shape[0], pred.shape[1]
    figSize = width * 3 / float(dpi), height * 2 / float(dpi)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figSize, sharex=True, sharey=True)

    title = "image: ", str(image_name)
    fig.suptitle(title, fontsize=14)

    ax[0, 0].set_title("original image")
    ax[0, 0].imshow(image)
    ax[0, 1].set_title("original label")
    ax[0, 1].imshow(new_label)
    ax[0, 2].set_title("predicted label")
    ax[0, 2].imshow(new_pred)
    ax[1, 0].set_title("weights")
    ax[1, 0].imshow(weights)
    ax[1, 1].set_title("original label overlay")
    ax[1, 1].imshow(masked_label)
    ax[1, 2].set_title("predicted label overlay")
    ax[1, 2].imshow(masked_pred)

    # saves figures
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, image_name))

    plt.show()


def display_images_list(images_list, im_path, model_output_path, output_dir=None, class_id=None):
    for image_name in images_list:
        image_path = os.path.join(im_path, image_name)
        display_image(image_path, model_output_path, output_dir=output_dir, class_id=class_id)


def display_results(
    sorted_images_dict,
    im_path,
    model_output_path,
    best_percent=1,
    worst_percent=1,
    num_best=None,
    num_worst=None,
    save_plots=False,
    class_id=None,
    visualize_name="",
):
    # calculates number of best and worst images according to percent given
    num_images = len(sorted_images_dict)
    if not num_best:
        num_best = (num_images * best_percent) // 100
    if not num_worst:
        num_worst = (num_images * worst_percent) // 100

    # creates lists of best and worst images
    best_list = []
    worst_list = []

    for i in range(num_best):
        best_list.append(sorted_images_dict[i]["name"])

    for i in range(num_worst):
        worst_idx = num_images - i - 1
        worst_list.append(sorted_images_dict[worst_idx]["name"])

    # set path for saving images
    best_images_save_path = worst_images_save_path = None
    if save_plots:
        best_images_save_path = os.path.join(
            model_output_path, "visualize%s/best_images" % visualize_name
        )
        worst_images_save_path = os.path.join(
            model_output_path, "visualize%s/worst_images" % visualize_name
        )

        if class_id:
            best_images_save_path += "_class_%s" % class_id
            worst_images_save_path += "_class_%s" % class_id
        else:
            best_images_save_path += "_mean_iou"
            worst_images_save_path += "_mean_iou"

    # displays images
    print("best images:")
    print("")
    display_images_list(
        best_list, im_path, model_output_path, output_dir=best_images_save_path, class_id=class_id
    )
    print("worst images:")
    print("")
    display_images_list(
        worst_list, im_path, model_output_path, output_dir=worst_images_save_path, class_id=class_id
    )


def get_range(sorted_list, num_images, max_idx=-1):
    if max_idx == -1:
        min_idx = len(sorted_dict) - num_images
    else:
        max_idx += 1
        min_idx = max(max_idx - num_images, 0)
    new_list = sorted_list[min_idx:max_idx]
    return new_list

def fix_path_func_all_datasets(image_path):
    image_name = Path(image_path).stem
    labels_path = ""

    if "/cnvrg" in image_path:
        image_path = image_path.replace("/cnvrg", "")
    if "imaterialist" in image_path:
        image_path = image_path.replace("images", "train/images")
        labels_path = "/data/imaterialist-fashion/train/TopBottomDressJacket"
    elif "mhp" in image_path:
        labels_path = "/data/mhp/TopBottomJacketDress"
    elif "modanet" in image_path:
        labels_path = "/data/modanet_cnvrg-1/TopBottomJacketDress"

    label_path = os.path.join(labels_path, image_name + ".png")
    return image_path, label_path


def create_weights_dict(sorted_dict, fix_path_f=None, save_path=None):
    weights_dict = {}

    for i in range(len(sorted_dict)):
        path = sorted_dict[i]["name"]
        if fix_path_f is not None:
            path = fix_path_f(path)[0]
        weight = sorted_dict[i]["value"]
        weights_dict[path] = weight

    if save_path is not None:
        with open(save_path, "a+") as weights_yaml:
            yaml.dump(weights_dict, weights_yaml)

    return weights_dict


def display_original_label(images_list, number_list, save_path, fix_path_func=fix_path_func_all_datasets):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    i = 0
    for image_path in tqdm(images_list):

        image_path = image_path.replace("/cnvrg", "")
        image_num = number_list[i]

        image_path, label_path = fix_path_func(image_path)

        image = imread(image_path)
        label = imread(label_path)

        new_label = np.zeros_like(label)
        new_label[label != 0] = 255

        new_label = LUT[new_label]
        new_label = new_label.astype(image.dtype)

        masked_label = cv2.addWeighted(image, 0.3, new_label, 0.7, 0)
        masked_label[label == 0] = image[label == 0]

        # creates figure with the 6 images in original size
        dpi = mpl.rcParams["figure.dpi"]
        height, width = image.shape[0], image.shape[1]
        figSize = width * 3 / float(dpi), height / float(dpi)

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figSize, sharex=True, sharey=True)

        title = "image path: " + image_path + " , image number: " + str(image_num)
        fig.suptitle(title, fontsize=14)
        ax[0].set_title("image")
        ax[0].imshow(image)
        ax[1].set_title("mask")
        ax[1].imshow(new_label)
        ax[2].set_title("masked image")
        ax[2].imshow(masked_label)
        plt.savefig(os.path.join(save_path, str(image_num) + ".png"))
        i += 1


def top_of_under_value(sorted_dict, val=0.5, number=200):
    pathes = []
    vals = []
    numbers = []
    num_im = len(sorted_dict)
    for i in range(num_im):
        value = float(sorted_dict[i]['value'])
        if value < val:
            vals.append(value)
            pathes.append(sorted_dict[i]['name'])
            numbers.append(i)

    vals = np.asarray(vals)
    pathes = np.asarray(pathes)
    numbers = np.asarray(numbers)

    vals_top = vals[0:number]
    paths_top = pathes[0:number]
    numbers_top = numbers[0:number]

    display_original_label(paths_top, numbers_top, "/cnvrg/output/vis/" + str(val) + "_" + str(number))


if __name__ == "__main__":
    # yaml_path = "output/test_set_results.yaml"
    yaml_name = "mhp"
    yaml_path = os.path.join("/cnvrg/output/", yaml_name + "_results.yaml")
    # yaml_path = "/usrs/shira/Downloads/test_set_results.yaml"

    # get evaluation results from yaml file
    m, c_arr, im_names = load_eval_yaml(yaml_path, 3)

    # mean iou
    # sort the images according to mean iou
    sorted_dict = sort_images_by_values(m, im_names)

    # save mean iou sort
    new_yaml = os.path.join("/cnvrg/output/", yaml_name + "_sorted_images.yaml")
    with open(new_yaml, "a+") as n_yaml_file:
        yaml.dump(sorted_dict, n_yaml_file)

    # display results for mean iou
    im_path = "/data/mhp/images"
    model_output_path = "/cnvrg/output/"
    display_results(sorted_dict, im_path, model_output_path, save_plots=True)

    # iou per class
    for class_id, c in enumerate(c_arr):
        # sort the images according to iou for the class c
        c_sorted_dict = sort_images_by_values(c, im_names)
        # display results for class c
        display_results(
            c_sorted_dict, im_path, model_output_path, save_plots=True, class_id=class_id
        )
