from imageio import imread
import os
import ntpath
import numpy as np
from sklearn.metrics import confusion_matrix
import yaml
from pathlib import Path
from tqdm import tqdm
import glob

# save_path = "/cnvrg/output"

lut = np.zeros(256, dtype=np.int)
lut[127] = 1
lut[255] = 2

w_lut = np.zeros(256, dtype=np.int)
w_lut[255] = 1
w_lut[1] = 1


def get_image_list(images_dir, image_list_path=None):
    if image_list_path:
        return [os.path.join(images_dir, filename) for filename in read_lines(image_list_path)]

    types = ("png", "jpg", "JPEG", "jpeg")
    image_list = []
    for file_type in types:
        image_list.extend(glob.glob(os.path.join(images_dir, "*." + file_type)))

    return sorted(image_list)


def eval_from_dir(save_path, save_name, num_classes=3):
    log_file_path = os.path.join(save_path, save_name + "_log.txt")
    logfile = open(log_file_path, 'a+')

    yaml_file_path = os.path.join(save_path, save_name + ".yaml")

    if os.path.exists(yaml_file_path):
        yaml_file = open(yaml_file_path, 'r')
        yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        yaml_file.close()
    else:
        yaml_data = None

    with open(yaml_file_path, 'a+') as yaml_file:
        predictions_path = os.path.join(save_path, 'predictions')

        predictions_list = get_image_list(predictions_path)

        for name in tqdm(predictions_list):

            base_name = Path(name).stem
            name = base_name + ".png"
            image_name = base_name + ".jpg"

            if yaml_data and image_name in yaml_data:
                text = "%s already exists\n" % image_name
                logfile.write(text)
            #                 print(text)
            else:
                logfile.writelines(["image name:%s" % image_name, "\n"])

                image_data_dict = {}

                image_path = "/cnvrg/mhp_train_ready/images/" + image_name
                image_data_dict["path"] = image_path

                pred = imread(os.path.join(predictions_path, name)).ravel()
                label = imread(os.path.join(save_path, 'labels', name)).ravel()
                weights = imread(os.path.join(save_path, 'weights', name)).ravel()

                pred = lut[pred]
                label = lut[label]
                weights = w_lut[weights]

                all_iou = []
                classes = np.arange(num_classes)

                for c in classes:
                    if c not in pred:
                        logfile.writelines(
                            ["class %s" % str(c), " not in prediction ", "\n"]
                        )
                        if c in label:
                            logfile.writelines(
                                ["class %s" % str(c), " in label ", "\n"]
                            )
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

                    lines = ["class " + str(c) + " iou: " + str(iou) + "\n",
                             "tn=" + str(tn_p) + "\n",
                             "fp=" + str(fp_p) + "\n",
                             "fn=" + str(fn_p) + "\n",
                             "tp=" + str(tp_p) + "\n"]
                    logfile.writelines(lines)

                    c_m_data = [float(tn_p), float(fp_p), float(fn_p), float(tp_p)]
                    image_data_dict["confusion_mat_" + str(c)] = c_m_data
                    image_data_dict["class_" + str(c) + "_iou"] = float(iou)

                # mean iou calc
                mean_iou = np.mean(all_iou)
                image_data_dict["mean_iou"] = float(mean_iou)
                lines = ["mean_iou: " + str(mean_iou) + "\n"]
                logfile.writelines(lines)

                # update yaml file with image info
                yaml_dict = {image_name: image_data_dict}
                yaml.dump(yaml_dict, yaml_file)

        yaml_file.close()
        logfile.close()
        return yaml_dict