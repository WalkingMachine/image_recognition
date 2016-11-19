import cv2
import os
import datetime


def write_annotated(dir_path, image, label, verified=False):

    if dir_path is None:
        return False

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    annotated_dir = dir_path + "/annotated"
    if not os.path.exists(annotated_dir):
        os.makedirs(annotated_dir)


    annotated_verified_unverified_dir = annotated_dir + "/verified" if verified else annotated_dir + "/unverified"
    if not os.path.exists(annotated_verified_unverified_dir):
        os.makedirs(annotated_verified_unverified_dir)

    label_dir = annotated_verified_unverified_dir + "/" + label
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    filename = "%s/%s.jpg" % (label_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f"))
    cv2.imwrite(filename, image)

    return True


def write_raw(dir_path, image):

    if dir_path is None:
        return False


    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    raw_dir = dir_path + "/raw"
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    filename = "%s/%s.jpg" % (raw_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f"))
    cv2.imwrite(filename, image)

    return True
