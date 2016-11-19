#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge, CvBridgeError

from image_recognition_msgs.srv import Recognize, Annotate
from image_recognition_msgs.msg import Recognition, CategoricalDistribution, CategoryProbability
from sensor_msgs.msg import RegionOfInterest
from std_srvs.srv import Empty

import cv2
import os
from datetime import datetime
import sys

from image_recognition_util import Image_writer

from openface_ros.face_recognizer import FaceRecognizer, RecognizedFace


class OpenFaceRos:
    def __init__(self, align_path, net_path, save_images_folder):

        self._bridge = CvBridge()
        self._annotate_srv = rospy.Service('annotate',Annotate, self._annotate_srv)
        self._recognize_srv = rospy.Service('recognize', Recognize, self._recognize_srv)
        self._clear_srv = rospy.Service('clear', Empty, self._clear_srv)

        self._face_recognizer = RecognizedFace(align_path, net_path)
        self._save_images_folder = save_images_folder

        rospy.loginfo("OpenfaceROS node begin")
        rospy.loginfo("dlib_align_path=%s", align_path)
        rospy.loginfo("save_image_folder=%s", save_images_folder)

    def _annotate_srv(self, req):

        try:
            bgr_image = self._bridge.imgmsg_to_cv2((req.image, "bgr8"))

        except CvBridgeError as e:
            raise Exception("Could not convert to opencv image: %s" % str(e))

        for annotation in req.annotations:
            roi_image = bgr_image[annotation.roi.y_offset:annotation.roi.y_offset+annotation.roi.height,
                                  annotation.roi.x_offset:annotation.roi.x_offset+annotation.roi.width]

            if self._save_images_folder:
                Image_writer.write_annotated(self._save_images_folder, roi_image, annotation.label, True)

            try:
                self._face_recognizer.train(roi_image, annotation.label)

            except Exception as e:
                raise Exception("Could not get representation of face image: % s" % str(e))

            rospy.loginfo("Succesfully learned face of '%s'" % annotation.label)

        return  {}

    def _clear_srv(self, req):

        self._face_recognizer.clear_trained_faces()

        return {}

    def _recognize_srv(self, req):

        try:
            bgr_image = self._bridge.imgmsg_to_cv2(self._save_images_folder, "bgr8")

        except CvBridgeError as e:
            raise Exception("Could not convert to opencv image: %s" % str(e))

        if self._save_images_folder:
            Image_writer.write_raw(self._save_images_folder, bgr_image)

        face_recognitions = self._face_recognizer.recognize(bgr_image)

        recognitions = []

        rospy.loginfo("Face recognitions: %s", face_recognitions)
        for face_recognition in face_recognitions:

            if self._save_images_folder:
                label = face_recognition.l2_distances[0].label if len(
                    face_recognition.l2_distances) > 0 else "face_unknown"
                roi_image = bgr_image[
                            face_recognition.roi.y_offset:face_recognition.roi.y_offset + face_recognition.roi.height,
                            face_recognition.roi.x_offset:face_recognition.roi.x_offset + face_recognition.roi.width]
                Image_writer.write_annotated(self._save_images_folder, roi_image, label, False)

            recognitions.append(Recognition(
                categorical_distribution=CategoricalDistribution(
                    unknown_probability=0.0,
                    probabilities=[CategoryProbability(label=l2.label, probability=1.0 / l2.distance)
                                   for l2 in face_recognition.l2_distances]
                ),
                roi=RegionOfInterest(
                    x_offset=face_recognition.roi.x_offset,
                    y_offset=face_recognition.roi.y_offset,
                    width=face_recognition.roi.width,
                    height=face_recognition.roi.height
                )
            ))
        return {"recognitions": recognitions}

if __name__ == "__main__":

    rospy.init_node("face_recognition")

    try:
        dlib_shape_predictor_path = os.path.expanduser(
            rospy.get_param("~align_path", "~/openface/models/dlib/shape_predictor_68_face_landmarks.dat"))
        openface_neural_network_path = os.path.expanduser(
            rospy.get_param("~net_path", "~/openface/models/openface/nn4.small2.v1.t7"))
        save_images = rospy.get_param("~save_images", True)

        save_images_folder = None
        if save_images:
            save_images_folder = os.path.expanduser(rospy.get_param("~save_images_folder", "/tmp/openface"))
    except KeyError as e:
        rospy.logerr("Parameter %s not found" % e)
        sys.exit(1)

    openface_ros = OpenFaceRos(dlib_shape_predictor_path,
                               openface_neural_network_path,
                               save_images_folder)
    rospy.spin()