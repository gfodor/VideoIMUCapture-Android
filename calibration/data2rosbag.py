#!/usr/bin/python
import argparse
from recording_pb2 import VideoCaptureData
import os.path as osp
import os
import cv2
import csv
import rosbag
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge
from pyquaternion import Quaternion
import numpy as np
import shutil
import yaml
from utils import OpenCVDumper
import time

class VideoFinishedException(Exception):
    pass

bridge = CvBridge()
NSECS_IN_SEC=int(1e9)

def convert_to_bag(proto, video_path, result_path, subsample=1, compress_img=False, compress_bag=False, resize = [], raw_imu = False, imu_only = False):
    #Init rosbag
    # bz2 is better compression but lz4 is 3 times faster
    resolution = None
    img_topic = "/cam0/image_raw/compressed" if compress_img else "/cam0/image_raw"

    try:
        bag = rosbag.Bag(result_path, 'w', compression='lz4' if compress_bag else 'none')

        if not imu_only:
          # Open video stream
          try:
              cap = cv2.VideoCapture(video_path)

              # Generate images from video and frame data
              for frame_data in proto.video_meta:

                  # Read video frames until we find correct number
                  while  True:
                      video_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                      ret, frame = cap.read()
                      if not ret:
                          raise VideoFinishedException()

                      if video_frame_idx==frame_data.frame_number and (video_frame_idx % subsample) == 0:
                          # Correct frame and subsample index
                          rosimg, timestamp, resolution = img_to_rosimg(frame,
                                                                        frame_data.time_ns,
                                                                        compress=compress_img,
                                                                        resize = resize)
                          bag.write(img_topic, rosimg, timestamp)

                          # Go to next data frame
                          break

                      elif video_frame_idx==frame_data.frame_number:
                          #Skipping subsample
                          break

                      elif video_frame_idx < frame_data.frame_number:
                          print('skipping frame {}, missing data'.format(video_frame_idx))

                      else:
                          raise NotImplementedError('Missing video frame idx is not supported and not expected. \
                                                    Video frame idx {} > frame_data.frame_number {}'.format(video_frame_idx, frame_data.frame_number))

          except VideoFinishedException:
              # Nothing to worry about, video stream ended.
              pass

          finally:
              cap.release()

        c = 0

        # Now IMU
        for imu_frame in proto.imu:
            if not raw_imu:
                gyro_drift = getattr(imu_frame, 'gyro_drift', np.zeros(3))
                accel_bias = getattr(imu_frame, 'accel_bias', np.zeros(3))
            else:
                gyro_drift = accel_bias = np.zeros(3)
            rosimu, timestamp = imu_to_rosimu(imu_frame.time_ns, imu_frame.gyro, gyro_drift, imu_frame.accel, accel_bias)
            bag.write("/imu0", rosimu, timestamp)

            c += 1

        print("wrote ", c, " IMU samples")

    finally:
        bag.close()

    return resolution



def img_to_rosimg(img, timestamp_nsecs, compress = True, resize = []):
    timestamp = rospy.Time(secs=timestamp_nsecs//NSECS_IN_SEC,
                           nsecs=timestamp_nsecs%NSECS_IN_SEC)

    gray_img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if resize:
        gray_img = cv2.resize(gray_img, tuple(resize), cv2.INTER_AREA)
        assert gray_img.shape[0] == resize[1]

    if compress:
        rosimage = bridge.cv2_to_compressed_imgmsg(gray_img, dst_format='png')
    else:
        rosimage = bridge.cv2_to_imgmsg(gray_img, encoding="mono8")
    rosimage.header.stamp = timestamp

    return rosimage, timestamp, (gray_img.shape[1], gray_img.shape[0])

def imu_to_rosimu(timestamp_nsecs, omega, omega_drift, alpha, alpha_bias):
    timestamp = rospy.Time(secs=timestamp_nsecs//NSECS_IN_SEC,
                           nsecs=timestamp_nsecs%NSECS_IN_SEC)

    rosimu = Imu()
    rosimu.header.stamp = timestamp
    gyro_x = omega[0]# - omega_drift[0]
    gyro_y = omega[1]# - omega_drift[1]
    gyro_z = omega[2]# - omega_drift[2]
    accel_x = alpha[0]# - alpha_bias[0]
    accel_y = alpha[1]# - alpha_bias[1]
    accel_z = alpha[2]# - alpha_bias[2]

    #gyro_x = 1.0925741 * gyro_x
    #gyro_y = -0.02255 * gyro_x + 1.0319 * gyro_y + 0 * gyro_z
    #gyro_z = 0.02092 * gyro_x - 0.00527 * gyro_z + 1.01599 * gyro_z
    #gyro_x = 1.04943909 * gyro_x
    #gyro_y = 0.00043756 * gyro_x + 1.033398047 * gyro_y + 0 * gyro_z
    #gyro_z = 0.01105462 * gyro_x - 0.00202769 * gyro_z + 0.99306513 * gyro_z

    #accel_x = accel_x + 0.03
    #accel_y = accel_y - 0.22
    #accel_z = accel_z - 0.15

    rosimu.angular_velocity.x = gyro_x
    rosimu.angular_velocity.y = gyro_y
    rosimu.angular_velocity.z = gyro_z
    rosimu.linear_acceleration.x = accel_x
    rosimu.linear_acceleration.y = accel_y
    rosimu.linear_acceleration.z = accel_z

    return rosimu, timestamp

def adjust_calibration(input_yaml_path, output_yaml_path, resolution):
    with open(input_yaml_path,'r') as f:
        calib = yaml.safe_load(f)

    cam0 = calib['cam0']
    if cam0['resolution'][0] != resolution[0]:
        sx = float(resolution[0])/cam0['resolution'][0]
        cam0['intrinsics'][0] *= sx
        cam0['intrinsics'][2] *= sx
        cam0['resolution'][0] = resolution[0]

    if cam0['resolution'][1] != resolution[1]:
        sy = float(resolution[1])/cam0['resolution'][1]
        cam0['intrinsics'][1] *= sy
        cam0['intrinsics'][3] *= sy
        cam0['resolution'][1] = resolution[1]

    with open(output_yaml_path,'w') as f:
        yaml.dump(calib, f, Dumper=OpenCVDumper)


def _makedir(new_dir):
    try:
        os.mkdir(new_dir)
    except OSError:
        pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert video and proto to rosbag')
    parser.add_argument('data_dir', type=str, help='Path to folder with video_recording.mp4 and video_meta.pb3 or root-folder containing multiple datasets')
    parser.add_argument('--result-dir', type=str, help='Path to result folder, default same as proto', default = None)
    parser.add_argument('--subsample', type=int, help='Take every n-th video frame', default = 1)
    parser.add_argument('--raw-image', action='store_true', help='Store raw images in rosbag')
    parser.add_argument('--resize', type=int, nargs = 2, default = [], help='Resize image to this <width height>')
    parser.add_argument('--raw-imu', action='store_true', help='Do not compensate for bias')
    parser.add_argument('--imu-only', action='store_true', help='Only store imu data')
    parser.add_argument('--calibration', type=str, help='YAML file with kalibr camera and IMU calibration to copy, will also adjust for difference in resolution.', default = None)

    args = parser.parse_args()

    for root, dirnames, filenames in os.walk(args.data_dir):
        if not 'video_meta.pb3' in filenames:
            continue

        sub_path = '' if osp.samefile(root, args.data_dir) else osp.relpath(root,start=args.data_dir)
        result_dir = osp.join(args.result_dir, sub_path) if args.result_dir else osp.join(root, 'rosbag')
        _makedir(result_dir)

        # Read proto
        proto_path = osp.join(root, 'video_meta.pb3')
        with open(proto_path,'rb') as f:
            proto = VideoCaptureData.FromString(f.read())

        video_path = osp.join(root, 'video_recording.mp4')
        bag_path = osp.join(result_dir, 'data.bag')
        resolution = convert_to_bag(proto,
                                    video_path,
                                    bag_path,
                                    subsample = args.subsample,
                                    compress_img = not args.raw_image,
                                    resize = args.resize,
                                    raw_imu = args.raw_imu,
                                    imu_only = args.imu_only)

        if args.calibration:
            out_path = osp.join(result_dir, 'calibration.yaml')
            adjust_calibration(args.calibration, out_path, resolution)
