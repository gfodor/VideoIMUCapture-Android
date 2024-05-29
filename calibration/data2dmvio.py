#!/usr/bin/python
import argparse
import os.path as osp
from recording_pb2 import VideoCaptureData
import os
import cv2
import sys
from data2rosbag import _makedir, adjust_calibration
from interpolate_imu_file import interpolate_imu_file
import numpy as np

def convert_to_images(video_path, result_path, image_path, proto):
    w = proto.camera_meta.resolution.width
    h = proto.camera_meta.resolution.height

    for frame_data in proto.video_meta:
        if frame_data.yuv_plane == b'':
            continue

        yuv_plane = frame_data.yuv_plane

        # YUV plane is a series of bytes YUV 420 888, write these values out as is to a greyscale file
        yuv = np.frombuffer(yuv_plane, dtype=np.uint8)
        yuv = yuv.reshape(w, h)
        yuv = np.transpose(yuv)

        # Write out the Y plane
        cv2.imwrite(osp.join(image_path,'{:06d}.png'.format(frame_data.time_ns)), yuv[:h,:])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Creates a dmvio compatible input')
    parser.add_argument('video_path', type=str, help='Path to video and protobuf')
    parser.add_argument('--result-dir', type=str, help='Path to result folder, default same as video file', default = None)

    args = parser.parse_args()

    for root, dirnames, filenames in os.walk(args.video_path):
        if not 'video_meta.pb3' in filenames:
            continue

        sub_path = osp.relpath(root,start=args.video_path)
        result_dir = osp.join(args.result_dir, sub_path) if args.result_dir else osp.join(root, 'dmvio')
        image_dir = osp.join(result_dir, 'images')

        _makedir(result_dir)
        _makedir(image_dir)

        proto_path = osp.join(args.video_path, 'video_meta.pb3')
        with open(proto_path,'rb') as f:
            proto = VideoCaptureData.FromString(f.read())

        video_path = osp.join(root, 'video_recording.mp4')
        convert_to_images(video_path, result_dir, image_dir, proto)

        times_path = osp.join(result_dir, 'times.txt')
        imu_raw_path = osp.join(result_dir, 'imu_raw.txt')
        imu_path = osp.join(result_dir, 'imu.txt')

        with open(times_path, 'w') as f:
            for frame_data in proto.video_meta:
                f.write("{} {} {}\n".format(frame_data.time_ns, frame_data.time_ns, frame_data.exposure_time_ns/1e6))

        with open(imu_raw_path, 'w') as f:
            for imu_frame in proto.imu:
                gyro_drift = imu_frame.gyro_drift
                accel_bias = imu_frame.accel_bias
                gyro_x = imu_frame.gyro[0] - gyro_drift[0]
                gyro_y = imu_frame.gyro[1] - gyro_drift[1]
                gyro_z = imu_frame.gyro[2] - gyro_drift[2]
                accel_x = imu_frame.accel[0] - accel_bias[0]
                accel_y = imu_frame.accel[1] - accel_bias[1]
                accel_z = imu_frame.accel[2] - accel_bias[2]
                f.write("{} {} {} {} {} {} {}\n".format(imu_frame.time_ns, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z))

        interpolate_imu_file(imu_raw_path, times_path, osp.join(result_dir, 'imu.txt'))

        os.remove(imu_raw_path)
