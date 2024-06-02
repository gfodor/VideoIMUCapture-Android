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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Creates a dmvio compatible input')
    parser.add_argument('video_path', type=str, help='Path to video and protobuf')
    parser.add_argument('--result-dir', type=str, help='Path to result folder, default same as video file', default = None)
    parser.add_argument('--target-fps', type=float, help='Target FPS for the output', default = 20.0)
    parser.add_argument('--skip-iso-normalize', action='store_true', help='Normalize exposure time by ISO', default = False)
    parser.add_argument('--iso-factor', type=int, help='Normalize exposure time by ISO value', default = 400)

    args = parser.parse_args()

    for root, dirnames, filenames in os.walk(args.video_path):
        if not 'video_meta.pb3' in filenames:
            continue

        sub_path = osp.relpath(root,start=args.video_path)
        result_dir = osp.join(args.result_dir, sub_path) if args.result_dir else osp.join(root, 'dmvio')
        frame_duration = 1.0 / args.target_fps - 0.01
        image_dir = osp.join(result_dir, 'images')

        _makedir(result_dir)
        _makedir(image_dir)

        proto_path = osp.join(args.video_path, 'video_meta.pb3')
        with open(proto_path,'rb') as f:
            proto = VideoCaptureData.FromString(f.read())

        max_imu_ns = 0
        imu_raw_path = osp.join(result_dir, 'imu_raw.txt')
        imu_path = osp.join(result_dir, 'imu.txt')
        max_gyr_x = 0
        min_gyr_x = 99999
        max_gyr_y = 0
        min_gyr_y = 99999
        max_gyr_z = 0
        min_gyr_z = 99999
        max_acc_x = 0
        min_acc_x = 99999
        max_acc_y = 0
        min_acc_y = 99999
        max_acc_z = 0
        min_acc_z = 99999

        with open(imu_raw_path, 'w') as f:
            for imu_frame in proto.imu:
                gyro_drift = imu_frame.gyro_drift
                accel_bias = imu_frame.accel_bias
                gyro_x = imu_frame.gyro[0]
                gyro_y = imu_frame.gyro[1]
                gyro_z = imu_frame.gyro[2]
                accel_x = imu_frame.accel[0]
                accel_y = imu_frame.accel[1]
                accel_z = imu_frame.accel[2]
                max_imu_ns = max(max_imu_ns, imu_frame.time_ns)
                max_gyr_x = max(max_gyr_x, imu_frame.gyro[0])
                min_gyr_x = min(min_gyr_x, imu_frame.gyro[0])
                max_gyr_y = max(max_gyr_x, imu_frame.gyro[1])
                min_gyr_y = min(min_gyr_x, imu_frame.gyro[1])
                max_gyr_z = max(max_gyr_x, imu_frame.gyro[2])
                min_gyr_z = min(min_gyr_x, imu_frame.gyro[2])
                max_acc_x = max(max_acc_x, imu_frame.accel[0])
                min_acc_x = min(min_acc_x, imu_frame.accel[0])
                max_acc_y = max(max_acc_x, imu_frame.accel[1])
                min_acc_y = min(min_acc_x, imu_frame.accel[1])
                max_acc_z = max(max_acc_x, imu_frame.accel[2])
                min_acc_z = min(min_acc_x, imu_frame.accel[2])
                f.write("{} {} {} {} {} {} {}\n".format(imu_frame.time_ns, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z))

        print("gyr x", max_gyr_x, min_gyr_x)
        print("gyr y", max_gyr_y, min_gyr_y)
        print("gyr z", max_gyr_z, min_gyr_z)
        print("acc x", max_acc_x, min_acc_x)
        print("acc y", max_acc_y, min_acc_y)
        print("acc z", max_acc_z, min_acc_z)
        times_path = osp.join(result_dir, 'times.txt')

        w = proto.camera_meta.resolution.width
        h = proto.camera_meta.resolution.height

        last_frame_time = 0

        with open(times_path, 'w') as f:
            for frame_data in proto.video_meta:
                if frame_data.yuv_plane == b'':
                    continue

                if frame_data.time_ns > max_imu_ns:
                    raise ValueError("Frame time exceeds IMU time")

                if last_frame_time != 0 and frame_data.time_ns - last_frame_time < 1e9 * frame_duration:
                    continue

                last_frame_time = frame_data.time_ns
                exposure_time_ms = frame_data.exposure_time_ns / 1e6

                iso_factor = 1

                if not args.skip_iso_normalize:
                    iso_factor = frame_data.iso / args.iso_factor # Incorporate the iso into the exposure time

                #print("Frame time: {}s, Exposure time: {}ms, ISO: {}, ISO factor: {} New exposures: {}".format(frame_data.time_ns/1e9, exposure_time_ms, frame_data.iso, iso_factor, exposure_time_ms * iso_factor))
                #f.write("{} {} {}\n".format(frame_data.time_ns, frame_data.time_ns/ 1e9, exposure_time_ms * iso_factor * 0.25))
                f.write("{} {} {}\n".format(frame_data.time_ns, frame_data.time_ns/ 1e9, exposure_time_ms * iso_factor))

                yuv_plane = frame_data.yuv_plane

                # YUV plane is a series of bytes YUV 420 888, write these values out as is to a greyscale file
                yuv = np.frombuffer(yuv_plane, dtype=np.uint8)
                yuv = yuv.reshape(w, h)
                yuv = np.transpose(yuv)
                yuv = np.fliplr(yuv)

                # Write out the Y plane
                cv2.imwrite(osp.join(image_dir,'{:06d}.png'.format(frame_data.time_ns)), yuv[:h,:], [cv2.IMWRITE_PNG_COMPRESSION, 0])

        interpolate_imu_file(imu_raw_path, times_path, osp.join(result_dir, 'imu.txt'))

        os.remove(imu_raw_path)
