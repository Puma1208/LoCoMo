import os
import pandas as pd

import pyrealsense2 as rs
import numpy as np
import cv2


def main():
        
    object = ''

    json_directory = 'Json'
    # To use for the dataframes 
    joints_col = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']

    json = os.listdir(json_directory)
    sorted_json = sorted(json, key=lambda x: int(x.split("_")[2].split('.')[0]))

    for filename in sorted_json:
        f = os.path.join(json_directory, filename)
        if os.path.isfile(f):
            df_joints = pd.read_json(f)
            string_position = 'joint '
            for joint in joints_col:
                string_position = string_position + str(df_joints[joint].values[0]) + ' '
                # print('In file ', f, ' pos ', string_position)

            # Can perform the robot motion and image capturing here
            # First set the right positions from Guillaume's code
            # Then capture all the data we need
            shot = filename.split("_")[2].split(".")[0]
            print(filename, ' ', shot)
            # capture_images(object, int(shot))


def capture_images(object, shot):
    path = os.getcwd()

    # Path to store the object
    path_dir = os.path.join(path, object)
    print(path_dir)
    if(not os.path.exists(path_dir)):
        print('Path doesnt exist - creating a new path')
        os.mkdir(path_dir) 

    ###     1. Store the different kinds of images
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Can remove background i.e. anything that is dist_background meters away
    clipping_distance_in_meters = 1.5
    clipping_distance = clipping_distance_in_meters/depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)

    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())



    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 153
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

    # Render images
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
    images = np.hstack((bg_removed, depth_colormap))

    cv2.namedWindow('Align example', cv2.WINDOW_AUTOSIZE)

    image1 = path_dir + '/' + str(shot) + '_'+ str(object) + '_Color.jpg'
    image2 = path_dir + '/' + str(shot) + '_'+ str(object) + '_Depth.jpg'
    image3 = path_dir + '/' + str(shot) + '_'+ str(object) + '_bg_removed.jpg'
    image4 = path_dir + '/' + str(shot) + '_'+ str(object) + '_ColorDepth.jpg'
    image5 = path_dir + '/' + str(shot) + '_'+ str(object) + '_DepthColorMap.jpg'

    # Save the images
    cv2.imwrite(image1, color_image)
    cv2.imwrite(image2, depth_image)
    cv2.imwrite(image3, images)
    cv2.imwrite(image4, bg_removed)
    cv2.imwrite(image5, depth_colormap)

    key = cv2.waitKey(1)
    cv2.destroyAllWindows()

    pipeline.stop()

    ###     2. Store the point cloud

    # Declare RealSense pipeline, encapsulating the actual device and sensors
    pipe = rs.pipeline()
    config = rs.config()
    # Enable depth stream
    config.enable_stream(rs.stream.depth)

    # Start streaming with chosen configuration
    pipe.start(config)

    # We'll use the colorizer to generate texture for our PLY
    # (alternatively, texture can be obtained from color or infrared stream)
    colorizer = rs.colorizer()

    try:
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()
        colorized = colorizer.process(frames)

        # Create save_to_ply object
        ply_file_name = path_dir + '/' + str(shot) + '_'+ str(object) + '.ply'

        ply = rs.save_to_ply(ply_file_name)

        # Set options to the desired values
        # In this example we'll generate a textual PLY with normals (mesh is already created by default)
        ply.set_option(rs.save_to_ply.option_ply_binary, False)
        ply.set_option(rs.save_to_ply.option_ply_normals, True)

        print("Saving to ply to " + ply_file_name)
        # Apply the processing block to the frameset which contains the depth frame and the texture
        ply.process(colorized)
        print("Done")
    finally:
        
        pipe.stop()

main()
