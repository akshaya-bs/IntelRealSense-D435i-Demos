import tensorflow as tf
import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    def class_score_boxing():
        # Load the TensorFlow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile('/home/tiger/Downloads/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.compat.v1.import_graph_def(od_graph_def, name='')
            sess = tf.compat.v1.Session(graph=detection_graph)

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Output tensors are the detection boxes, scores, and classes
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Initialize the Intel RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)

        colors_hash = {}
        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                image_expanded = np.expand_dims(color_image, axis=0)

                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded}
                )

                H, W, _ = color_image.shape
                for idx in range(int(num[0])):
                    class_ = int(classes[0][idx])
                    score = scores[0][idx]
                    box = boxes[0][idx]
                    if score > 0.5:  # Detection threshold
                        if class_ not in colors_hash:
                            colors_hash[class_] = tuple(np.random.choice(range(256), size=3))

                        left = box[1] * W
                        top = box[0] * H
                        right = box[3] * W
                        bottom = box[2] * H

                        width = right - left
                        height = bottom - top
                        bbox = (int(left), int(top), int(width), int(height))
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                        r, g, b = colors_hash[class_]
                        cv2.rectangle(color_image, p1, p2, (int(r), int(g), int(b)), 2, 1)
                        #This part is the only thing different from function boxing
                        cv2.putText(color_image, f'Class: {class_} Score: {score:.2f}', (p1[0], p1[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(r), int(g), int(b)), 2)

                cv2.imshow('RealSense', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

    def boxing():

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile('/home/tiger/Downloads/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.compat.v1.import_graph_def(od_graph_def, name='')
            sess = tf.compat.v1.Session(graph=detection_graph)

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Output tensors are the detection boxes, scores, and classes
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Initialize the Intel RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)

        colors_hash = {}
        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                image_expanded = np.expand_dims(color_image, axis=0)

                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded}
                )

                H, W, _ = color_image.shape
                for idx in range(int(num[0])):
                    class_ = int(classes[0][idx])
                    score = scores[0][idx]
                    box = boxes[0][idx]
                    if class_ not in colors_hash:
                        colors_hash[class_] = tuple(np.random.choice(range(256), size=3))
                    if score > 0.5:  # Adjusted threshold for more detections
                        left = box[1] * W
                        top = box[0] * H
                        right = box[3] * W
                        bottom = box[2] * H

                        width = right - left
                        height = bottom - top
                        bbox = (int(left), int(top), int(width), int(height))
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        # draw box
                        r, g, b = colors_hash[class_]
                        cv2.rectangle(color_image, p1, p2, (int(r), int(g), int(b)), 2, 1)

                cv2.imshow('RealSense', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
    def height_of_human():
    
        W = 848
        H = 480
    
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    
    
        print("[INFO] start streaming...")
        pipeline.start(config)
    
        aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
        point_cloud = rs.pointcloud()
    
        print("[INFO] loading model...")
        PATH_TO_CKPT = r"frozen_inference_graph.pb"
        # download model from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#run-network-in-opencv
    
        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.compat.v1.import_graph_def(od_graph_def, name='')
            sess = tf.compat.v1.Session(graph=detection_graph)
    
        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # code source of tensorflow model loading: https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/
    
        while True:
            frames = pipeline.wait_for_frames()
            frames = aligned_stream.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            points = point_cloud.calculate(depth_frame)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz
    
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            scaled_size = (int(W), int(H))
            # expand image dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            image_expanded = np.expand_dims(color_image, axis=0)
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                      feed_dict={image_tensor: image_expanded})
    
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)
    
            print("[INFO] drawing bounding box on detected objects...")
            print("[INFO] each detected object has a unique color")
    
            for idx in range(int(num)):
                class_ = classes[idx]
                score = scores[idx]
                box = boxes[idx]
                print(" [DEBUG] class : ", class_, "idx : ", idx, "num : ", num)
    
                if score > 0.8 and class_ == 1: # 1 for human
                    left = box[1] * W
                    top = box[0] * H
                    right = box[3] * W
                    bottom = box[2] * H
    
                    width = right - left
                    height = bottom - top
                    bbox = (int(left), int(top), int(width), int(height))
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    # draw box
                    cv2.rectangle(color_image, p1, p2, (255,0,0), 2, 1)
    
                    # x,y,z of bounding box
                    obj_points = verts[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])].reshape(-1, 3)
                    zs = obj_points[:, 2]
    
                    z = np.median(zs)
    
                    ys = obj_points[:, 1]
                    ys = np.delete(ys, np.where(
                        (zs < z - 1) | (zs > z + 1)))  # take only y for close z to prevent including background
    
                    my = np.amin(ys, initial=1)
                    My = np.amax(ys, initial=-1)
    
                    height = (My - my)  # add next to rectangle print of height using cv library
                    height = float("{:.2f}".format(height))
                    print("[INFO] object height is: ", height, "[m]")
                    height_txt = str(height) + "[m]"
    
                    # Write some Text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (p1[0], p1[1] + 20)
                    fontScale = 1
                    fontColor = (255, 255, 255)
                    lineType = 2
                    cv2.putText(color_image, height_txt,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)
    
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(1)
    
        # Stop streaming
        pipeline.stop()
            
    def distance_from_box_human():

        W = 848
        H = 480

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

        print("[INFO] start streaming...")
        pipeline.start(config)

        aligned_stream = rs.align(rs.stream.color)  # alignment between color and depth
        point_cloud = rs.pointcloud()

        print("[INFO] loading model...")
        PATH_TO_CKPT = r"frozen_inference_graph.pb"

        # Load the TensorFlow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.compat.v1.import_graph_def(od_graph_def, name='')
            sess = tf.compat.v1.Session(graph=detection_graph)

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while True:
            frames = pipeline.wait_for_frames()
            frames = aligned_stream.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            points = point_cloud.calculate(depth_frame)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            scaled_size = (int(W), int(H))
            # expand image dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            image_expanded = np.expand_dims(color_image, axis=0)
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                      feed_dict={image_tensor: image_expanded})

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)

            print("[INFO] drawing bounding box on detected objects...")
            print("[INFO] each detected object has a unique color")

            for idx in range(int(num)):
                class_ = classes[idx]
                score = scores[idx]
                box = boxes[idx]

                if score > 0.8 and class_ == 1:  # 1 for human
                    left = box[1] * W
                    top = box[0] * H
                    right = box[3] * W
                    bottom = box[2] * H

                    width = right - left
                    height = bottom - top
                    bbox = (int(left), int(top), int(width), int(height))
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    # draw box
                    cv2.rectangle(color_image, p1, p2, (255, 0, 0), 2, 1)

                    # Calculate distance of the object from the camera
                    # Center point of the bounding box
                    cx = (p1[0] + p2[0]) // 2
                    cy = (p1[1] + p2[1]) // 2

                    # Depth value at the center of the bounding box
                    depth_value = depth_frame.get_distance(cx, cy)

                    # Convert depth value to distance in meters (assuming depth_frame units are millimeters)
                    distance = depth_value * 1000  # convert to millimeters

                    print("[INFO] object distance is: ", distance, "[mm]")
                    distance_txt = "{:.2f} m".format(distance / 1000)  # convert to meters for display

                    # Write distance information
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (p1[0], p1[1] + 40)
                    fontScale = 0.6
                    fontColor = (255, 255, 255)
                    lineType = 1
                    cv2.putText(color_image, distance_txt,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(1)

        # Stop streaming
        pipeline.stop()
          
    def tensorflow_through_net():

        W = 848
        H = 480

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)


        print("[INFO] start streaming...")
        pipeline.start(config)

        aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
        point_cloud = rs.pointcloud()

        print("[INFO] loading model...")
        # download model from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#run-network-in-opencv
        net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "faster_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        while True:
            frames = pipeline.wait_for_frames()
            frames = aligned_stream.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame().as_depth_frame()

            points = point_cloud.calculate(depth_frame)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            # skip empty frames
            if not np.any(depth_image):
                continue
            print("[INFO] found a valid depth frame")
            color_image = np.asanyarray(color_frame.get_data())

            scaled_size = (int(W), int(H))
            net.setInput(cv2.dnn.blobFromImage(color_image, size=scaled_size, swapRB=True, crop=False))
            detections = net.forward()

            print("[INFO] drawing bounding box on detected objects...")

            for detection in detections[0,0]:
                score = float(detection[2])
                idx = int(detection[1])
                print(" [DEBUG] classe : ",idx)

                if score > 0.8 and idx == 0:
                    left = detection[3] * W
                    top = detection[4] * H
                    right = detection[5] * W
                    bottom = detection[6] * H
                    width = right - left
                    height = bottom - top

                    bbox = (int(left), int(top), int(width), int(height))

                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(color_image, p1, p2, (255, 0, 0), 2, 1)

                    # x,y,z of bounding box
                    obj_points = verts[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])].reshape(-1, 3)
                    zs = obj_points[:,2]

                    z = np.median(zs)

                    ys = obj_points[:,1]
                    ys = np.delete(ys, np.where((zs < z - 1) | (zs > z + 1))) # take only y for close z to prevent including background

                    my = np.amin(ys, initial=1)
                    My = np.amax(ys, initial=-1)

                    height = (My - my) # add next to rectangle print of height using cv library
                    height = float("{:.2f}".format(height))
                    print("[INFO] object height is: ", height, "[m]")
                    height_txt = str(height)+"[m]"

                    # Write some Text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (p1[0], p1[1]+20)
                    fontScale = 1
                    fontColor = (255, 255, 255)
                    lineType = 2
                    cv2.putText(color_image, height_txt,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(1)

        # Stop streaming
        pipeline.stop()
    
    def obstacle_detect():
        def apply_contrast_stretching(image, low, high):
            stretched_image = image.copy()
            if high == low:
                return stretched_image

            stretched_image = np.where(stretched_image < low, 0, stretched_image)
            stretched_image = np.where((low <= stretched_image) & (stretched_image <= high),
                                        (255 / (high - low)) * (stretched_image - low), stretched_image)
            stretched_image = np.where(stretched_image > high, 255, stretched_image)

            return stretched_image

        def Canny_detector(img, weak_th=None, strong_th=None):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (5, 5), 1.4)
            gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
            gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
            mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

            mag_max = np.max(mag)
            if not weak_th:
                weak_th = mag_max * 0.2  # Increased threshold
            if not strong_th:
                strong_th = mag_max * 0.6  # Increased threshold

            height, width = img.shape
            for i_x in range(width):
                for i_y in range(height):
                    grad_ang = ang[i_y, i_x]
                    grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)
                    if grad_ang <= 22.5:
                        neighb_1_x, neighb_1_y = i_x - 1, i_y
                        neighb_2_x, neighb_2_y = i_x + 1, i_y
                    elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                        neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                        neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
                    elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                        neighb_1_x, neighb_1_y = i_x, i_y - 1
                        neighb_2_x, neighb_2_y = i_x, i_y + 1
                    elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                        neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
                        neighb_2_x, neighb_2_y = i_x + 1, i_y - 1
                    elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                        neighb_1_x, neighb_1_y = i_x - 1, i_y
                        neighb_2_x, neighb_2_y = i_x + 1, i_y

                    if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                        if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                            mag[i_y, i_x] = 0
                            continue

                    if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                        if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                            mag[i_y, i_x] = 0

            weak_ids = np.zeros_like(img)
            strong_ids = np.zeros_like(img)
            ids = np.zeros_like(img)

            for i_x in range(width):
                for i_y in range(height):
                    grad_mag = mag[i_y, i_x]
                    if grad_mag < weak_th:
                        mag[i_y, i_x] = 0
                    elif strong_th > grad_mag >= weak_th:
                        ids[i_y, i_x] = 1
                    else:
                        ids[i_y, i_x] = 2

            mag = np.uint8(mag)
            return mag

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        pipeline.start(config)

        try:
            while True:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                low_value = np.percentile(depth_image, 1)
                high_value = np.percentile(depth_image, 99)
                stretched_image = apply_contrast_stretching(depth_image, low_value, high_value)
                stretched_image = np.uint8(stretched_image)

                canny_edges_depth = cv2.Canny(stretched_image, 100, 200)

                # Apply morphological operations to clean up the edges
                kernel = np.ones((3, 3), np.uint8)
                canny_edges_depth = cv2.dilate(canny_edges_depth, kernel, iterations=1)
                canny_edges_depth = cv2.erode(canny_edges_depth, kernel, iterations=1)

                contours_depth, _ = cv2.findContours(canny_edges_depth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                depth_mask = np.zeros_like(stretched_image)
                cv2.drawContours(depth_mask, contours_depth, -1, 255, thickness=cv2.FILLED)

                canny_edges_rgb = Canny_detector(color_image)
                contours_rgb, _ = cv2.findContours(canny_edges_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                 # Detect and filter out floor regions in the RGB feed
                floor_mask = np.zeros_like(stretched_image)
                for cnt in contours_rgb:
                    if cv2.contourArea(cnt) > 5000:  # Consider large areas as potential floor
                        cv2.drawContours(floor_mask, [cnt], -1, 255, thickness=cv2.FILLED)

                depth_scale = depth_frame.get_units()
                for cnt in contours_rgb:
                    x, y, w, h = cv2.boundingRect(cnt)
                    mean_depth = np.mean(depth_image[y:y+h, x:x+w]) * depth_scale
                    if 0.5 < mean_depth < 3 and cv2.contourArea(cnt) > 500:  # Ignore small contours
                        # Filter out floor regions
                        if np.mean(floor_mask[y:y+h, x:x+w]) < 255:
                            cv2.drawContours(depth_mask, [cnt], -1, 255, thickness=cv2.FILLED)

                final_contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in final_contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    mean_depth = np.mean(depth_image[y:y+h, x:x+w]) * depth_scale
                    if 0.5 < mean_depth < 3 and cv2.contourArea(cnt) > 500:  # Final check to ignore small contours and floor
                        cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                cv2.imshow('Depth Feed', stretched_image)
                cv2.imshow('Canny Edge Detection Depth', canny_edges_depth)
                cv2.imshow('Canny Edge Detection RGB', canny_edges_rgb)
                cv2.imshow('Detected Obstacles', color_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
    #class_score_boxing()
    #boxing()
    #height_of_human()
    #distance_from_box_human()
    #tensorflow_through_net() 
    #obstacle_detect()

if __name__ == '__main__':
    main()