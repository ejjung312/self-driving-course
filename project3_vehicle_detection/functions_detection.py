import cv2
import matplotlib.pyplot as plt
import numpy as np

from project3_vehicle_detection.functions_feat_extraction import image_to_features
# from project_5_utils import stitch_together

def draw_labeled_bounding_boxes(img, labeled_frame, num_objects):
    """
    Starting from labeled regions, draw enclosing rectangles in the original color frame.
    """
    # Iterate through all detected cars
    for car_number in range(1, num_objects+1):
        # Find pixels with each car_number label value
        rows, cols = np.where(labeled_frame == car_number)

        # Find minimum enclosing rectangle
        x_min, y_min = np.min(cols), np.min(rows)
        x_max, y_max = np.max(cols), np.max(rows)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=6)

    return img


def compute_heatmap_from_detections(frame, hot_windows, threshold=5, verbose=False):
    """
    Compute heatmaps from windows classified as positive, in order to filter false positives.
    """
    h,w,c = frame.shape

    heatmap = np.zeros(shape=(h,w), dtype=np.uint8)

    for bbox in hot_windows:
        # for each bounding box, add heat to the corresponding rectangle in the image
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]
        heatmap[y_min:y_max, x_min:x_max] += 1 # add heat

    # apply threshold + morphological closure to remove noise
    _, heatmap_thresh = cv2.threshold(heatmap, threshold, 255, type=cv2.THRESH_BINARY)
    heatmap_thresh = cv2.morphologyEx(heatmap_thresh, op=cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13)), iterations=1)

    if verbose:
        f, ax = plt.subplots(1,3)
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[1].imshow(heatmap, cmap='hot')
        ax[2].imshow(heatmap_thresh, cmap='hot')
        plt.show()

    return heatmap, heatmap_thresh


def slide_window(img, x_start_stop=[None,None], y_start_stop=[None,None],
                 xy_window=(64,64), xy_overlap=(0.5,0.5)):
    """
    Implementation of a sliding window in a region of interest of the image.
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    x_span = x_start_stop[1] - x_start_stop[0]
    y_span = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    n_x_pix_per_step = np.int(xy_window[0] * (1-xy_overlap[0]))
    n_y_pix_per_step = np.int(xy_window[1] * (1-xy_overlap[1]))

    # Compute the number of windows in x / y
    n_x_windows = np.int(x_span/n_x_pix_per_step) - 1
    n_y_windows = np.int(y_span/n_y_pix_per_step) - 1

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions.
    for i in range(n_y_windows):
        for j in range(n_x_windows):
            # Calculate window position
            start_x = j * n_x_pix_per_step + x_start_stop[0]
            end_x = start_x + xy_window[0]
            start_y = i * n_y_pix_per_step + y_start_stop[0]
            end_y = start_y + xy_window[1]

            # Append window position to list
            window_list.append(((start_x, start_y), (end_x, end_y)))

    return window_list


def draw_boxes(img, bbox_list, color=(0,0,255), thick=6):
    """
    Draw all bounding boxes in `bbox_list` onto a given image.
    """
    img_copy = np.copy(img)

    # Iterate through the bounding boxes
    for bbox in bbox_list:
        tl_corner = tuple(bbox[0])
        br_corner = tuple(bbox[1])
        cv2.rectangle(img_copy, tl_corner, br_corner, color, thick)

    # Return the image copy with boxes drawn
    return img_copy

# Define a function you will pass an image and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, feat_extraction_params):
    hot_windows = [] # list to receive positive detection windows

    for window in windows:
        # Extract the current window from original image
        resize_h, resize_w = feat_extraction_params['resize_h'], feat_extraction_params['resize_w']
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]],
                              (resize_w, resize_h))

        # Extract features for that window using single_img_features()
        features = image_to_features(test_img, feat_extraction_params)

        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1,-1))

        # Predict on rescaled features
        prediction = clf.predict(test_features)

        # If positive (prediction == 1) then save the window
        if prediction == 1:
            hot_windows.append(window)

    return hot_windows



















