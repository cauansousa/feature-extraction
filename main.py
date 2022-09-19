import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from PIL import Image, ImageFile
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import extcolors
import webcolors
import os
from colormap import rgb2hex
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.transform import resize
from u2net_test import mask
import argparse
import json


def creating_mask(img_path):
    output = mask(img_path)
    RESCALE = 255
    out_img = img_to_array(output)
    THRESHOLD = 0.2
    out_img[out_img > THRESHOLD] = 1
    out_img[out_img <= THRESHOLD] = 0
    shape = out_img.shape
    a_layer_init = np.ones(shape=(shape[0], shape[1], 1))
    mul_layer = np.expand_dims(out_img[:, :, 0], axis=2)
    a_layer = mul_layer * a_layer_init
    rgba_out = np.append(out_img, a_layer, axis=2)
    original_image_path = img_path
    original_image = load_img(original_image_path)
    inp_img = img_to_array(original_image)
    inp_img /= RESCALE
    # since the output image is rgba, convert this also to rgba, but with no transparency
    a_layer = np.ones(shape=(shape[0], shape[1], 1))
    rgba_inp = np.append(inp_img, a_layer, axis=2)
    # simply multiply the 2 rgba images to remove the backgound
    rem_back = (rgba_inp * rgba_out)
    rem_back_scaled = Image.fromarray((rem_back * RESCALE).astype('uint8'), 'RGBA')
    display(rem_back_scaled)
    # save the resulting image to colab
    rem_back_scaled.save('results/image_background_removed.png')
    return 'results/image_background_removed.png'


def pose_points(path_image):
    os.remove('images/*')
    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

    POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
    inWidth = 640
    inHeight = 480
    net = cv2.dnn.readNetFromTensorflow("/content/cv2pose/graph_opt.pb")
    cap = cv2.imread('/content/U-2-Net/results/image_background_removed.png')

    while cv2.waitKey(1) < 0:
        frame = cap

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        net.setInput(
            cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert (len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > 0.2 else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert (partFrom in BODY_PARTS)
            assert (partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        t, _ = net.getPerfProfile()
        freq = cv2.getTickFrequency() / 1000
        cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2_imshow(frame)
        break

    with open("points.json", "w") as f:
        json.dump(points, f)

    modelo = cv2.imread('results/image_background_removed.png')

    null = 'null'

    with open('/content/U-2-Net/points.json', 'r') as f:
        point = json.load(f)

    i = 0
    for pontos in point:
        if pontos == 'null':
            continue
        if i == 1:
            neck = pontos[1]
        if i == 8:
            cint = pontos[1] + 25
        if i == 15:
            eye = pontos[1] - 25
        i += 1

    cv2.imwrite('images/torso.png', modelo[neck - 25:cint, 0:])
    cv2.imwrite('images/pernas.png', modelo[cint - 25:, 0:])
    cv2.imwrite('images/cabeca.png', modelo[:neck - 10])

    return ('images/torso.png', 'images/pernas.png')


def color(img_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Permitir que imagens corrompidas sejam usadas
    segm_image = img_path  # Caminho da imagem
    # Mostrar imagem
    plt.figure(figsize=(9, 9))
    img = plt.imread(segm_image)
    plt.imshow(img)
    plt.axis('off')

    colors_x = extcolors.extract_from_path(segm_image, tolerance=12, limit=12)
    rgb = (colors_x[0][0][0])

    if rgb == (255, 255, 255):
        print('White')
    if rgb == (0, 0, 0):
        print('Black')
    elif rgb[0] == rgb[1] and rgb[1] == rgb[2] and rgb[0] == rgb[2]:
        print('Grey')

    elif rgb[0] > rgb[1] and rgb[0] > rgb[2]:
        if rgb[0] > 209 and rgb[1] > 179 and rgb[2] > 134 and rgb != (255, 192, 203):
            print('Beige')
        elif (rgb == (184, 134, 11) or rgb == (189, 83, 107) or rgb == (139, 69, 19) or rgb == (160, 82, 45) or rgb == (
        188, 143, 143)) or rgb[0] > 204 and rgb[1] > 104 and rgb[2] < 144:
            print('Brown')
        elif rgb[0] > 204 and rgb[1] < 193 and rgb[2] > 91:
            print('Pink')
        elif rgb == (255, 140, 0) or rgb == (255, 165, 0):
            print('Orange')
        elif rgb == (255, 215, 0):
            print('Gold')
        elif rgb == (189, 83, 107):
            print('Green')
        else:
            print('Red')

    elif rgb[1] > rgb[0] and rgb[1] > rgb[2] or rgb == (47, 79, 79):
        print('green')

    elif rgb[2] > rgb[1] and rgb[2] > rgb[0] or rgb == (0, 255, 255) or rgb == (0, 139, 139) or rgb == (0, 128, 128):
        if rgb[0] > 122 and rgb[1] < 113 and rgb[2] > 203 or rgb == (128, 0, 128) or rgb == (75, 0, 130):
            print('Purple')
        else:
            print('Blue')

    elif rgb == (128, 128, 0):
        print('Green')
    elif rgb == (255, 255, 0):
        print('Yellow')
    elif rgb == (255, 0, 255) or rgb == (238, 130, 238) or rgb == (218, 112, 214) or rgb == (221, 160, 221):
        print('Pink')


def color_extraction(img_path):
    # Create a mask for the person and remove background
    mask_path = creating_mask(img_path)
    # Separate the person in three parts and save a photo of each part and takes the path from it
    # body_parts - list of the body_image paths
    body_parts = pose_points(mask_path)
    # Create a empty list to save the colors
    body_colors = []

    for parts in body_parts:
        # Create a new mask for each body part and remove the background again
        body_masks = creating_mask(parts)
        # Get the main color of the image
        color = color(body_masks)
        # Add to the body_colors list
        # First element[0] is the torso color, second[1] is the legs color
        body_colors.append(color)

    # Return the list of colors
    return body_colors



