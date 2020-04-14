import cv2
import numpy as np
import os
import math
from scipy.ndimage import interpolation as inter
from collections import OrderedDict

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


def align_images_v2(img, template_img, result_path="Output"):
    # Open the image files.
    # img1_color = cv2.imread(img1)  # Image to be aligned.
    # img2_color = cv2.imread(template_img)  # Reference image.

    # Convert to grayscale.
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 99)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

        # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img,
                                          homography, (width, height))

    return transformed_img


def correct_table_skew(img, result_path="Output"):
    # grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # grayscale = cv2.bitwise_not(grayscale)
    # (thresh, img_bin) = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imwrite(os.path.join(result_path, "img_bin.jpg"), img_bin)
    # coordinates = np.column_stack(np.where(img_bin > 0))
    # ang = cv2.minAreaRect(coordinates)[-1]
    #
    # if ang < -45:
    #     ang = -(90 + ang)
    # else:
    #     ang = -ang
    #
    # height, width = img.shape[:2]
    #
    # center_img = (width / 2, height / 2)
    # rotationMatrix = cv2.getRotationMatrix2D(center_img, ang, 1.0)
    #
    # rotated_img = cv2.warpAffine(img, rotationMatrix, (width, height), borderMode=cv2.BORDER_REPLICATE,
    #                              flags=cv2.INTER_CUBIC)

    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (thresh, img_bin) = cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin_inv = cv2.bitwise_not(img_bin)  # 255 - img_bin  # Invert the image

    cv2.imwrite(os.path.join(result_path, "img_bin.jpg"), img_bin_inv)

    kernel_length_verti = np.array(grayscale_img).shape[1] // 140
    kernel_length_hori = np.array(grayscale_img).shape[1] // 80

    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_verti))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_hori, 1))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_temp1 = cv2.erode(img_bin_inv, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

    img_temp2 = cv2.erode(img_bin_inv, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    alpha = 0.5
    beta = 1.0 - alpha

    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    cv2.imwrite(os.path.join(result_path, "img_final_bin.jpg"), img_final_bin)

    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=3)
    cv2.imwrite(os.path.join(result_path, "img_final_bin_inverted.jpg"), img_final_bin)

    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imwrite(os.path.join(result_path, "img_final_bin_inverted_threshed.jpg"), img_final_bin)
    img_final_bin = cv2.bitwise_not(img_final_bin)
    cv2.imwrite(os.path.join(result_path, "img_final_bin_inverted_1.jpg"), img_final_bin)

    coordinates = np.column_stack(np.where(img_final_bin > 0))
    ang = cv2.minAreaRect(coordinates)[-1]

    if ang < -45:
        ang = -(90 + ang)
    else:
        ang = -ang

    height, width = img.shape[:2]

    center_img = (width / 2, height / 2)
    rotationMatrix = cv2.getRotationMatrix2D(center_img, ang, 1.0)

    rotated_img = cv2.warpAffine(img, rotationMatrix, (width, height), borderMode=cv2.BORDER_REPLICATE,
                                 flags=cv2.INTER_CUBIC)
    return rotated_img


def correct_skew(image, delta=1, limit=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    return rotated


def deskew(img):
    thresh = img
    edges = cv2.Canny(thresh, 50, 200, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 1000, 55)
    try:
        d1 = OrderedDict()
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                deg = np.rad2deg(theta)
                #                print(deg)
                if deg in d1:
                    d1[deg] += 1
                else:
                    d1[deg] = 1

        t1 = OrderedDict(sorted(d1.items(), key=lambda x: x[1], reverse=False))
        print(list(t1.keys())[0], 'Angle', thresh.shape)
        non_zero_pixels = cv2.findNonZero(thresh)
        center, wh, theta = cv2.minAreaRect(non_zero_pixels)
        angle = list(t1.keys())[0]
        if angle > 160:
            angle = 180 - angle
        if angle < 160 and angle > 20:
            angle = 12
        root_mat = cv2.getRotationMatrix2D(center, angle, 1)
        rows, cols = img.shape
        rotated = cv2.warpAffine(img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)

    except:
        rotated = img
        pass
    return rotated


def unshear(img):
    gray = img
    thresh = img.copy()
    # print(thresh)
    trans = thresh.transpose()

    arr = []
    for i in range(thresh.shape[1]):
        arr.insert(0, trans[i].sum())

    arr = []
    for i in range(thresh.shape[0]):
        arr.insert(0, thresh[i].sum())

    y = thresh.shape[0] - 1 - np.nonzero(arr)[0][0]
    y_top = thresh.shape[0] - 1 - np.nonzero(arr)[0][-1]

    trans1 = thresh.transpose()
    sum1 = []
    for i in range(trans1.shape[0]):
        sum1.insert(i, trans1[i].sum())

    height = y - y_top
    max_value = 255 * height
    prev_num = len([i for i in sum1 if i >= (0.6 * max_value)])
    final_ang = 0

    # # print(arr)
    # # print(x,y)
    for ang in range(-25, 25, 3):
        thresh = gray.copy()
        # print(thresh[0].shape)
        # print(ang)
        #print('Ang', ang)
        if ang > 0:
            # print(ang)
            for i in range(y):
                temp = thresh[i]
                move = int((y - i) * (math.tan(math.radians(ang))))
                if move >= temp.size:
                    move = temp.size
                thresh[i][:temp.size - move] = temp[move:]
                thresh[i][temp.size - move:] = [0 for m in range(move)]
        else:
            # print(ang)
            for i in range(y):
                temp = thresh[i]
                move = int((y - i) * (math.tan(math.radians(-ang))))
                if move >= temp.size:
                    move = temp.size
                # print(temp[:-3])
                # print(temp[:temp.size-move].shape, thresh[i][move%temp.size:].shape)
                thresh[i][move:] = temp[:temp.size - move]
                thresh[i][:move] = [0 for m in range(move)]

        #         plt.imshow(thresh)
        #         plt.show()
        trans1 = thresh.transpose()
        sum1 = []
        for i in range(trans1.shape[0]):
            sum1.insert(i, trans1[i].sum())
        # print(sum1)
        num = len([i for i in sum1 if i >= (0.60 * max_value)])
        # print(num, prev_num)
        if (num >= prev_num):
            prev_num = num
            final_ang = ang
        # plt.imshow(thresh)
        # plt.show()
    # print("final_ang:", final_ang)

    thresh = gray.copy()
    if final_ang > 0:
        for i in range(y):
            temp = thresh[i]
            move = int((y - i) * (math.tan(math.radians(final_ang))))
            if move >= temp.size:
                move = temp.size
            thresh[i][:temp.size - move] = temp[move:]
            thresh[i][temp.size - move:] = [0 for m in range(move)]
    else:
        for i in range(y):
            temp = thresh[i]
            move = int((y - i) * (math.tan(math.radians(-final_ang))))
            # print(move)
            if move >= temp.size:
                move = temp.size
            thresh[i][move:] = temp[:temp.size - move]
            thresh[i][:move] = [0 for m in range(move)]

    #    plt.imshow(thresh)
    #    plt.show()
    return thresh
