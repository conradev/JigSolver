#!/usr/bin/env python2

import cv2
import time
import numpy as np

FLANN_INDEX_KDTREE = 1

def bounding_box(points):
    """Takes a list of (x, y) and returns upper left and lower right points"""
    x_min = min([x for x, _ in points])
    x_max = max([x for x, _ in points])
    y_min = min([y for _, y in points])
    y_max = max([y for _, y in points])
    return (x_min, y_min), (x_max, y_max)

def get_edges(img):
    """Takes an image and gives an array same dimension as image that has 1 if
    edge, 0 if not for each pixel."""
    lowThreshold = 30
    ratio = 3
    kernel_size = 3
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(gray,(3,3),6)
    img = cv2.Canny(img,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    return img

def get_all_nonzero(img):
    """Return a list of (x,y) for each point in the image that is not zero"""
    edge_points = []
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            if col != 0:
                edge_points.append((j, i))
    return edge_points


def get_trimmed_box(img, edges):
    """Approximate the square bit off the puzzle piece in img"""
    edge_points = get_all_nonzero(edges)
    (x_min, y_min), (x_max, y_max) = bounding_box(edge_points)

    def should_trim(img, (x_min, y_min), (x_max, y_max)):
        threshold = 1.0 / 25
        img = img[y_min:y_max,x_min:x_max]
        if np.size(img):
            return float(np.count_nonzero(img))/np.size(img) < threshold
        return False

    trim_step = 25

    new_x_min = x_min
    while should_trim(edges, (new_x_min, y_min), (new_x_min + trim_step, y_max)):
        new_x_min += trim_step

    new_x_max = x_max
    while should_trim(edges, (new_x_max - trim_step, y_min), (new_x_max, y_max)):
        new_x_max -= trim_step

    new_y_min = y_min
    while should_trim(edges, (x_min, new_y_min), (x_max, new_y_min+trim_step)):
        new_y_min += trim_step

    new_y_max = y_max
    while should_trim(edges, (x_min, new_y_max - trim_step), (x_max, new_y_max)):
        new_y_max -= trim_step

    return new_x_min, new_x_max, new_y_min, new_y_max

def get_subimages(img, rows, columns):
    """Cuts an image into the number of rows and columns provided and returns an array of images"""
    height = len(img)
    width = len(img[0])
    sub_height = np.ceil(height / rows)
    sub_width = np.ceil(width / columns)
    sub_images = []
    for i in range(columns):
        for j in range(rows):
            (x_min, y_min) = (i * sub_width, j * sub_height)
            (x_max, y_max) = (x_min + sub_width, y_min + sub_height)
            sub_images.append(img[y_min:y_max,x_min:x_max])
    return sub_images

def compute(piece_file, board_file):
    piece_img = cv2.imread(piece_file)
    edges = get_edges(piece_img)
    fgrnd = get_blob(piece_img)
    mask = cv2.bitwise_or(edges, fgrnd)
    x_min, x_max, y_min, y_max = get_trimmed_box(piece_img, mask)

    piece = piece_img[y_min:y_max,x_min:x_max]

    board_img = cv2.imread(board_file)
    sub_images = get_subimages(board_img, 5, 7)
    return 'eventually this will be a good response'

def get_blob(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(hsv,(5,5),6)
    return cv2.inRange(img, (0,100,34), (255,255,255))

def find_features(img):
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT()
	return sift.detectAndCompute(gray_img, None)

def find_image_box(outer_img, inner_img):
	(outer_points, outer_descs) = find_features(outer_img)
	(inner_points, inner_descs) = find_features(inner_img)
	matcher = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_KDTREE, trees = 4), {})
	matches = matcher.knnMatch(inner_descs.astype(np.float32), outer_descs.astype(np.float32), 2)
	matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
	p0 = [inner_points[m.queryIdx].pt for m in matches]
	p1 = [outer_points[m.trainIdx].pt for m in matches]
	p0, p1 = np.float32((p0, p1))
	H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
	
	print matches

def main():
    piece_img = cv2.imread('samples/piece_small.jpg')[100:-100,100:-100]
    edges = get_edges(piece_img)
    fgrnd = get_blob(piece_img)
    mask = cv2.bitwise_or(fgrnd, edges)
    x_min, x_max, y_min, y_max = get_trimmed_box(piece_img, mask)

    piece = piece_img[y_min:y_max,x_min:x_max]
    # cv2.imshow('Display', piece)

    board_img = cv2.imread('samples/box_med.jpg')
    sub_images = get_subimages(board_img, 5, 7)
    # cv2.imshow('Bottom Left', sub_images[4])

    find_image_box(board_img, piece_img)

if __name__ == '__main__':
    main()
