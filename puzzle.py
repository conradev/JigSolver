#!/usr/bin/env python2

import cv2
import time
import numpy as np



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


def main():
    piece_img = cv2.imread('samples/piece_small.jpg')
    edges = get_edges(piece_img)
    x_min, x_max, y_min, y_max = get_trimmed_box(piece_img, edges)

    piece = piece_img[y_min:y_max,x_min:x_max]

    cv2.imshow('Display', piece)
    cv2.waitKey(500)
    while True:
        time.sleep(0.5)

if __name__ == '__main__':
    main()
