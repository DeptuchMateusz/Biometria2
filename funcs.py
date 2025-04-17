from Morphology import erode, dilate
from Contour import keep_largest_contour
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter1d
import streamlit as st
import numpy as np
import cv2
from math import pi


def get_pupil(image_raw):
    image_bin = binarize(image_raw, threshold=0.22)

    # czyszczenie
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image = cv2.erode(image_bin, kernel, iterations=2)
    image = cv2.dilate(image, kernel, iterations=2)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    image = cv2.medianBlur(image, 5)
    image = keep_largest_contour(image)

    # projekcja
    binary_image = (image > 0).astype(np.uint8)
    horizontal_proj = np.sum(binary_image, axis=1)
    vertical_proj = np.sum(binary_image, axis=0)

    # środek
    x = int(np.mean(np.where(vertical_proj == np.min(vertical_proj))[0]))
    y = int(np.mean(np.where(horizontal_proj == np.min(horizontal_proj))[0]))

    # promień
    left_edge = np.min(np.where(vertical_proj < max(vertical_proj)))
    right_edge = np.max(np.where(vertical_proj < max(vertical_proj)))
    radius_horizontal = (right_edge - left_edge) // 2

    top_edge = np.min(np.where(horizontal_proj < max(horizontal_proj)))
    bottom_edge = np.max(np.where(horizontal_proj < max(horizontal_proj)))
    radius_vertical = (bottom_edge - top_edge) // 2

    radius_pupil = (radius_horizontal + radius_vertical) // 2

    # zaznaczenie środka i promienia
    line_width = image_bin.shape[1] // 128
    image_center = cv2.cvtColor(image_raw.copy(), cv2.COLOR_GRAY2BGR)
    image_center = cv2.circle(image_center, (x, y), radius_pupil, (255, 0, 0), line_width)
    image_center = cv2.circle(image_center, (x, y), 0, (255, 0, 0), line_width)

    return image_center, (x, y), radius_pupil

def get_iris(image_raw, x, y, radius_pupil):


    mask_pupil = np.zeros_like(image_raw)
    cv2.circle(mask_pupil, (x, y), radius_pupil, 255, thickness=-1)

    mask_outer = np.ones_like(image_raw) * 255
    cv2.circle(mask_outer, (x, y), radius_pupil, 0, thickness=-1)  

    outer_region = cv2.bitwise_and(image_raw, image_raw, mask=mask_outer)    
    mean_brightness = np.mean(outer_region[mask_outer > 0])

    threshold = mean_brightness / 255 *1.25

    image_bin = binarize(image_raw, threshold=threshold)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image = cv2.dilate(image_bin, big_kernel, iterations=12)
    image = cv2.erode(image, small_kernel, iterations=5)
    image = keep_largest_contour(image)
    image = cv2.erode(image, big_kernel, iterations=2)

    binary_image = (image > 0).astype(np.uint8)
    horizontal_proj = np.sum(binary_image, axis=1)
    vertical_proj = np.sum(binary_image, axis=0)

    left_edge = np.min(np.where(vertical_proj < max(vertical_proj)))
    right_edge = np.max(np.where(vertical_proj < max(vertical_proj)))
    radius_horizontal = (right_edge - left_edge) // 2

    top_edge = np.min(np.where(horizontal_proj < max(horizontal_proj)))
    bottom_edge = np.max(np.where(horizontal_proj < max(horizontal_proj)))
    radius_vertical = (bottom_edge - top_edge) // 2

    radius_iris = (radius_horizontal + radius_vertical) // 2

    line_width = image_bin.shape[1] // 128
    image_center = cv2.cvtColor(image_raw.copy(), cv2.COLOR_GRAY2BGR)
    image_center = cv2.circle(image_center, (x, y), radius_iris, (255, 0, 0), line_width)
    image_center = cv2.circle(image_center, (x, y), 0, (255, 0, 0), line_width)

    return image_center, (x, y), radius_iris


def unwrap_iris(image, cx, cy, r_pupil, r_iris, num_angular_samples=None):
    def angle_mask(theta_deg, mask_ranges):
        for start, end in mask_ranges:
            if start <= theta_deg <= end:
                return False
        return True

    # Angle cutouts per ring group
    angle_masks = [
        #[(0, 255), (285, 360)],  # rings 1-4
        [(0, 75), (105, 360)],
        [(0, 56.5), (123.5, 236.5), (303.5, 360)],  # rings 5-6
        [(0, 45), (135, 225), (315, 360)],  # rings 7-8
    ]

    all_rings = []
    max_width = 0  # max horizontal resolution

    for i in range(8):
        # Compute radial bounds
        r_start = r_pupil + (r_iris - r_pupil) * i / 8
        r_end = r_pupil + (r_iris - r_pupil) * (i + 1) / 8
        ring_height = int(np.ceil(r_end - r_start))

        # Select mask
        if i < 4:
            valid_ranges = angle_masks[0]
        elif i < 6:
            valid_ranges = angle_masks[1]
        else:
            valid_ranges = angle_masks[2]

        samples = []

        for r in np.linspace(r_start, r_end, ring_height):
            num_points = int(np.ceil(2 * pi * r))
            for t in range(num_points):
                theta = 2 * pi * t / num_points
                theta_deg = np.degrees(theta)

                if not angle_mask(theta_deg % 360, valid_ranges):
                    continue

                x = int(round(cx + r * np.cos(theta)))
                y = int(round(cy + r * np.sin(theta)))

                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    samples.append((theta_deg % 360, r, image[y, x]))

        if not samples:
            all_rings.append(np.zeros((ring_height, 1), dtype=image.dtype))
            continue

        # Sort by angle
        samples = sorted(samples, key=lambda s: s[0])
        theta_vals = np.array([s[0] for s in samples])
        pixel_vals = np.array([s[2] for s in samples])

        if num_angular_samples is None:
            max_width = max(max_width, len(np.unique(theta_vals)))

        all_rings.append((theta_vals, pixel_vals, ring_height))

    if num_angular_samples is None:
        num_angular_samples = max_width

    unwrapped_image = []

    for ring in all_rings:
        if isinstance(ring, np.ndarray):
            resized = cv2.resize(ring, (num_angular_samples, ring.shape[0]), interpolation=cv2.INTER_LINEAR)
            unwrapped_image.append(resized)
            continue

        theta_vals, pixel_vals, height = ring

        # Target angle sampling
        target_theta = np.linspace(0, 360, num_angular_samples, endpoint=False)
        interp_vals = np.interp(target_theta, theta_vals, pixel_vals)

        # Repeat vertically
        ring_img = np.tile(interp_vals[np.newaxis, :], (height, 1))
        unwrapped_image.append(ring_img.astype(image.dtype))

    # Stack all rings
    full_unwrapped = np.vstack(unwrapped_image)

    return full_unwrapped


def draw_rings_with_cuts(image, cx, cy, r_pupil, r_iris, ring_color=(0, 255, 0), cut_color=(255, 0, 0), thickness=1):
    """
    Draws the edges of 8 iris rings (as green circles) and marks cutout angle arcs as red curved lines.
    """
    output = image.copy()

    # Convert grayscale to color if needed
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    # Define cutout angle ranges
    angle_masks = [
        #[(255, 285)], 
        [(75, 105)],                                 # Rings 1–4
        [(56.5, 123.5), (236.5, 303.5)],             # Rings 5–6
        [(45, 135), (225, 315)],                     # Rings 7–8
    ]

    for i in range(8):
        r_start = r_pupil + (r_iris - r_pupil) * i / 8
        r_end   = r_pupil + (r_iris - r_pupil) * (i + 1) / 8

        # Draw outer and inner edges of the ring as green circles
        cv2.circle(output, (int(cx), int(cy)), int(r_start), ring_color, thickness)
        cv2.circle(output, (int(cx), int(cy)), int(r_end), ring_color, thickness)

        # Select cutout angle ranges based on ring index
        if i < 4:
            cuts = angle_masks[0]
        elif i < 6:
            cuts = angle_masks[1]
        else:
            cuts = angle_masks[2]

        # Draw red arc(s) for each cutout on both inner and outer edges
        for (start_angle, end_angle) in cuts:
            # Inner arc
            cv2.ellipse(
                output,
                center=(int(cx), int(cy)),
                axes=(int(r_start), int(r_start)),
                angle=0,
                startAngle=start_angle,
                endAngle=end_angle,
                color=cut_color,
                thickness=thickness
            )
            # Outer arc
            cv2.ellipse(
                output,
                center=(int(cx), int(cy)),
                axes=(int(r_end), int(r_end)),
                angle=0,
                startAngle=start_angle,
                endAngle=end_angle,
                color=cut_color,
                thickness=thickness
            )

    return output


def unwrap_iris_with_masks(image, x, y, r_pupil, r_iris, height=64, width=512):
    """
    Unwraps iris region into a rectangular strip (height x width),
    excluding angular segments (cutouts) specific to ring index.
    """

    # Define angle masks per group
    angle_masks = [
        #[(255, 285)],                                # Rings 1–4
        [(75, 105)],
        [(56.5, 123.5), (236.5, 303.5)],             # Rings 5–6
        [(45, 135), (225, 315)]                      # Rings 7–8
    ]

    # Build normalized polar grid
    theta = np.linspace(0, 2 * np.pi, width, endpoint=False)
    r = np.linspace(0, 1, height)
    r_grid, theta_grid = np.meshgrid(r, theta)

    # Map each (r, θ) to a ring index (0–7)
    ring_indices = np.floor(r_grid * 8).astype(int)
    ring_indices[ring_indices >= 8] = 7  # Clamp to 7

    # Convert θ to degrees [0, 360)
    theta_deg_grid = np.degrees(theta_grid) % 360

    # Create validity mask
    mask = np.ones_like(theta_deg_grid, dtype=bool)
    for i in range(8):
        if i < 4:
            blocked = angle_masks[0]
        elif i < 6:
            blocked = angle_masks[1]
        else:
            blocked = angle_masks[2]

        for (start, end) in blocked:
            invalid = (ring_indices == i) & (theta_deg_grid >= start) & (theta_deg_grid <= end)
            mask[invalid] = False

    # Compute sampling coordinates (interpolation from pupil to iris)
    x_pupil = x + r_pupil * np.cos(theta_grid)
    y_pupil = y + r_pupil * np.sin(theta_grid)
    x_iris = x + r_iris * np.cos(theta_grid)
    y_iris = y + r_iris * np.sin(theta_grid)

    x_coords = (1 - r_grid) * x_pupil + r_grid * x_iris
    y_coords = (1 - r_grid) * y_pupil + r_grid * y_iris

    # Clip to image boundaries
    x_coords = np.clip(x_coords, 0, image.shape[1] - 1).astype(np.float32)
    y_coords = np.clip(y_coords, 0, image.shape[0] - 1).astype(np.float32)

    # Sample using bilinear interpolation
    sampled = cv2.remap(image, x_coords, y_coords, interpolation=cv2.INTER_LINEAR)

    # Apply angular cutout mask
    sampled[~mask] = 0  # or 255 or np.nan — you decide

    return sampled.T


def normalize_iris(image, x, y, r_pupil, r_iris, height=64, width=512):
    theta = np.linspace(0, 2 * np.pi, width)
    r = np.linspace(0, 1, height)

    # Create grid for polar coords
    r_grid, theta_grid = np.meshgrid(r, theta)

    # Interpolate from pupil to iris boundary
    x_pupil = x + r_pupil * np.cos(theta_grid)
    y_pupil = y + r_pupil * np.sin(theta_grid)
    x_iris = x + r_iris * np.cos(theta_grid)
    y_iris = y + r_iris * np.sin(theta_grid)

    # Linear interpolation between pupil and iris
    x_coords = (1 - r_grid) * x_pupil + r_grid * x_iris
    y_coords = (1 - r_grid) * y_pupil + r_grid * y_iris

    # Map coordinates to image
    x_coords = np.clip(x_coords, 0, image.shape[1] - 1).astype(np.float32)
    y_coords = np.clip(y_coords, 0, image.shape[0] - 1).astype(np.float32)

    # Remap image using polar coords
    normalized = cv2.remap(image, x_coords, y_coords, cv2.INTER_LINEAR)

    return normalized.T # transpose to get (height x width)


def gabor_wavelet_1d(size=31, f=3):
        sigma = 0.5 * np.pi * f
        x = np.linspace(-size // 2, size // 2, size)
        gabor = np.exp(-x**2 / (2 * sigma**2)) * np.cos(2 * np.pi * f * x)
        return gabor

    # def gabor_filter(size=31, f=3, orientation=0.0):
    #     sigma = 0.5 * np.pi * f
    #     x = np.linspace(-size // 2, size // 2, size)
    #     y = np.linspace(-size // 2, size // 2, size)
    #     x, y = np.meshgrid(x, y)
    #     x_theta = x * np.cos(orientation) + y * np.sin(orientation)
    #     y_theta = -x * np.sin(orientation) + y * np.cos(orientation)

    #     gb = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2)) * \
    #          np.cos(2 * np.pi * x_theta * f)

    #     return gb

def iris_code(normalized_iris, num_strips=8, f =3):
    num_strips = num_strips + 2 
    height, width = normalized_iris.shape
    strip_height = height // num_strips
    iris_code = []

    # gabor = gabor_filter(size=31, f = f, orientation=0)
    gabor = gabor_wavelet_1d(size=31, f=f)

    for i in range(1, num_strips - 1):  # pomijamy górny i dolny pasek
        start_row = i * strip_height
        end_row = (i + 1) * strip_height
        strip = normalized_iris[start_row:end_row, :]

        rect_width = strip.shape[1] // 128
        one_d_strip = []

        for j in range(128):
            start_col = j * rect_width
            end_col = (j + 1) * rect_width
            segment = strip[:, start_col:end_col]

            # Uśrednianie wzdłuż kolumny (czyli kierunek radialny)
            mean_intensity = segment.mean(axis=0)
            # Gaussowskie wygładzenie – radialne
            smoothed = gaussian_filter1d(mean_intensity, sigma=2)
            # Jedna wartość jako średnia z wygładzonego sygnału 1D
            value = np.mean(smoothed)

            one_d_strip.append(value)

        # Zamiana 1D sygnału na 1D kod binarny przy pomocy falki Gabora
        one_d_strip = np.array(one_d_strip) # 2D dla conv
        response = np.convolve(one_d_strip, gabor, mode='same')
        binary = (response > 0).astype(np.uint8).flatten()

        iris_code.append(binary)

    # Zwracamy kod jako macierz (pasy x 128 bitów)
    return np.array(iris_code)


def hamming_distance(code1, code2):
    """
    Calculate the Hamming distance between two binary codes.
    """
    if code1.shape != code2.shape:
        raise ValueError("Iris codes must have the same shape.")
    return np.sum(code1 != code2) / (code1.shape[0] * code1.shape[1])



def binarize(grayscale_image, threshold):
    h, w = grayscale_image.shape[0:2]
    mean_intensity = np.sum(grayscale_image) / (h * w)
    binary_image = np.where(grayscale_image > mean_intensity * threshold, 255, 0).astype(np.uint8)

    
    return binary_image