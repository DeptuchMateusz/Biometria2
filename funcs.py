import os
import numpy as np
import cv2
import numpy as np
import cv2

def binarize(grayscale_image, threshold):
    h, w = grayscale_image.shape[0:2]
    mean_intensity = np.sum(grayscale_image) / (h * w)
    binary_image = np.where(grayscale_image > mean_intensity * threshold, 255, 0).astype(np.uint8)
    
    return binary_image

def keep_largest_contour(image):
    inverted = cv2.bitwise_not(image)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        biggest_contour = max(contours, key=cv2.contourArea)
        result = np.ones_like(image) * 255
        cv2.drawContours(result, [biggest_contour], -1, (0), thickness=cv2.FILLED)
        
        return result
    
    return image

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

def draw_rings_with_cuts(image, cx, cy, r_pupil, r_iris, ring_color=(0, 255, 0), cut_color=(255, 0, 0), thickness=1):

    output = image.copy()

    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    angle_masks = [
        [(75, 105)],
        [(56.5, 123.5), (236.5, 303.5)],
        [(45, 135), (225, 315)]
    ]

    for i in range(8):
        r_start = r_pupil + (r_iris - r_pupil) * i / 8
        r_end   = r_pupil + (r_iris - r_pupil) * (i + 1) / 8

        cv2.circle(output, (int(cx), int(cy)), int(r_start), ring_color, thickness)
        cv2.circle(output, (int(cx), int(cy)), int(r_end), ring_color, thickness)

        if i < 4:
            cuts = angle_masks[0]
        elif i < 6:
            cuts = angle_masks[1]
        else:
            cuts = angle_masks[2]

        for (start_angle, end_angle) in cuts:
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

    angle_masks = [
        [(75, 105)],
        [(56.5, 123.5), (236.5, 303.5)],
        [(45, 135), (225, 315)]
    ]

    theta = np.linspace(0, 2 * np.pi, width, endpoint=False)
    r = np.linspace(0, 1, height)
    r_grid, theta_grid = np.meshgrid(r, theta)

    ring_indices = np.floor(r_grid * 8).astype(int)
    ring_indices[ring_indices >= 8] = 7

    theta_deg_grid = np.degrees(theta_grid) % 360

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

    x_pupil = x + r_pupil * np.cos(theta_grid)
    y_pupil = y + r_pupil * np.sin(theta_grid)
    x_iris = x + r_iris * np.cos(theta_grid)
    y_iris = y + r_iris * np.sin(theta_grid)

    x_coords = (1 - r_grid) * x_pupil + r_grid * x_iris
    y_coords = (1 - r_grid) * y_pupil + r_grid * y_iris

    x_coords = np.clip(x_coords, 0, image.shape[1] - 1).astype(np.float32)
    y_coords = np.clip(y_coords, 0, image.shape[0] - 1).astype(np.float32)

    sampled = cv2.remap(image, x_coords, y_coords, interpolation=cv2.INTER_LINEAR)

    sampled[~mask] = 0

    return sampled.T

def build_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi=0):

    if isinstance(ksize, int):
        kx = ky = ksize
    else:
        kx, ky = ksize

    x_max = kx // 2
    y_max = ky // 2
    x = np.linspace(-x_max, x_max, kx)
    y = np.linspace(-y_max, y_max, ky)
    xv, yv = np.meshgrid(x, y)

    x_theta = xv * np.cos(theta) + yv * np.sin(theta)
    y_theta = -xv * np.sin(theta) + yv * np.cos(theta)

    gauss = np.exp(- (x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2))

    sinus = np.exp(1j * (2 * np.pi * x_theta / lambd + psi))

    kernel = gauss * sinus
    kernel_real = np.real(kernel)
    kernel_imag = np.imag(kernel)
    return kernel_real, kernel_imag

def generate_iris_code(unwrapped_iris, 
                       ksize=31, sigma=4.0, theta=0, lambd=10.0, gamma=0.5, psi=0,
                       n_rows=8, n_cols=128):

    height, width = unwrapped_iris.shape
    block_h = height // n_rows
    valid_mask = unwrapped_iris > 0

    ker_real, ker_imag = build_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi)

    iris_bits = np.zeros((n_rows, n_cols, 2), dtype=int)

    for row in range(n_rows):
        y0 = row * block_h
        y1 = y0 + block_h
        band = unwrapped_iris[y0:y1, :]
        mask_band = valid_mask[y0:y1, :]

        col_valid = np.sum(mask_band, axis=0)
        total_valid = np.sum(col_valid) / n_cols

        blocks = []
        acc = 0
        start = 0
        for x in range(width):
            acc += col_valid[x]
            if acc >= total_valid or x == width-1:
                blocks.append((start, x+1))
                start = x+1
                acc = 0
        if len(blocks) > n_cols:
            blocks = blocks[:n_cols]
        while len(blocks) < n_cols:
            blocks.append(blocks[-1])

        for i, (xs, xe) in enumerate(blocks):
            seg = band[:, xs:xe]
            seg_mask = mask_band[:, xs:xe]
            if np.sum(seg_mask) == 0:
                continue

            resp_real = cv2.filter2D(seg.astype(float), -1, ker_real)
            resp_imag = cv2.filter2D(seg.astype(float), -1, ker_imag)

            phase = np.arctan2(resp_imag, resp_real)
            mean_phase = np.mean(phase[seg_mask])

            iris_bits[row, i, 0] = int(mean_phase > 0)
            iris_bits[row, i, 1] = int(np.abs(mean_phase) > np.pi/2)

    iris_code_bin = np.zeros((n_rows, n_cols * 2), dtype=int)
    iris_code_bin[:, 0::2] = iris_bits[:, :, 0]
    iris_code_bin[:, 1::2] = iris_bits[:, :, 1]

    return iris_code_bin

def hamming_distance(code1, code2):

    if code1.shape != code2.shape:
        raise ValueError("Iris codes must have the same shape.")
    return np.sum(code1 != code2) / (code1.shape[0] * code1.shape[1])