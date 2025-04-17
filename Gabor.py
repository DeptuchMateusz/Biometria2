import numpy as np
import cv2


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