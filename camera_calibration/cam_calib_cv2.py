# camera_calib_cv2.py
# Brief: Calibrate a camera from chessboard PNG images using OpenCV.
# Inputs: PNG chessboard images located in `./camera_calibration/images_png`.
# Outputs: Prints intrinsic matrix, distortion coefficients, and reprojection errors;
#          returns (K, dist, rvecs, tvecs, img_paths, imgpoints). Visualizes matches.

import cv2
import numpy as np
from glob import glob
from PIL import Image

# --- CONFIGURATION ---
IMG_DIR = "./camera_calibration/images_png"   # folder of PNG chessboard images
BOARD_SIZE = (7, 7)      # number of inner corners per row and column
SQUARE_SIZE = 0.035      # square size in meters (35 mm)
TEST = False     # whether to run the test and visualization
# Termination criteria for cornerSubPix
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def load_and_resize(path, max_dim=800):
    """
    Load image and downsize to max width or height = max_dim, keeping aspect ratio.
    Returns:
        gray: grayscale image (numpy array)
        color: resized color image (numpy array)
        scale: scaling factor
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    color = np.array(img)
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    return gray, color, scale


def make_object_points(board_size, square_size):
    """Create 3D points for chessboard corners in the board plane."""
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp[:, :2] *= square_size
    return objp


def calibrate_camera():
    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    objp = make_object_points(BOARD_SIZE, SQUARE_SIZE)

    img_paths = sorted(glob(f"{IMG_DIR}/*png"))
    print("Found images:", img_paths)

    for p in img_paths:
       gray, color, scale = load_and_resize(p, max_dim=800)
    
       ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
       if ret:
           corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    
           # Draw on the resized image for visualization
           cv2.drawChessboardCorners(color, BOARD_SIZE, corners2, ret)
           cv2.imshow('Corners', color)
           cv2.waitKey(200)
    
           # Rescale corners back to original image coordinates
           corners2_orig = corners2 / scale
    
           objpoints.append(objp)
           imgpoints.append(corners2_orig)
       else:
           print("Chessboard not found in image:", p)


    cv2.destroyAllWindows()

    # calibrate
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("Calibration RMS error:", ret)
    print("Intrinsic matrix K:\n", K)
    print("Distortion coefficients:", dist.ravel())

    # per-image reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        print(f"Image {i} reprojection error: {error:.3f} px")
        total_error += error
    print("Mean reprojection error:", total_error / len(objpoints))

    return K, dist, rvecs, tvecs, img_paths, imgpoints


def test_and_visualize(K, dist, rvecs, tvecs, img_paths, objpoints, imgpoints):
    """Project points and visualize reprojection vs detected corners."""
    for i, p in enumerate(img_paths):
        color = cv2.imread(p)
        imgpts, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        imgpts = imgpts.reshape(-1, 2)

        for (u, v) in imgpts:
            cv2.circle(color, (int(u), int(v)), 5, (0, 0, 255), -1)  # red projected
        for (u, v) in imgpoints[i].reshape(-1, 2):
            cv2.circle(color, (int(u), int(v)), 3, (0, 255, 0), -1)  # green detected

        cv2.imshow("Reprojection vs Detected", color)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

def save_calibration(file_path, K, dist, rvecs, tvecs, image_paths, imgpoints):
    """Save calibration parameters to a file."""
    np.savez(file_path, K=K, dist=dist, rvecs=rvecs, tvecs=tvecs, image_paths=image_paths, imgpoints=imgpoints)
    print(f"Calibration parameters saved to {file_path}")

if __name__ == "__main__":
    K, dist, rvecs, tvecs, img_paths, imgpoints = calibrate_camera()
    save_calibration("./output/camera_calibration_params.npz", K, dist, rvecs, tvecs, img_paths, imgpoints)

    if TEST:
        objp = make_object_points(BOARD_SIZE, SQUARE_SIZE)
        test_and_visualize(K, dist, rvecs, tvecs, img_paths, [objp]*len(img_paths), imgpoints)
