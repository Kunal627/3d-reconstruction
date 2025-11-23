# illustrate_calib.py
# Brief: Demonstrate common uses of saved camera calibration parameters.
# Uses of calibrated camera params (K, dist, rvecs, tvecs):
#  - Undistort images to remove lens distortion for accurate visualization and processing.
#  - Project 3D points (e.g., cube corners) onto chessboard images to visualize calibration accuracy.
import numpy as np
import cv2

def load_calibration(file_path):
    """Load calibration parameters from a file."""
    data = np.load(file_path, allow_pickle=True)    
    K = data['K']
    dist = data['dist']
    rvecs = data['rvecs']
    tvecs = data['tvecs']
    image_paths = data['image_paths']
    imgpoints = data['imgpoints']
    return K, dist, rvecs, tvecs, image_paths, imgpoints



if __name__ == "__main__":
    K, dist, rvecs, tvecs, img_paths, imgpoints = load_calibration("./output/camera_calibration_params.npz")
    print("Loaded calibration parameters:")
    print("Intrinsic matrix K:\n", K)
    print("Distortion coefficients:", dist.ravel())
    print(f"Number of images: {len(img_paths)}")
    print(f"Number of image points sets: {len(imgpoints)}")

    # 1. Undistort a sample image and display

    sample_img = cv2.imread(img_paths[4])   
    corners = imgpoints[4]
    h, w = sample_img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))

    cv2.drawChessboardCorners(sample_img, (7,7), corners, True)
    cv2.imshow("Original Image", sample_img)
    cv2.waitKey(0)

    # The corners are drawn on the original distorted image to show the effect of distortion
    # the corners may appear slightly off from the actual chessboard corners
    sample_img = cv2.imread(img_paths[4])  
    undistorted_img = cv2.undistort(sample_img, K, dist, None, new_K)
    cv2.drawChessboardCorners(undistorted_img, (7,7), corners, True)
    cv2.imshow("Undistorted Image", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2. Project a cube onto the chessboard in the image
    SQUARE_SIZE = 0.035  # square size in meters

    cube_pts = np.float32([
        [0, 0, 0], [SQUARE_SIZE, 0, 0], [SQUARE_SIZE, SQUARE_SIZE, 0], [0, SQUARE_SIZE, 0],
        [0, 0, -SQUARE_SIZE], [SQUARE_SIZE, 0, -SQUARE_SIZE], [SQUARE_SIZE, SQUARE_SIZE, -SQUARE_SIZE], 
        [0, SQUARE_SIZE, -SQUARE_SIZE]
    ])

    img = cv2.imread(img_paths[4])
    rvec = rvecs[4]
    tvec = tvecs[4]
    imgpts, _ = cv2.projectPoints(cube_pts, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2)
    # Draw cube edges
    img = cv2.drawContours(img, [np.int32(imgpts[:4])], -1, (0,255,0), 3)  # base
    for i in range(4):
        img = cv2.line(img, tuple(np.int32(imgpts[i])), tuple(np.int32(imgpts[i+4])), (255,0,0), 3)  # vertical edges
    img = cv2.drawContours(img, [np.int32(imgpts[4:])], -1, (0,0,255), 3)  # top
    cv2.imshow("Cube Projection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()