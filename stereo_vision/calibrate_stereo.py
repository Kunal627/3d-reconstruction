import cv2
import numpy as np

if __name__ == "__main__":
    img_left = cv2.imread("./stereo_vision/images_png/left_cam.png")
    img_right = cv2.imread("./stereo_vision/images_png/right_cam.png")

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    h, w = gray_left.shape

    orb = cv2.ORB_create(200)
    kp1, des1 = orb.detectAndCompute(gray_left, None)
    kp2, des2 = orb.detectAndCompute(gray_right, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts_left = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_right = np.float32([kp2[m.trainIdx].pt for m in matches])

    F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)
    pts_left_inliers = pts_left[mask.ravel() == 1]
    pts_right_inliers = pts_right[mask.ravel() == 1]

    retval, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts_left_inliers, pts_right_inliers, F, imgSize=(w, h)
    )

    rect_left = cv2.warpPerspective(img_left, H1, (w, h))
    rect_right = cv2.warpPerspective(img_right, H2, (w, h))

    rect_left_gray = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    rect_right_gray = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    num_disparities = 64  # must be divisible by 16
    block_size = 11

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8*3*block_size**2,
        P2=32*3*block_size**2,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        disp12MaxDiff=1
    )

    disparity = stereo.compute(rect_left_gray, rect_right_gray).astype(np.float32) / 16.0

    # Normalize disparity for visualization
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    cv2.imshow('Disparity', disp_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
