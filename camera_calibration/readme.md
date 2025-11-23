# Camera calibration — quick notes

## Overview

Calibrate a camera using chessboard images. Mounted the chess board (8x8) with 35 mm square size on the wall and took 12 images from various angles and positions.

## What I tried

* Tried writing the cam calibration code from scratch. Failed miserably!!!! (`cam_calib_scratch.py`)
* Resorted to OpenCV → `cam_calib_cv.py`
* undistorted the images after calibration and projected a 3d cube on one of the image
