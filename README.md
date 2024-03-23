# vio-hello-world-impl


* [Feature Extraction](#feature-extraction)
  + [Harris Features](#harris-features)
  + [ORB Features](#orb-features)
* [Pyramid Optical Flow](#pyramid-optical-flow)
* [Triangulate](#triangulate)
* [Jacobi Evaluate](#jacobi-evaluate)
  + [Numerical Jacobi Evaluate](#numerical-jacobi-evaluate)
  + [Auto Differ Jacobi Evaluate](#auto-differ-jacobi-evaluate)
* [IMU Integration](#imu-integration)
  + [Midpoint Integration](#midpoint-integration)
  + [Euler Integration](#euler-integration)

## Feature Extraction

### Harris Features

| OpenCV                          | Impl                          |
| ------------------------------- | ----------------------------- |
| ![](./images/opencv_harris_screenshot_10.03.2024.png) | ![](./images/self_harris_screenshot_10.03.2024.png) |

### ORB Features

| OpenCV                          | Impl                                      | ORB-SLAM  Impl |
| ------------------------------- |----------------------------------------------------|----------------------------------------------------|
| ![](./images/OpenCV%20Impl_screenshot_10.03.2024.png) | ![](./images/ORB%20Impl_screenshot_10.03.2024.png) | ![](./images/ORB%20Impl_screenshot_10.03.2024.png) |

## Pyramid Optical Flow

Implementation of optical flow tracking

<div align="center"><img src="./images/feat_tracker_result.png" style="zoom:48%;" /></div>

## Triangulate

Triangulation via SVD(Singular Value Decomposition).

## Jacobi Evaluate

Jacobi Evaluate

### Numerical Jacobi Evaluate

Numerical Jacobi Evaluate

### Auto Differ Jacobi Evaluate

Auto Differ Jacobi Evaluate

## IMU Integration 

IMU Integration

### Midpoint Integration

Midpoint Integration

### Euler Integration

Euler Integration