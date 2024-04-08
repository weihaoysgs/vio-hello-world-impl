/**
 *    This file is part of OV²SLAM.
 *
 *    Copyright (C) 2020 ONERA
 *
 *    For more information see <https://github.com/ov2slam/ov2slam>
 *
 *    OV²SLAM is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    OV²SLAM is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with OV²SLAM.  If not, see <https://www.gnu.org/licenses/>.
 *
 *    Authors: Maxime Ferrera     <maxime.ferrera at gmail dot com> (ONERA, DTIS - IVA),
 *             Alexandre Eudes    <first.last at onera dot fr>      (ONERA, DTIS - IVA),
 *             Julien Moras       <first.last at onera dot fr>      (ONERA, DTIS - IVA),
 *             Martial Sanfourche <first.last at onera dot fr>      (ONERA, DTIS - IVA)
 */
#pragma once

#include <cstddef>
namespace backend {
class CameraIntrinsicParametersBlock
{
public:
  CameraIntrinsicParametersBlock() = default;

  CameraIntrinsicParametersBlock( const int id, const double fx, const double fy, const double cx,
                                  const double cy ) {
    id_ = id;
    values_[0] = fx;
    values_[1] = fy;
    values_[2] = cx;
    values_[3] = cy;
  }

  CameraIntrinsicParametersBlock( const int id, const Eigen::Matrix3d &K ) {
    id_ = id;
    values_[0] = K( 0, 0 );
    values_[1] = K( 1, 1 );
    values_[2] = K( 0, 2 );
    values_[3] = K( 1, 2 );
  }

  CameraIntrinsicParametersBlock( const CameraIntrinsicParametersBlock &block ) {
    id_ = block.id_;
    for ( size_t i = 0; i < ndim_; i++ ) {
      values_[i] = block.values_[i];
    }
  }

  CameraIntrinsicParametersBlock &operator=( const CameraIntrinsicParametersBlock &block ) {
    id_ = block.id_;
    for ( size_t i = 0; i < ndim_; i++ ) {
      values_[i] = block.values_[i];
    }
    return *this;
  }

  void getCalib( double &fx, double &fy, double &cx, double &cy ) {
    fx = values_[0];
    fy = values_[1];
    cx = values_[2];
    cy = values_[3];
  }

  inline double *values() { return values_; }

  static const size_t ndim_ = 4;
  double values_[ndim_] = { 0., 0., 0., 0. };
  int id_ = -1;
};
}  // namespace backend