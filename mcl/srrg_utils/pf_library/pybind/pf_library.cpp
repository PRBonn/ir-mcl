#include "pf_library.h"
#include <iostream>

namespace pf_library {
MotionModelAndResample::MotionModelAndResample() : rd(), gen(rd()) {
  _noise_coeffs << 0.01, 0.0005, 0.0002, 0.0005, 0.0001, 0.0001, 0.001, 0.00001,
      0.05;
}

Vector4dVector
MotionModelAndResample::predict(const Vector4dVector &particles_positions,
                                const Vector3d &control) {
  Vector4dVector new_particles_positions(particles_positions);
  if (control.squaredNorm() < 1e-8) {
    return new_particles_positions;
  }
  Vector3d scales(fabs(control[0]), fabs(control[1]), fabs(control[2]));
  Vector3d std_deviations = (_noise_coeffs * scales).cwiseSqrt();
  int num_particles = new_particles_positions.size();
  for (int p = 0; p < num_particles; ++p) {
    Vector3d noise;
    for (int i = 0; i < 3; ++i) {
      noise[i] = std_deviations[i] * normal_distribution_(gen);
    }
    Vector3d noisy_control = control + noise;
    const Vector3d &position = particles_positions[p].head<3>();
    new_particles_positions[p].head<3>() =
        t2v(v2t(position) * v2t(noisy_control));
  }
  return new_particles_positions;
}

std::vector<int>
MotionModelAndResample::resample_uniform(const std::vector<double> &weights) {
  double acc = 0;
  for (const double &w : weights) {
    acc += w;
  }
  double inverse_acc = 1. / acc;
  double cumulative_value = 0;
  int n = weights.size();
  double step = 1. / n;
  double threshold = step * drand48();
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  auto idx = indices.begin();
  int k = 0;
  for (int i = 0; i < n; i++) {
    cumulative_value += weights[i] * inverse_acc;
    while (cumulative_value > threshold) {
      *idx = i;
      idx++;
      k++;
      threshold += step;
    }
  }
  return indices;
}
Vector3d MotionModelAndResample::t2v(const Isometry2d &T) {
  Vector3d v = Vector3d::Zero();
  v.head<2>() = T.translation();
  const auto &R = T.linear();
  v(2) = std::atan2(R(1, 0), R(0, 0));
  return v;
}
Isometry2d MotionModelAndResample::v2t(const Vector3d &v) {
  Isometry2d T = Isometry2d::Identity();
  T.translation() = v.head<2>();
  Rotation2Dd R(v(2));
  T.linear() = R.matrix();
  return T;
}

} // namespace pf_library
