#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>
#include <vector>
using namespace Eigen;

using Vector4dVector = std::vector<Vector4d>;
namespace pf_library {
class MotionModelAndResample {
public:
  MotionModelAndResample();

  Vector4dVector predict(const Vector4dVector &particles_positions,
                         const Vector3d &control);

  std::vector<int> resample_uniform(const std::vector<double> &weights);

protected:
  Matrix3d _noise_coeffs;
  Vector3d t2v(const Isometry2d &T);
  Isometry2d v2t(const Vector3d &v);
  std::random_device rd;
  std::mt19937 gen;
  std::normal_distribution<double> normal_distribution_;
};
} // namespace pf_library
