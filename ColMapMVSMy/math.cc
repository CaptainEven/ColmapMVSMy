#include "math.h"

namespace colmap {

	size_t NChooseK(const size_t n, const size_t k)
	{
		if (k == 0)
		{
			return 1;
		}

		return (n * NChooseK(n - 1, k - 1)) / k;
	}


	double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center_1,
		const Eigen::Vector3d& proj_center_2,
		const Eigen::Vector3d& point3D)
	{
		const double baseline = (proj_center_1 - proj_center_2).squaredNorm();

		const double ray_1 = (point3D - proj_center_1).norm();
		const double ray_2 = (point3D - proj_center_2).norm();

		// Angle between rays at point within the enclosing triangle,
		// see "law of cosines".
		const double angle = std::abs(
			std::acos((ray_1 * ray_1 + ray_2 * ray_2 - baseline) / (2 * ray_1 * ray_2)));

		if (IsNaN(angle))
		{
			return 0.0;
		}
		else
		{
			// Triangulation is unstable for acute angles (far away points) and
			// obtuse angles (close points), so always compute the minimum angle
			// between the two intersecting rays.
			return std::min(angle, M_PI - angle);
		}
	}

}  // namespace colmap
