import numpy as np
import torch


class PluckerRay:
    def __init__(self, direction, moment):
        direction = np.array(direction)
        moment = np.array(moment)
        assert direction.shape == (3,)
        assert moment.shape == (3,)
        self.direction = direction
        self.moment = moment

    def __repr__(self):
        return f"PluckerRay({self.direction}, {self.moment})"

    def data(self):
        return [*self.direction, *self.moment]

    @classmethod
    def from_point_direction(cls, point, direction):
        """
        Convert from point direction representation.

        Args:
            rays: (..., 6).

        Returns:
            rays: (..., 6).
        """
        direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
        moment = np.cross(point, direction)
        return cls(direction=direction, moment=moment)

    def to_point_direction(self):
        """
        Convert to point direction representation.

        Returns:
            rays: (..., 6).
        """
        point = np.cross(self.direction, self.moment)
        direction = self.direction / np.linalg.norm(self.direction, axis=-1, keepdims=True)
        return np.concatenate((point, direction), axis=-1)

    def intersectPlane(self, plane_point, plane_normal):
        plane_normal = np.array(plane_normal)
        plane_point = np.array(plane_point)
        assert plane_normal.shape == (3,)
        assert plane_point.shape == (3,)
        # compute plane coefficients a, s.t. 0 = a0 + a1x + a2y + a3z represents the plane
        plane_coefs = np.array([-np.dot(plane_normal, plane_point), *plane_normal])
        return self.__intersectPlane(plane_coefs)

    def __intersectPlane(self, plane_coefs):
        """
        plane_coefs: (a0, a1, a2, a3) representing plane a0 + a1x + a2y + a3z = 0
        """
        plane_coefs = np.array(plane_coefs)
        assert plane_coefs.shape == (4,)
        a0 = plane_coefs[0]
        a = plane_coefs[1:]
        # compute intersection point in homogeneous coordinates (w, p)
        w = np.dot(a, self.direction)
        p = np.cross(a, self.moment) - a0 * self.direction

        if w == 0:
            return None
        else:
            return p / w


if __name__ == "__main__":
    radius = 0.5

    rays = []
    for i in range(20):
        d = np.random.rand(3) - 0.5
        d = d / np.linalg.norm(d) * radius
        # generate a random vector perpendicular to d
        m = np.random.rand(3)
        m = m - np.dot(m, d) * d
        m = m / np.linalg.norm(m) / 2 * radius
        ray = PluckerRay(d, m)
        rays.append(ray)
    # visualize the rays in 3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for ray in rays:
        # find intersection with cube [-1, 1]^3
        # find i1
        intersections = []
        for i in range(3):
            if ray.direction[i] != 0:
                inter = ray.intersectPlane([-1, -1, -1], [1 if j == i else 0 for j in range(3)])
                if (
                    inter[0] >= -1.1
                    and inter[0] <= 1.1
                    and inter[1] >= -1.1
                    and inter[1] <= 1.1
                    and inter[2] >= -1.1
                    and inter[2] <= 1.1
                ):
                    intersections.append(inter)
        # find i2
        for i in range(3):
            if ray.direction[i] != 0:
                inter = ray.intersectPlane([1, 1, 1], [-1 if j == i else 0 for j in range(3)])
                if (
                    inter[0] >= -1.1
                    and inter[0] <= 1.1
                    and inter[1] >= -1.1
                    and inter[1] <= 1.1
                    and inter[2] >= -1.1
                    and inter[2] <= 1.1
                ):
                    intersections.append(inter)
        if len(intersections) != 2:
            continue
        # plot line formed by intersection points
        ax.plot(
            (intersections[0][0], intersections[1][0]),
            (intersections[0][1], intersections[1][1]),
            (intersections[0][2], intersections[1][2]),
        )

    # draw unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v)) * radius
    y = np.outer(np.sin(u), np.sin(v)) * radius
    z = np.outer(np.ones(np.size(u)), np.cos(v)) * radius
    ax.plot_surface(x, y, z, color="b", alpha=0.1)

    plt.show()
