import math

from .core import NO_INTERSECTION, min_pos
from .vector import Vec


class Sphere:
    def __init__(self, pos, radius, material):
        self.pos = pos
        self.r2 = radius ** 2
        self.material = material

    def intersection(self, p, d):
        m = p - self.pos
        b = m.dot(d)
        c = m.mag2() - self.r2

        if c > 0 and b > 0:
            return NO_INTERSECTION

        discr = b ** 2 - c

        if discr < 0:
            return NO_INTERSECTION

        t = min_pos(-b - math.sqrt(discr), -b + math.sqrt(discr))
        q = p + d * t

        return t, q

    def normal(self, p, q):
        return ~(q - self.pos)


class Plane:
    def __init__(self, v0, v1, v2, material):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.material = material

    def intersection(self, p, d):
        n = (self.v1 - self.v0).cross(self.v2 - self.v0)

        n_dot_dir = n.dot(d)

        if abs(n_dot_dir) < .0001:
            return NO_INTERSECTION

        s = n.dot(self.v0)
        t = (n.dot(p) + s) / n_dot_dir

        if t < 0:
            return NO_INTERSECTION

        q = p + d * t

        return t, q

    def normal(self, p, q):
        n = (self.v1 - self.v0).cross(self.v2 - self.v0)
        d = n.dot(self.v0)
        r = Vec(*n, d).dot(Vec(*p, 1))

        if r < 0:
            n *= -1

        return ~n


class Triangle:
    def __init__(self, v0, v1, v2, material):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.material = material

    def intersection(self, p, d):
        n = (self.v1 - self.v0).cross(self.v2 - self.v0)

        n_dot_dir = n.dot(d)

        if abs(n_dot_dir) < 0.0001:
            return NO_INTERSECTION

        s = n.dot(self.v0)
        t = (n.dot(p) + s) / n_dot_dir

        if t < 0:
            return NO_INTERSECTION

        q = p + d * t

        if ((n.dot((self.v1 - self.v0).cross(q - self.v0)) < 0) or
                (n.dot((self.v2 - self.v1).cross(q - self.v1)) < 0) or
                (n.dot((self.v0 - self.v2).cross(q - self.v2)) < 0)):
            return NO_INTERSECTION

        return t, q

    def normal(self, p, q):
        n = (self.v1 - self.v0).cross(self.v2 - self.v0)
        s = n.dot(self.v0)
        r = Vec(*n, s).dot(Vec(*p, 1))

        if r < 0:
            n *= -1

        return ~n
