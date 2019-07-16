import math


class VectorException(Exception):
    pass


class VectorSizeException(VectorException):
    pass


class CrossProductException(VectorException):
    pass


class VectorDivisionException(VectorException):
    pass


class Vec:
    """ Simple n-dimensional geometric vector library """

    def __init__(self, *argv):
        self.elem = list(argv)
        self.nElem = len(self.elem)

    def __str__(self):
        return '[{}]'.format(', '.join([str(e) for e in self.elem]))

    def __repr__(self):
        return 'Vec{}({})'.format(self.nElem, ', '.join([str(e) for e in self.elem]))

    def dot(self, v):
        """ Calculate the dot product of two vectors """
        if self.nElem != v.nElem:
            raise VectorSizeException(
                'Vectors must contain the same number of elements in order to calculate a dot product')
        else:
            return sum([self.elem[i] * v.elem[i] for i in range(self.nElem)])

    def cross(self, v):
        """ Calculate the cross product of two vectors """
        if self.nElem != 3 or v.nElem != 3:
            raise CrossProductException('The cross product operation is only defined for two 3-dimensional vectors')
        else:
            return Vec(
                self.elem[1] * v.elem[2] - self.elem[2] * v.elem[1],
                self.elem[2] * v.elem[0] - self.elem[0] * v.elem[2],
                self.elem[0] * v.elem[1] - self.elem[1] * v.elem[0]
            )

    def mag2(self):
        """ Calculate the squared magnitude of vector """
        return sum([math.pow(e, 2) for e in self.elem])

    def mag(self):
        """ Calculate the magnitude of vector """
        return math.sqrt(sum([math.pow(e, 2) for e in self.elem]))

    def norm(self):
        """ Normalize the vector """
        m = self.mag()
        if m == 0:
            return Vec(*[0 for e in self.elem])
        return Vec(*[e / m for e in self.elem])

    def angle(self, v):
        """ Calculate the angle between two vectors """
        m1, m2 = self.mag(), v.mag()
        d = self.dot(v)
        return math.acos(d / (m1 * m2))

    def proj(self, v):
        """ Project the vector along a normal vector """
        return self.dot(v.norm())

    def dist(self, v):
        """ Calculate the distance between two vectors """
        return (v - self).mag()

    def each(self, f):
        """ Iterate through each item in the vector """
        return [f(e) for e in self.elem]

    def map(self, f):
        """ Apply a function to each item in the vector """
        self.elem = self.each(f)

    def __add__(self, o):
        if isinstance(o, Vec):
            if self.nElem != o.nElem:
                raise VectorSizeException(
                    'Vectors must contain the same number of elements in order to perform an addition operation')
            else:
                return Vec(*[self.elem[i] + o.elem[i] for i in range(self.nElem)])
        else:
            return Vec(*[e + o for e in self.elem])

    def __sub__(self, o):
        if isinstance(o, Vec):
            if self.nElem != o.nElem:
                raise VectorSizeException(
                    'Vectors must contain the same number of elements in order to perform a subtraction operation')
            else:
                return Vec(*[self.elem[i] - o.elem[i] for i in range(self.nElem)])
        else:
            return Vec(*[e - o for e in self.elem])

    def __mul__(self, o):
        # Cross product / Scalar multiplication
        if isinstance(o, Vec):
            return self.cross(o)
        else:
            return Vec(*[e * o for e in self.elem])

    def __invert__(self):
        # Normalization
        return self.norm()

    def __pow__(self, o):
        # Dot product / Power operator
        if isinstance(o, Vec):
            return self.dot(o)
        else:
            return Vec(*[e ** o for e in self.elem])

    def __mod__(self, o):
        # Angle / Modulo
        if isinstance(o, Vec):
            return self.angle(o)
        else:
            return Vec(*[e % o for e in self.elem])

    def __lshift__(self, n):
        # Rotate left
        n %= self.nElem
        return self.elem[n:] + self.elem[:n]

    def __rshift__(self, n):
        # Rotate right
        n %= self.nElem
        return self.elem[-n:] + self.elem[:-n]

    def __neg__(self):
        # Negate
        return Vec(*[-e for e in self.elem])

    def __truediv__(self, o):
        # Division
        if isinstance(o, Vec):
            raise VectorDivisionException('Unable to divide a vector by another vector')
        else:
            return Vec(*[e / o for e in self.elem])

    def __floordiv__(self, o):
        # Floor division
        if isinstance(o, Vec):
            raise VectorDivisionException('Unable to divide a vector by another vector')
        else:
            return Vec(*[e // o for e in self.elem])

    def __getitem__(self, i):
        return self.elem[i]

    def __setitem__(self, i, v):
        self.elem[i] = v

    def __len__(self):
        return len(self.elem)

    def __abs__(self):
        return self.mag()

    def __pos__(self):
        return self
