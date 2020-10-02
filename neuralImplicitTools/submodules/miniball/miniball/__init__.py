__author__ = 'Konstantin Weddige'
import miniball.bindings


class Miniball:
    def __init__(self, points):
        """Computes the smallest enclosing ball of the points.

        :param points: coordinates as nested lists.
        """
        self._result = miniball.bindings.miniball(points)

    def center(self):
        """Returns a list that holds the coordinates of the center of the computed ball."""
        return self._result[0]

    def squared_radius(self):
        """Returns the squared radius of the computed ball."""
        return self._result[1]

    def relative_error(self):
        """Returns the maximum excess of any input point w.r.t. the computed
        ball, divided by the squared radius of the computed ball. The
        excess of a point is the difference between its squared distance
        from the center and the squared radius; Ideally, the return value
        is 0.
        """
        return self._result[2]

    def suboptimatily(self):
        """Returns the absolute value of the most negative
        coefficient in the affine combination of the support points that
        yields the center. Ideally, this is a convex combination, and there
        is no negative coefficient in which case 0 is returned.
        """
        return self._result[3]

    def is_valid(self):
        """Returns true if the relative error is at most tol, and the
        suboptimality is 0; the default tolerance is 10 times the
        coordinate type's machine epsilon
        """
        return self._result[4]

    def get_time(self):
        """Returns the time in seconds taken by the constructor call for
        computing the smallest enclosing ball."""
        return self._result[5]