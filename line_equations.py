import math

class Line(object):

    def __init__(self, data):
            self.first, self.second = data

    def slope(self):
            '''Get the slope of a line segment'''
            (x1, y1), (x2, y2) = self.first, self.second
            try:
                    return (float(y2)-y1)/(float(x2)-x1)
            except ZeroDivisionError:
                    # line is vertical
                    return None

    def yintercept(self, slope):
            '''Get the y intercept of a line segment'''
            if slope != None:
                    x, y = self.first
                    return y - slope * x
            else:
                    return None

    def solve_for_y(self, x, slope, yintercept):
            '''Solve for Y cord using line equation'''
            if slope != None and yintercept != None:
                    return float(slope) * x + float(yintercept)
            else:
                    raise Exception('Can not solve on a vertical line')

    def solve_for_x(self, y, slope, yintercept):
            '''Solve for X cord using line equatio'''
            if slope != 0 and slope:
                    return float((y - float(yintercept))) / float(slope)
            else:
                    raise Exception('Can not solve on a horizontal line')

    def dist(self):
        (x1, y1), (x2, y2) = self.first, self.second
        return math.hypot(x2 - x1, y2 - y1)
