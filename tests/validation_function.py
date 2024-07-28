from utils import Coord2D, intersection_line_segment

if __name__ == '__main__':
    # Test the intersection function
    line_p1 = Coord2D(0, 0)
    line_p2 = Coord2D(3, 0)
    seg_p1 = Coord2D(5, 1)
    seg_p2 = Coord2D(7, -1)

    intersection = intersection_line_segment(line_p1, line_p2, seg_p1, seg_p2)
    print(intersection)