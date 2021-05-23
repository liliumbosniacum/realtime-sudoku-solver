package com.lilium.sudoku.util;

import org.opencv.core.Point;

import static java.lang.Float.NEGATIVE_INFINITY;

/**
 * https://rosettacode.org/wiki/Find_the_intersection_of_two_lines#Java
 */
public final class LineUtil {
    private LineUtil() {
    }

    public static Point findIntersection(final Line l1, final Line l2) {
        final double a1 = l1.e.y - l1.s.y;
        final double b1 = l1.s.x - l1.e.x;
        final double c1 = a1 * l1.s.x + b1 * l1.s.y;

        final double a2 = l2.e.y - l2.s.y;
        final double b2 = l2.s.x - l2.e.x;
        final double c2 = a2 * l2.s.x + b2 * l2.s.y;

        final double delta = a1 * b2 - a2 * b1;
        Point point = new Point((b2 * c1 - b1 * c2) / delta, (a1 * c2 - a2 * c1) / delta);

        if (point.x == NEGATIVE_INFINITY || point.y == NEGATIVE_INFINITY) {
            return null;
        }

        return point;
    }
}
