package com.lilium.sudoku.util;

import org.opencv.core.Point;

/**
 * Class representing a line defined by a start and end point.
 *
 * @author mirza
 */
public class Line {
    public Point s, e;

    public Line(Point s, Point e) {
        this.s = s;
        this.e = e;
    }
}
