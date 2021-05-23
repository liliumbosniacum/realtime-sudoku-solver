package com.lilium.sudoku;

import com.lilium.sudoku.util.Line;
import com.lilium.sudoku.util.LineUtil;
import org.junit.jupiter.api.Test;
import org.opencv.core.Point;

import static org.junit.jupiter.api.Assertions.*;

public class LineUtilTest {

    @Test
    public void testIntersecting() {
        final Line l1 = new Line(
                new Point(0, 5),
                new Point(10, 5)
        );

        final Line l2 = new Line(
                new Point(5, 0),
                new Point(5, 10)
        );

        final Point intersection = LineUtil.findIntersection(l1, l2);
        assertNotNull(intersection);
        assertEquals(intersection.toString(), new Point(5, 5).toString());
    }

    @Test
    public void testNotIntersecting() {
        final Line l1 = new Line(
                new Point(0, 0),
                new Point(10, 0)
        );

        final Line l2 = new Line(
                new Point(0, 10),
                new Point(10, 10)
        );

        final Point intersection = LineUtil.findIntersection(l1, l2);
        assertNull(intersection);
    }
}
