package com.lilium.sudoku.util;

import com.lilium.sudoku.classifier.util.DataUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Helper class containing methods regarding image interactions.
 *
 * @author mirza
 */
public final class ImageUtils {
    // region Members
    private static final MultiLayerNetwork NETWORK = DataUtil.loadModel();
    // endregion

    // region Constructor
    private ImageUtils() {}
    // endregion

    // region Implementation
    /**
     * Used to process forwarded {@link Mat} image.
     *
     * @param currentFrame Image to process.
     */
    public static void processImage(final Mat currentFrame) {
        final Mat processedFrame = new Mat(currentFrame.height(), currentFrame.width(), currentFrame.type());
        // Blur an image using a Gaussian filter
        Imgproc.GaussianBlur(currentFrame, processedFrame, new Size(7, 7), 1);

        // Switch from RGB to GRAY
        Imgproc.cvtColor(currentFrame, processedFrame, Imgproc.COLOR_RGB2GRAY);

        // Find edges in an image using the Canny algorithm
        Imgproc.Canny(processedFrame, processedFrame, 250, 150);

        // Dilate an image by using a specific structuring element
        // https://en.wikipedia.org/wiki/Dilation_(morphology)
        Imgproc.dilate(processedFrame, processedFrame, new Mat(), new Point(-1, -1), 1);

        // Mark outer contour
        final Point[] cornerPoints = ImageUtils.markOuterContourAndGetCornerPoints(processedFrame, currentFrame);

        // If corners are invalid we can return
        if (arePointsInvalid(cornerPoints)) {
            return;
        }

        // Warp image
        final Mat warpedImage = warpImage(currentFrame, cornerPoints);

        // Solve sudoku puzzle
        solveSudokuPuzzle(warpedImage);
    }

    /**
     * Used to mark outer rectangle and its corners.
     *
     * @param processedImage Image used for calculation of contours and corners.
     * @param originalImage Image on which marking is done.
     */
    public static Point[] markOuterContourAndGetCornerPoints(final Mat processedImage,
                                                             final Mat originalImage) {
        // Find contours of an image
        final List<MatOfPoint> allContours = new ArrayList<>();
        Imgproc.findContours(
                processedImage,
                allContours,
                new Mat(processedImage.size(), processedImage.type()),
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_NONE
        );

        // Filter out noise and display contour area value
        final List<MatOfPoint> filteredContours = allContours.stream()
                .filter(contour -> Imgproc.contourArea(contour) > 1000)
                .collect(Collectors.toList());

        int biggestContourIndex = getBiggestContourIndex(filteredContours);

        if (filteredContours.size() == 0) {
            return new Point[0];
        }

        // Find corner points and mark them
        final Point[] points = ImageUtils.getSortedCornerPoints(filteredContours.get(biggestContourIndex));
        if (points.length == 0) {
            return new Point[0];
        }

        for (final Point point : points) {
            if (point == null) {
                continue;
            }
            Imgproc.drawMarker(originalImage, point, new Scalar(255, 0, 0), 0, 30, 2);
        }

        return points;
    }
    // endregion

    // region Helpers
    /**
     * Used to find index of biggest polygonal curve.
     *
     * @param contours Contours for which index of biggest polygonal curve is calculated.
     * @return Returns an integer representing index of biggest polygonal curve.
     */
    private static int getBiggestContourIndex(final List<MatOfPoint> contours) {
        double maxValue = 0;
        var maxValueIndex = 0;
        for (var i = 0; i < contours.size(); i++) {
            final double contourArea = Imgproc.contourArea(contours.get(i));
            // If current value (contourArea) is bigger then maxValue then it becomes maxValue
            if (maxValue < contourArea) {
                maxValue = contourArea;
                maxValueIndex = i;
            }
        }

        return maxValueIndex;
    }
    /**
     * Used to get corner points of provided polygonal curve.
     *
     * @param poly Polygonal curve for which corner points are found.
     * @return Returns an array of found corner points.
     */
    private static Point[] getSortedCornerPoints(final MatOfPoint poly) {
        MatOfPoint2f approxPolygon = ImageUtils.approxPolygon(poly);
        Point[] sortedPoints = new Point[4];

        if (!approxPolygon.size().equals(new Size(1, 4))) {
            return sortedPoints;
        }

        // Calculate the center of mass of our contour image using moments
        final Moments moment = Imgproc.moments(approxPolygon);
        final int centerX = (int) (moment.get_m10() / moment.get_m00());
        final int centerY = (int) (moment.get_m01() / moment.get_m00());

        // We need to sort corner points as there is not guarantee that we will always get them in same order
        for(int i=0; i<approxPolygon.rows(); i++){
            final double[] data = approxPolygon.get(i, 0);
            final double dataX = data[0];
            final double dataY = data[1];

            // Sorting is done in reverence to center points (centerX, centerY)
            if(dataX < centerX && dataY < centerY) {
                sortedPoints[0] = new Point(dataX, dataY);
            } else if(dataX > centerX && dataY < centerY) {
                sortedPoints[1] = new Point(dataX, dataY);
            } else if (dataX < centerX && dataY > centerY) {
                sortedPoints[2] = new Point(dataX, dataY);
            } else if (dataX > centerX && dataY > centerY) {
                sortedPoints[3] = new Point(dataX, dataY);
            }
        }

        return sortedPoints;
    }

    /**
     * Approximates a polygonal curve.
     *
     * @param poly Polygonal curve.
     * @return .
     */
    private static MatOfPoint2f approxPolygon(final MatOfPoint poly) {
        final MatOfPoint2f destination = new MatOfPoint2f();
        final MatOfPoint2f source = new MatOfPoint2f();
        poly.convertTo(source, CvType.CV_32FC2);

        // Approximates a polygonal curve with the specified precision
        Imgproc.approxPolyDP(
                source,
                destination,
                0.02 * Imgproc.arcLength(source, true),
                true
        );

        return destination;
    }

    /**
     * Warp image perspective.
     *
     * @param currentFrame Current frame.
     * @param cornerPoints Corner points of sudoku puzzle.
     * @return Returns warped image.
     */
    private static Mat warpImage(final Mat currentFrame,
                                 final Point[] cornerPoints) {
        final Mat warpedImage = new Mat(currentFrame.height(), currentFrame.width(), currentFrame.type());

        double x = cornerPoints[1].x - cornerPoints[0].x;
        double y = cornerPoints[2].y - cornerPoints[1].y;

        final MatOfPoint2f dst = new MatOfPoint2f(
                new Point(0, 0),
                new Point(x, 0),
                new Point(0, y),
                new Point(x, y)
        );

        final MatOfPoint2f src = new MatOfPoint2f(cornerPoints);

        Imgproc.warpPerspective(currentFrame, warpedImage, Imgproc.getPerspectiveTransform(src, dst), new Size(x, y));

        // RGB to GRAY
        Imgproc.cvtColor(warpedImage, warpedImage, Imgproc.COLOR_RGB2GRAY);

        return warpedImage;
    }

    /**
     * Solves sudoku puzzle.
     *
     * @param warpedImage Warped image.
     */
    private static void solveSudokuPuzzle(final Mat warpedImage) {
        final int cellWidth = warpedImage.width() / 9;
        final int cellHeight = warpedImage.height() / 9;

        // Remove lines from warped image
        removeLines(warpedImage);

        final Size cellSize = new Size(
                cellWidth,
                cellHeight
        );

        final int[][] matrix = new int[9][9];
        for (int row = 0; row < 9; row++) {
            for (int col = 0; col < 9; col++) {
                final double tempXPosition = (col * cellWidth);
                final double tempYPosition = (row * cellHeight);

                final Mat cell = new Mat(
                        warpedImage,
                        new Rect(
                                new Point(
                                        tempXPosition,
                                        tempYPosition
                                ), cellSize)
                ).clone();

                final int count = Core.countNonZero(cell);
                if (count <= 50) {
                    matrix[row][col] = 0;
                } else { // We assume that there is a digit in the cell
                    // Save cell image for debugging
                    final Mat resizedCell = new Mat();
                    Imgproc.resize(cell, resizedCell, new Size(60,60));

                    try {
                        // Estimate cell value
                        final int estimatedValue = NETWORK != null
                                ? DataUtil.evaluateImage(resizedCell, NETWORK)
                                : -1;
                        matrix[row][col] = estimatedValue;
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }

        boolean solve = SudokuUtil.solve(matrix);
        if (solve) {
            // Print out solved matrix to the console
            printOutMatrix(matrix);
        }
    }

    public static void printOutMatrix(final int[][] matrix) {
        System.out.println("######################################");
        for (int[] values : matrix) {
            System.out.print(" - ");
            for (int value : values) {
                System.out.print(value + " - ");
            }
            System.out.println("");
        }
        System.out.println("######################################");
    }

    /**
     * Used to remove lines from processed image (lines forming the cells which hold digits).
     *
     * @param processedImage Processed image which should be black and white at this point.
     */
    private static void removeLines(final Mat processedImage) {
        final Mat lines = new Mat();

        Imgproc.adaptiveThreshold(
                processedImage,
                processedImage,
                255,
                Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY,
                9,
                11);

        Core.bitwise_not(processedImage, processedImage);

        // Detect lines
        Imgproc.HoughLinesP(
                processedImage,
                lines,
                1,
                Math.PI / 180,
                150,
                250,
                50
        );

        // Remove found lines. Removing in our case means just drawing over them with black color (our background is
        // also black).
        for (int r = 0; r < lines.rows(); r++) {
            double[] l = lines.get(r, 0);
            Imgproc.line(
                    processedImage,
                    new Point(l[0], l[1]),
                    new Point(l[2], l[3]),
                    new Scalar(0, 0, 255),
                    13,
                    Imgproc.FILLED,
                    0
            );
        }

        lines.release();
    }

    private static boolean arePointsInvalid(final Point[] cornerPoints) {
        if (cornerPoints == null || cornerPoints.length != 4) {
            return true;
        }

        for (final Point cornerPoint : cornerPoints) {
            if (cornerPoint== null) {
                return true;
            }
        }

        return false;
    }
    // endregion
}
