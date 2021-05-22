package com.lilium.sudoku.util;

import org.opencv.core.Mat;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;

/**
 * Helper class containing all methods concerning UI.
 */
public final class UiUtils {
    // region Constructor
    private UiUtils() {}
    // endregion

    // region Implementation
    public static void createJFrame(final JPanel... panels) {
        final JFrame window = new JFrame("Realtime Sudoku Solver");
        window.setSize(new Dimension(panels.length * 640, 480));
        window.setLocationRelativeTo(null);
        window.setResizable(false);
        window.setLayout(new GridLayout(1, panels.length));

        for (final JPanel panel : panels) {
            window.add(panel);
        }

        window.setVisible(true);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    /**
     * Draw forwarded mat image to forwarded panel.
     *
     * @param mat Image to draw.
     * @param panel Panel on which to draw image.
     */
    public static void drawImage(final Mat mat, final JPanel panel) {
        // Get buffered image from mat frame
        final BufferedImage image = UiUtils.convertMatToBufferedImage(mat);

        // Draw image to panel
        final Graphics graphics = panel.getGraphics();
        graphics.drawImage(image, 0, 0, panel);
    }
    // endregion

    // region Helpers
    /**
     * Converts forwarded {@link Mat} to {@link BufferedImage}.
     *
     * @param mat Mat to convert.
     * @return Returns converted BufferedImage.
     */
    private static BufferedImage convertMatToBufferedImage(final Mat mat) {
        // Create buffered image
        final BufferedImage bufferedImage = new BufferedImage(
                mat.width(),
                mat.height(),
                mat.channels() == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR
        );

        // Write data to image
        final WritableRaster raster = bufferedImage.getRaster();
        final DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();
        mat.get(0, 0, dataBuffer.getData());

        return bufferedImage;
    }
    // endregion
}
