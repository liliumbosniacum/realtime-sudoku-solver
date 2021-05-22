package com.lilium.sudoku;

import com.lilium.sudoku.util.ImageUtils;
import com.lilium.sudoku.util.UiUtils;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;

public class Application {

    public static void main(final String args[]) {
        // Load OpenCV
        OpenCV.loadShared();

        // Create panels
        final JPanel cameraFeed = new JPanel();
        UiUtils.createJFrame(cameraFeed);

        // Create video capture object (index 0 is default camera)
        final VideoCapture camera = new VideoCapture(0);

        // Start realtime solving
        startSolving(cameraFeed, camera).run();
    }

    private static Runnable startSolving(final JPanel cameraFeed,
                                         final VideoCapture camera) {
        return () -> {
            final Mat frame = new Mat();

            while (true) {
                // Read frame from camera
                camera.read(frame);

                // Process frame
                ImageUtils.processImage(frame);

                // Draw current frame
                UiUtils.drawImage(frame, cameraFeed);
            }
        };
    }
}
