package jflow.utils;

import java.awt.Color;
import java.awt.Graphics;

import javax.swing.JFrame;
import javax.swing.JPanel;

class ImageDisplay extends JPanel{

    private JFrame frame;
    private float[][][] image;
    private int scaleFactor;

    protected ImageDisplay(float[][][] image, int scaleFactor, String label) {
        this.image = image;
        int height = image.length;
        int width = image[0].length;


        // Since JFrame has a minimum visual width > 0
        if (height < 100 && scaleFactor < 2) {
            scaleFactor = 2;
        }

        this.scaleFactor = scaleFactor;

        frame = new JFrame();
        frame.setBounds(0, 0, width * scaleFactor, height * scaleFactor + 25);
        frame.setResizable(false);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setTitle(label);
        frame.add(this);
        repaint();
        frame.setVisible(true);

    }

    public void paintComponent(Graphics g) {
        int channel1; int channel2; int channel3;
        // RGB
        if (image[0][0].length == 3) {
            channel1 = 0;
            channel2 = 1;
            channel3 = 2;
        } 
        // grayscale
        else {
            channel1 = channel2 = channel3 = 0;
        }
        for (int i = 0; i < image.length; i++) {
            for (int j = 0; j < image[0].length; j++) {
                g.setColor(new Color((int)image[i][j][channel1], 
                    (int)image[i][j][channel2], (int)image[i][j][channel3]));
                g.fillRect(i * scaleFactor, j * scaleFactor, scaleFactor, scaleFactor);
            }
        }
    }
}
