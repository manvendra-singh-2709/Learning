package com.data;

public class Image {
    private final double[][] data;
    private final int label;

    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public double[][] getData() {
      return this.data;
    }

    public int getLabel() {
      return this.label;
    }
}
