package com.data;

public class MatrixUtility {

    public static double[] add(double[] a, double[] b) {

        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must be of same length.");
        }

        double[] result = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }

        return result;
    }

    public static double[][] add(double[][] a, double[][] b) {

        double[][] out = new double[a.length][a[0].length];

        int outRow = out.length;
        int outColumn = out[0].length;

        for (int i = 0; i < outRow; i++) {
            for (int j = 0; j < outColumn; j++) {
                out[i][j] = a[i][j] + b[i][j];
            }
        }
        return out;
    }

    public static double[] multiply(double[] a, double scalar) {

        double[] result = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * scalar;
        }

        return result;
    }

    public static double[][] multiply(double[][] a, double scalar) {

        int aRows = a.length;
        int aCols = a[0].length;

        double[][] result = new double[aRows][aCols];

        for (int i = 0; i < aRows; i++) {
            for (int j = 0; j < aCols; j++) {
                result[i][j] = a[i][j] * scalar;
            }
        }

        return result;
    }

    public static double[][] multiply(double[][] a, double[][] b) {

        int aRows = a.length;
        int aCols = a[0].length;
        int bRows = b.length;
        int bCols = b[0].length;

        if (aCols != bRows) {
            throw new IllegalArgumentException(
                    "Matrix multiplication not possible: columns of A must match rows of B.");
        }

        double[][] result = new double[aRows][bCols];

        for (int i = 0; i < aRows; i++) {
            for (int j = 0; j < bCols; j++) {
                for (int k = 0; k < aCols; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        return result;
    }

}
