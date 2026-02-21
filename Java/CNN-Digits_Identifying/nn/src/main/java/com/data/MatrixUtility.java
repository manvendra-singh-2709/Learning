package com.data;

public class MatrixUtility {

    public static double[] add(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    public static double[][] add(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            double[] aRow = a[i];
            double[] bRow = b[i];
            double[] outRow = out[i];
            for (int j = 0; j < cols; j++) {
                outRow[j] = aRow[j] + bRow[j];
            }
        }
        return out;
    }

    public static void addInPlace(double[][] target, double[][] source) {
        int rows = target.length;
        int cols = target[0].length;
        for (int i = 0; i < rows; i++) {
            double[] tRow = target[i];
            double[] sRow = source[i];
            for (int j = 0; j < cols; j++) {
                tRow[j] += sRow[j];
            }
        }
    }

    public static void multiplyInPlace(double[][] target, double scalar) {
        int rows = target.length;
        int cols = target[0].length;
        for (int i = 0; i < rows; i++) {
            double[] row = target[i];
            for (int j = 0; j < cols; j++) {
                row[j] *= scalar;
            }
        }
    }

    public static double[] multiply(double[] a, double scalar) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * scalar;
        }
        return result;
    }

    public static double[][] multiply(double[][] a, double scalar) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            double[] aRow = a[i];
            double[] resRow = result[i];
            for (int j = 0; j < cols; j++) {
                resRow[j] = aRow[j] * scalar;
            }
        }
        return result;
    }
}