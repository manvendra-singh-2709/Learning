package com.layers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public abstract class Layer implements Serializable {

    private static final long serialVersionUID = 1L;

    protected Layer _nextLayer;
    protected Layer _previousLayer;

    public abstract double[] getOutput(List<double[][]> input);

    public abstract double[] getOutput(double[] input);

    public abstract void backPropagation(List<double[][]> dld0);

    public abstract void backPropagation(double[] dld0);

    public abstract int getOutputLength();

    public abstract int getOutputRows();

    public abstract int getOutputColumns();

    public abstract int getOutputElements();

    public double[] matrixToVector(List<double[][]> matrix) {
        if (matrix == null || matrix.isEmpty())
            return new double[0];

        int length = matrix.size();
        int rows = matrix.get(0).length;
        int columns = matrix.get(0)[0].length;
        double[] vector = new double[length * rows * columns];

        int offset = 0;
        for (int l = 0; l < length; l++) {
            double[][] mat = matrix.get(l);
            for (int r = 0; r < rows; r++) {
                System.arraycopy(mat[r], 0, vector, offset, columns);
                offset += columns;
            }
        }
        return vector;
    }

    public List<double[][]> vectorToMatrix(double[] vector, int length, int rows, int columns) {
        List<double[][]> matrix = new ArrayList<>(length);
        int offset = 0;

        for (int l = 0; l < length; l++) {
            double[][] mat = new double[rows][columns];
            for (int r = 0; r < rows; r++) {
                System.arraycopy(vector, offset, mat[r], 0, columns);
                offset += columns;
            }
            matrix.add(mat);
        }
        return matrix;
    }

    public Layer getNextLayer() {
        return this._nextLayer;
    }

    public void setNextLayer(Layer value) {
        this._nextLayer = value;
    }

    public Layer getPreviousLayer() {
        return this._previousLayer;
    }

    public void setPreviousLayer(Layer value) {
        this._previousLayer = value;
    }
}