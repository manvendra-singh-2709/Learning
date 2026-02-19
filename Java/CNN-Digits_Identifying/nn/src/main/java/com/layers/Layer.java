package com.layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {

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
        int length = matrix.size();
        int rows = matrix.get(0).length;
        int columns = matrix.get(0)[0].length;

        double[] vector = new double[length * rows * columns];

        int i = 0;
        for (int l = 0; l < length; l++) {
            double[][] mat = matrix.get(l);

            for (int r = 0; r < rows; r++) {
                double[] row = mat[r];
                System.arraycopy(row, 0, vector, i, columns);
                i += columns;
            }
        }

        return vector;
    }

    public List<double[][]> vectorToMatrix(double[] vector, int length, int rows, int columns) {

        List<double[][]> matrix = new ArrayList<>(length);

        int i = 0;

        for (int l = 0; l < length; l++) {
            double[][] mat = new double[rows][columns];

            for (int r = 0; r < rows; r++) {
                System.arraycopy(vector, i, mat[r], 0, columns);
                i += columns;
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
