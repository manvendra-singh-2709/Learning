package com.layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer{

    private final int _stepSize;
    private final int _windowSize;

    private final int _inLength;
    private final int _inRows;
    private final int _inColumns;

    public MaxPoolLayer(int _stepSize, int _windowSize, int _inLength, int _inRows, int _inColumns) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inColumns = _inColumns;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double[] getOutput(double[] input) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void backPropagation(List<double[][]> dld0) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void backPropagation(double[] dld0) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public int getOutputLength() {
        return _inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows-_windowSize)/_stepSize + 1;
    }

    @Override
    public int getOutputColumns() {
        return (_inColumns-_windowSize)/_stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return _inLength * getOutputColumns() * getOutputRows();
    }

    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {
        List<double[][]> output = new ArrayList<>();

        for (int i = 0; i < input.size(); i++) {
            output.add(pool(input.get(i)));
        }

        return output;
    }

    public double[][] pool(double[][] input) {
        double[][] output = new double[getOutputRows()][getOutputColumns()];

        for (int r = 0; r < getOutputRows(); r += _stepSize) {
            for (int c = 0; c < getOutputColumns(); c =+ _stepSize) {
                double max = 0.0;

                for (int x = 0; x < _windowSize; x++) {
                    for (int y = 0; y < _windowSize; y++) {
                        max = (max < input[r+x][c+y]) ? input[r+x][c+y] : max;
                    }
                }
                output[r][c] = max;
            }
        }
        return output;
    }
}
