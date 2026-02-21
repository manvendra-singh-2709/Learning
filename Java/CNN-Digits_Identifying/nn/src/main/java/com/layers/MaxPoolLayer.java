package com.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class MaxPoolLayer extends Layer {

    private final int _stepSize;
    private final int _windowSize;
    private final int _inLength;
    private final int _inRows;
    private final int _inColumns;

    private int[][][] _lastMaxRow;
    private int[][][] _lastMaxColumn;

    public MaxPoolLayer(int _stepSize, int _windowSize, int _inLength, int _inRows, int _inColumns) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inColumns = _inColumns;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        return _nextLayer.getOutput(maxPoolForwardPass(input));
    }

    @Override
    public double[] getOutput(double[] input) {
        return getOutput(vectorToMatrix(input, _inLength, _inRows, _inColumns));
    }

    @Override
    public void backPropagation(List<double[][]> dld0) {
        int outR = getOutputRows();
        int outC = getOutputColumns();

        List<double[][]> dxdl = new ArrayList<>(_inLength);
        for (int i = 0; i < _inLength; i++) {
            dxdl.add(new double[_inRows][_inColumns]);
        }

        IntStream.range(0, _inLength).parallel().forEach(l -> {
            double[][] error = dxdl.get(l);
            double[][] array = dld0.get(l);
            int[][] maxRows = _lastMaxRow[l];
            int[][] maxCols = _lastMaxColumn[l];

            for (int r = 0; r < outR; r++) {
                for (int c = 0; c < outC; c++) {
                    int max_i = maxRows[r][c];
                    int max_j = maxCols[r][c];

                    if (max_i != -1) {
                        error[max_i][max_j] += array[r][c];
                    }
                }
            }
        });

        if (_previousLayer != null) {
            _previousLayer.backPropagation(dxdl);
        }
    }

    @Override
    public void backPropagation(double[] dld0) {
        backPropagation(vectorToMatrix(dld0, getOutputLength(), getOutputRows(), getOutputColumns()));
    }

    @Override
    public int getOutputLength() {
        return _inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows - _windowSize) / _stepSize + 1;
    }

    @Override
    public int getOutputColumns() {
        return (_inColumns - _windowSize) / _stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return _inLength * getOutputColumns() * getOutputRows();
    }

    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {
        int size = input.size();
        List<double[][]> output = new ArrayList<>(size);
        for (int i = 0; i < size; i++)
            output.add(null);

        _lastMaxRow = new int[size][getOutputRows()][getOutputColumns()];
        _lastMaxColumn = new int[size][getOutputRows()][getOutputColumns()];

        IntStream.range(0, size).parallel().forEach(i -> {
            output.set(i, pool(input.get(i), i));
        });

        return output;
    }

    public double[][] pool(double[][] input, int index) {
        int outR = getOutputRows();
        int outC = getOutputColumns();
        double[][] output = new double[outR][outC];

        int[][] maxRows = _lastMaxRow[index];
        int[][] maxCols = _lastMaxColumn[index];

        for (int r = 0; r < outR; r++) {
            for (int c = 0; c < outC; c++) {
                double max = -Double.MAX_VALUE;
                int mRow = -1;
                int mCol = -1;

                for (int x = 0; x < _windowSize; x++) {
                    int inRow = r * _stepSize + x;
                    if (inRow >= _inRows)
                        break;

                    double[] inputRow = input[inRow];
                    for (int y = 0; y < _windowSize; y++) {
                        int inCol = c * _stepSize + y;
                        if (inCol >= _inColumns)
                            break;

                        double val = inputRow[inCol];
                        if (max < val) {
                            max = val;
                            mRow = inRow;
                            mCol = inCol;
                        }
                    }
                }
                output[r][c] = max;
                maxRows[r][c] = mRow;
                maxCols[r][c] = mCol;
            }
        }
        return output;
    }
}