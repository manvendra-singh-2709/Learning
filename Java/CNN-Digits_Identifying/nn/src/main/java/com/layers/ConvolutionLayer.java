package com.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

@SuppressWarnings("all")
public class ConvolutionLayer extends Layer {

    private final long SEED;
    private double[][] _filtersFlattened;
    private final int _filterSize;
    private final int _stepSize;
    private final int _inLength;
    private final int _inRows;
    private final int _inColumns;
    private final double _learningRate;
    private List<double[][]> _lastInput;

    public ConvolutionLayer(int _filterSize, int _stepSize, int _inLength, int _inRows,
            int _inColumns, long SEED, int _numFilters, double _learningRate) {
        this.SEED = SEED;
        this._filterSize = _filterSize;
        this._stepSize = _stepSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inColumns = _inColumns;
        this._learningRate = _learningRate;

        _filtersFlattened = new double[_numFilters][_filterSize * _filterSize];
        Random random = new Random(SEED);
        for (int n = 0; n < _numFilters; n++) {
            for (int i = 0; i < _filterSize * _filterSize; i++) {
                _filtersFlattened[n][i] = random.nextGaussian();
            }
        }
    }

    public List<double[][]> convolutionForwardPass(List<double[][]> list) {
        _lastInput = list;
        int outRows = getOutputRows();
        int outCols = getOutputColumns();
        int numFilters = _filtersFlattened.length;

        List<double[][]> output = new ArrayList<>(list.size() * numFilters);
        for (int i = 0; i < list.size() * numFilters; i++)
            output.add(null);

        IntStream.range(0, list.size()).parallel().forEach(i -> {
            double[][] input = list.get(i);
            for (int f = 0; f < numFilters; f++) {
                output.set(i * numFilters + f, convolveOptimized(input, _filtersFlattened[f], outRows, outCols));
            }
        });

        return output;
    }

    private double[][] convolveOptimized(double[][] input, double[] filterFlat, int outRows, int outCols) {
        double[][] output = new double[outRows][outCols];
        for (int i = 0; i < outRows; i++) {
            int rOffset = i * _stepSize;
            for (int j = 0; j < outCols; j++) {
                int cOffset = j * _stepSize;
                double sum = 0.0;
                for (int x = 0; x < _filterSize; x++) {
                    double[] row = input[rOffset + x];
                    int fOffset = x * _filterSize;
                    for (int y = 0; y < _filterSize; y++) {
                        sum += filterFlat[fOffset + y] * row[cOffset + y];
                    }
                }
                output[i][j] = sum;
            }
        }
        return output;
    }

    @Override
    public void backPropagation(List<double[][]> dld0) {
        int numFilters = _filtersFlattened.length;
        double[][] filtersDelta = new double[numFilters][_filterSize * _filterSize];
        List<double[][]> dld0PreviousLayer = new ArrayList<>(_lastInput.size());
        for (int i = 0; i < _lastInput.size(); i++)
            dld0PreviousLayer.add(new double[_inRows][_inColumns]);

        IntStream.range(0, _lastInput.size()).parallel().forEach(i -> {
            double[][] input = _lastInput.get(i);
            for (int f = 0; f < numFilters; f++) {
                double[][] error = dld0.get(i * numFilters + f);
                double[][] spacedError = spaceArray(error);

                double[][] dldf = convolve(input, spacedError, 1);
                double factor = -_learningRate;

                synchronized (filtersDelta[f]) {
                    for (int x = 0; x < _filterSize; x++) {
                        for (int y = 0; y < _filterSize; y++) {
                            filtersDelta[f][x * _filterSize + y] += dldf[x][y] * factor;
                        }
                    }
                }

                double[][] flippedError = flipAndFlipped(spacedError);
                double[][] fullConv = fullConvolveOptimized(_filtersFlattened[f], flippedError);

                double[][] prevLayerError = dld0PreviousLayer.get(i);
                synchronized (prevLayerError) {
                    for (int r = 0; r < _inRows; r++) {
                        for (int c = 0; c < _inColumns; c++) {
                            prevLayerError[r][c] += fullConv[r][c];
                        }
                    }
                }
            }
        });

        for (int f = 0; f < numFilters; f++) {
            for (int i = 0; i < _filtersFlattened[f].length; i++) {
                _filtersFlattened[f][i] += filtersDelta[f][i];
            }
        }

        if (_previousLayer != null) {
            _previousLayer.backPropagation(dld0PreviousLayer);
        }
    }

    private double[][] fullConvolveOptimized(double[] filterFlat, double[][] flippedError) {
        int outRows = _inRows;
        int outCols = _inColumns;
        double[][] output = new double[outRows][outCols];
        int fR = _filterSize;
        int fC = _filterSize;
        int eR = flippedError.length;
        int eC = flippedError[0].length;

        for (int i = 0; i < outRows; i++) {
            for (int j = 0; j < outCols; j++) {
                double sum = 0.0;
                for (int x = 0; x < fR; x++) {
                    int errIdxR = i - x + (fR - 1);
                    if (errIdxR >= 0 && errIdxR < eR) {
                        for (int y = 0; y < fC; y++) {
                            int errIdxC = j - y + (fC - 1);
                            if (errIdxC >= 0 && errIdxC < eC) {
                                sum += filterFlat[x * fC + y] * flippedError[errIdxR][errIdxC];
                            }
                        }
                    }
                }
                output[i][j] = sum;
            }
        }
        return output;
    }

    private double[][] flipAndFlipped(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[rows - i - 1][cols - j - 1] = array[i][j];
            }
        }
        return out;
    }

    private double[][] convolve(double[][] input, double[][] filter, int stepSize) {
        int outRows = (input.length - filter.length) / stepSize + 1;
        int outCols = (input[0].length - filter[0].length) / stepSize + 1;
        double[][] output = new double[outRows][outCols];
        for (int i = 0; i < outRows; i++) {
            for (int j = 0; j < outCols; j++) {
                double sum = 0.0;
                for (int x = 0; x < filter.length; x++) {
                    for (int y = 0; y < filter[0].length; y++) {
                        sum += filter[x][y] * input[i * stepSize + x][j * stepSize + y];
                    }
                }
                output[i][j] = sum;
            }
        }
        return output;
    }

    public double[][] spaceArray(double[][] input) {
        if (_stepSize == 1)
            return input;
        int outRows = (input.length - 1) * _stepSize + 1;
        int outCols = (input[0].length - 1) * _stepSize + 1;
        double[][] output = new double[outRows][outCols];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i * _stepSize][j * _stepSize] = input[i][j];
            }
        }
        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        return _nextLayer.getOutput(convolutionForwardPass(input));
    }

    @Override
    public double[] getOutput(double[] input) {
        return getOutput(vectorToMatrix(input, _inLength, _inRows, _inColumns));
    }

    @Override
    public void backPropagation(double[] dld0) {
        backPropagation(vectorToMatrix(dld0, _inLength, _inRows, _inColumns));
    }

    @Override
    public int getOutputLength() {
        return _filtersFlattened.length * _inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows - _filterSize) / _stepSize + 1;
    }

    @Override
    public int getOutputColumns() {
        return (_inColumns - _filterSize) / _stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputRows() * getOutputColumns() * getOutputLength();
    }
}