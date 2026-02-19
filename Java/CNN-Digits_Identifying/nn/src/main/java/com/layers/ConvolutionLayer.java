package com.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.data.MatrixUtility;

@SuppressWarnings("all")
public class ConvolutionLayer extends Layer {

    private final long SEED;

    private List<double[][]> _filters;
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

        generateRandomFilters(_numFilters);
    }

    private void generateRandomFilters(int numFilters) {
        List<double[][]> filters = new ArrayList<>();
        Random random = new Random(SEED);

        for (int n = 0; n < numFilters; n++) {
            double[][] newFilter = new double[_filterSize][_filterSize];

            for (int i = 0; i < _filterSize; i++) {
                for (int j = 0; j < _filterSize; j++) {
                    double value = random.nextGaussian();
                    newFilter[i][j] = value;
                }
            }

            filters.add(newFilter);
        }

        _filters = filters;
    }

    public List<double[][]> convolutionForwardPass(List<double[][]> list) {
        _lastInput = list;

        List<double[][]> output = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            for (double[][] filter : _filters) {
                output.add(convolve(list.get(i), filter, _stepSize));
            }
        }
        return output;
    }

    private double[][] convolve(double[][] input, double[][] filter, int stepSize) {
        int outRows = (input.length - filter.length) / stepSize + 1;
        int outColumns = (input[0].length - filter[0].length) / stepSize + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fColumns = filter[0].length;

        double[][] output = new double[outRows][outColumns];

        int outRow = 0;
        int outColumn;

        for (int i = 0; i <= inRows - fRows; i += stepSize) {

            outColumn = 0;

            for (int j = 0; j <= inCols - fColumns; j += stepSize) {

                double sum = 0.0;

                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fColumns; y++) {

                        int inputRowIndex = i + x;
                        int inputColumnIndex = j + y;

                        double value = filter[x][y] * input[inputRowIndex][inputColumnIndex];
                        sum += value;
                    }
                }
                output[outRow][outColumn] = sum;
                outColumn++;
            }
            outRow++;
        }

        return output;
    }

    private double[][] fullConvolve(double[][] input, double[][] filter) {
        int outRows = (input.length + filter.length) + 1;
        int outColumns = (input[0].length + filter[0].length) + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fColumns = filter[0].length;

        double[][] output = new double[outRows][outColumns];

        int outRow = 0;
        int outColumn;

        for (int i = -fRows; i < inRows; i++) {

            outColumn = 0;

            for (int j = -fColumns; j < inCols; j++) {

                double sum = 0.0;

                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fColumns; y++) {

                        int inputRowIndex = i + x;
                        int inputColumnIndex = j + y;

                        if(inputRowIndex >= 0 && inputColumnIndex >= 0 && inputRowIndex < inRows && inputColumnIndex < inCols) {
                            double value = filter[x][y] * input[inputRowIndex][inputColumnIndex];
                            sum += value;
                        }
                    }
                }
                output[outRow][outColumn] = sum;
                outColumn++;
            }
            outRow++;
        }

        return output;
    }

    public double[][] spaceArray(double[][] input) {
        if (_stepSize == 1) {
            return input;
        }

        int outRows = (input.length - 1) * _stepSize + 1;
        int outColumns = (input[0].length - 1) * _stepSize + 1;

        double[][] output = new double[outRows][outColumns];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input.length; j++) {
                output[i * _stepSize][j * _stepSize] = input[i][j];
            }
        }

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = convolutionForwardPass(input);
        return _nextLayer.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixInput = vectorToMatrix(input, _inLength, _inRows, _inColumns);
        return getOutput(matrixInput);
    }

    @Override
    public void backPropagation(List<double[][]> dld0) {
        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]>  dld0PreviousLayer = new ArrayList<>();

        for (int f = 0; f < _filters.size(); f++) {
            filtersDelta.add(new double[_filterSize][_filterSize]);
        }

        for (int i = 0; i < _lastInput.size(); i++) {

            double[][] errorForInput = new double[_inRows][_inColumns];

            for (int f = 0; f < _filters.size(); f++) {

                double[][] currentFilter = _filters.get(f);
                double[][] error = dld0.get(i * _filters.size() + f);

                double[][] spacedError = spaceArray(error);
                double[][] dldf = convolve(_lastInput.get(i), spacedError, _stepSize);

                double[][] delta = MatrixUtility.multiply(dldf, _learningRate * -1);
                double[][] newTotalDelta = MatrixUtility.add(filtersDelta.get(f), delta);
                filtersDelta.set(f, newTotalDelta);

                double[][] flippedError = flipArrayHorizontal(flipArrayVertical(spacedError));
                errorForInput = MatrixUtility.add(errorForInput, fullConvolve(currentFilter, flippedError));
            }

            dld0PreviousLayer.add((errorForInput));
        }

        for (int f = 0; f < _filters.size(); f++) {
            double[][] modified = MatrixUtility.add(filtersDelta.get(f), _filters.get(f));
            _filters.set(f, modified);
        }

        if (_previousLayer != null) {
            _previousLayer.backPropagation((dld0PreviousLayer));
        }
    }

    public double[][] flipArrayHorizontal(double[][] array) {
        int rows = array.length;
        int columns = array[0].length;
        
        double[][] output = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                output[rows-i-1][j] = array[i][j];
            }
        }
        return output;
    }

    public double[][] flipArrayVertical(double[][] array) {
        int rows = array.length;
        int columns = array[0].length;
        
        double[][] output = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                output[i][columns-j-1] = array[i][j];
            }
        }
        return output;
    }

    @Override
    public void backPropagation(double[] dld0) {
        List<double[][]> matrixInput = vectorToMatrix(dld0, _inLength, _inRows, _inColumns);
        backPropagation(matrixInput);
    }

    @Override
    public int getOutputLength() {
        return _filters.size() * _inLength;
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
