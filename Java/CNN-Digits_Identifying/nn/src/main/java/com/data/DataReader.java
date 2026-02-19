package com.data;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {
    private final int rows;
    private final int columns;

    public DataReader(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
    }

    public List<Image> readFromResources(String resourcePath) {
        List<Image> images = new ArrayList<>();

        try (BufferedReader dataReader = new BufferedReader(
                new InputStreamReader(DataReader.class.getClassLoader().getResourceAsStream(resourcePath)))) {

            String line;

            while ((line = dataReader.readLine()) != null) {
                String[] lineItems = line.split(",");

                double[][] data = new double[rows][columns];
                int label = Integer.parseInt(lineItems[0]);

                int i = 1;
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < columns; col++) {
                        data[row][col] = Integer.parseInt(lineItems[i]);
                        i++;
                    }
                }

                images.add(new Image(data, label));
            }

        } catch (Exception e) {
            throw new RuntimeException("Failed to load resource: " + resourcePath, e);
        }
        return images;
    }

}
