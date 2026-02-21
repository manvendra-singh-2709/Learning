package digits;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class App {
    public static void main(String[] args) {
        // --- Path Verification Block ---
        System.out.println("--- Environment Verification ---");
        System.out.println("LD_LIBRARY_PATH: " + System.getenv("LD_LIBRARY_PATH"));
        System.out.println("PATH: " + System.getenv("PATH"));
        System.out.println("Java Lib Path: " + System.getProperty("java.library.path"));
        System.out.println("Force PTX JIT: " + System.getenv("CUDA_FORCE_PTX_JIT"));
        System.out.println("--------------------------------\n");

        try {
            System.out.println("Starting ND4J GPU Test...");
            INDArray matrixA = Nd4j.rand(new int[]{3, 3});
            System.out.println("Success! Matrix created on GPU.");
            System.out.println(matrixA);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}