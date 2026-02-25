package dlj;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.Shape;

public class Main {

	// Benchmark dot product on a given device
	private static long benchmark(Device device, int size, int warmup, int runs) {

		try (NDManager manager = NDManager.newBaseManager(device)) {

			System.out.println("\nRunning on: " + device);

			// Warmup (important for GPU)
			for (int i = 0; i < warmup; i++) {
				NDArray a = manager.randomUniform(0f, 1f, new Shape(size, size));
				NDArray b = manager.randomUniform(0f, 1f, new Shape(size, size));
				a.dot(b).toDevice(Device.cpu(), true); // sync GPU
			}

			long total = 0;

			for (int i = 0; i < runs; i++) {
				NDArray a = manager.randomUniform(0f, 1f, new Shape(size, size));
				NDArray b = manager.randomUniform(0f, 1f, new Shape(size, size));

				long start = System.currentTimeMillis();
				NDArray c = a.dot(b);

				// FORCE sync GPU
				c.toDevice(Device.cpu(), true);

				long end = System.currentTimeMillis();

				long time = end - start;
				total += time;

				System.out.println("Run " + (i + 1) + " time: " + time + " ms | checksum: " + c.getFloat(0));
			}

			return total / runs;
		}
	}

	public static void main(String[] args) {

		int size = 3000; // adjust for bigger matrices
		int warmup = 2;
		int runs = 3;
		System.out.println("DJL cache: " + System.getenv("DJL_CACHE_DIR"));

		System.out.println("Engine: " + Engine.getInstance().getEngineName());
		System.out.println("Default Device: " + Engine.getInstance().defaultDevice());
		System.out.println("GPU count: " + Engine.getInstance().getGpuCount());

		// CPU benchmark
		long cpuTime = benchmark(Device.cpu(), size, warmup, runs);
		System.out.println("\nCPU average time: " + cpuTime + " ms");

		// GPU benchmark
		if (Engine.getInstance().getGpuCount() > 0) {
			long gpuTime = benchmark(Device.gpu(), size, warmup, runs);
			System.out.println("\nGPU average time: " + gpuTime + " ms");

			double speedup = (double) cpuTime / gpuTime;
			System.out.println("\nSpeedup: " + speedup + "x");
		} else {
			System.out.println("\nNo GPU detected by DJL.");
		}
	}
}