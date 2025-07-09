import os
import time

import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt


def benchmark_onnx_model(model_path, num_iterations=1000, providers=None):
    if providers is None:
        providers = ort.get_available_providers()

    results = {}
    for provider in providers:
        print(f"Benchmarking with provider: {provider}")
        try:
            sess = ort.InferenceSession(
                model_path,
                providers=[provider],
            )
            input_name = sess.get_inputs()[0].name
            input_shape = sess.get_inputs()[0].shape
            input_dtype = sess.get_inputs()[0].type

            dtype_map = {
                "tensor(float)": np.float32,
                "tensor(double)": np.float64,
                "tensor(int64)": np.int64,
                "tensor(int32)": np.int32,
                "tensor(int16)": np.int16,
                "tensor(int8)": np.int8,
                "tensor(uint8)": np.uint8,
                "tensor(bfloat16)": np.float16,
                "tensor(bool)": np.bool_,
            }

            input_data = np.random.rand(*input_shape).astype(dtype_map.get(input_dtype, np.float32))
            # Warm-up runs
            for _ in range(3):
                ort_inputs = {input_name: input_data}
                sess.run(None, ort_inputs)

            # Benchmarking
            timings = []
            for _ in range(num_iterations):
                ort_inputs = {input_name: input_data}
                start = time.perf_counter()
                sess.run(None, ort_inputs)
                end = time.perf_counter()
                duration = (end - start) * 1000  # Convert to milliseconds
                timings.append(duration)

            results[provider] = timings
        except Exception as e:
            print(f"Provider {provider} failed: {e}")
            results[provider] = None

    return results


def visualize_results(results, model_name):
    plt.figure(figsize=(10, 6))
    for provider, timings in results.items():
        if timings is not None:
            plt.plot(timings, label=f"{provider} (mean: {np.mean(timings):.2f} ms)")

    plt.xlabel("Iteration")
    plt.ylabel("Time (ms)")
    plt.title(f"ONNX Model Benchmarking Across Providers - {model_name}")
    plt.legend()
    plt.grid(True)
    # plt.show()

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    plt.savefig(f"outputs/{model_name}_benchmark_results.png")


if __name__ == "__main__":
    model_paths = [
        "conv_embeddings-sim.onnx",
        "linear_embeddings-sim.onnx",
        "linear_embeddings_image_input-sim.onnx",
    ]

    for model_path in model_paths:
        print(f"Benchmarking model: {model_path}")
        results = benchmark_onnx_model(model_path, num_iterations=100)

        for provider, timings in results.items():
            if timings is not None:
                mean_time = np.mean(timings).item()
                std_time = np.std(timings).item()
                print(f"Provider: {provider}, Average time: {mean_time:.2f} ms, Std: {std_time:.2f} ms")
            else:
                print(f"Provider: {provider} failed.")

        model_name = model_path.split("/")[-1]
        visualize_results(results, model_name)
        print("-" * 40)
        print()
