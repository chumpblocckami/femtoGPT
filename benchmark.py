import os
import tracemalloc
import time
import psutil


def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Memory usage in MB


def cpu_time():
    return time.process_time()  # CPU time in seconds


def compile_time(func):
    start = time.time()
    func()
    end = time.time()
    return end - start  # Compilation time in seconds


def benchmark(func):
    # Measure memory usage before function execution
    start_memory = memory_usage()

    # Measure compilation time
    start_compile = compile_time(func)

    # Measure CPU time
    start_cpu = cpu_time()

    # Run the function
    func()

    # Measure memory usage after function execution
    end_memory = memory_usage()

    # Measure CPU time after function execution
    end_cpu = cpu_time()

    # Print benchmark results
    print("Memory Used: {} MB".format(end_memory - start_memory))
    print("Compilation Time: {:.6f} seconds".format(start_compile))
    print("CPU Time: {:.6f} seconds".format(end_cpu - start_cpu))


if __name__ == "__main__":
    from debugger import benchmark_generate

    tracemalloc.start()  # Enable tracing memory allocation
    benchmark(benchmark_generate)
