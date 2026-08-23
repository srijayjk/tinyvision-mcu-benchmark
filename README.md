# TinyVision MCU Benchmark

TinyVision MCU Benchmark is a project for evaluating tiny computer vision models on resource-constrained microcontroller platforms. The goal is to make it easy to compare model accuracy, latency, memory footprint, and energy efficiency across MCU-class devices using a consistent benchmarking workflow.

This repository is structured as a benchmark blueprint and reference project for running small vision workloads on embedded targets such as STM32, ESP32, and other Cortex-M based devices.

## Why this project?

Running vision models on microcontrollers is challenging because the real bottleneck is not only model size or FLOPs, but also:

- SRAM and flash constraints
- Inference latency under real-time limits
- Power consumption and thermal behavior
- Quantized model accuracy trade-offs
- Deployment complexity across different MCU families

This benchmark helps standardize these trade-offs so teams can compare implementations in a reproducible way.

## Benchmark goals

The benchmark focuses on measuring:

- Accuracy on representative vision tasks
- Inference time per frame
- Peak memory usage
- Flash usage
- Energy consumption
- Throughput under realistic workloads

Typical workloads may include:

- Image classification
- Object detection
- Segmentation
- Keyword or gesture recognition from visual inputs

## Supported targets

The project is designed to work with MCU-class hardware, including but not limited to:

- STM32 family devices
- ARM Cortex-M microcontrollers
- ESP32 / ESP32-S3 style targets
- Other low-power embedded vision platforms

The repository is intended to be portable and benchmarking-friendly rather than tied to a single vendor platform.

## Benchmark methodology

A typical evaluation run should include:

1. Model selection and quantization
2. Firmware build and code generation
3. On-device inference run
4. Measurement of runtime, power, and memory
5. Reporting of results in a standardized format

The benchmark should capture both model-level metrics and platform-level behavior.

## Example metrics

Each benchmark entry can report:

- Model name and architecture
- Input resolution
- Quantization format (INT8, FP16, etc.)
- Accuracy score
- Latency (ms/frame)
- Memory usage (RAM / Flash)
- Power draw or energy per inference
- Throughput (FPS)

## Repository structure

This repository is currently a foundation for the benchmark project. A typical layout may evolve into something like:

```text
.
├── README.md
├── models/
│   ├── tiny_classifiers/
│   └── quantized_models/
├── datasets/
│   └── benchmark_samples/
├── firmware/
│   ├── stm32/
│   └── esp32/
├── scripts/
│   ├── build.sh
│   ├── run_benchmark.py
│   └── report_results.py
├── results/
│   └── summary.csv
├── docs/
│   └── methodology.md
└── LICENSE
```

## Getting started

Clone the repository:

```bash
git clone https://github.com/srijayjk/tinyvision-mcu-benchmark.git
cd tinyvision-mcu-benchmark
```

Install dependencies for your chosen benchmarking workflow as needed, then add the model, firmware, and measurement scripts for your target platform.

## Suggested workflow

```bash
# 1. Prepare a model
# 2. Quantize it for MCU deployment
# 3. Build firmware for the target MCU
# 4. Run inference on-device
# 5. Log latency, memory, and accuracy
# 6. Publish a results table
```

## Contributing

Contributions are welcome for:

- Model collections and example deployments
- MCU target support
- Benchmark scripts and reporting
- Documentation and methodology improvements
- Results tables and reproducibility fixes

## Roadmap

Planned work may include:

- Standard benchmark definitions
- Reference datasets and sample images
- Quantized model zoo
- MCU result dashboards
- Cross-platform performance comparison tools

## License

This project is provided as an open benchmark repository. Add an appropriate license file if you intend to publish it publicly with specific usage terms.

## Notes

This repository is currently in its early stage. The long-term objective is to provide a clear, repeatable set of tools and metrics for evaluating tiny vision models on constrained embedded platforms in a transparent and comparable way.
