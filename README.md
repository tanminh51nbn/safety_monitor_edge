# Safety Monitor Edge AI

## Phase 2 checklist (profiling + quantization)
- [ ] Warm-up 30 frames to stabilize timings
- [ ] Log avg + p95 latency for resize, layout, inference, postprocess, total
- [ ] Track proposal count before/after NMS
- [ ] Log FPS + process RSS/CPU every 1s
- [ ] Quantize model (INT8) and compare FP32 vs INT8

## Profiling output (runtime)
- `[Profiling]` lines show avg/p95 (ms) per stage
- `[Benchmark]` lines show FPS + RSS (MB) + CPU (%)

## Quantization (INT8)
```bash
python tools/quantize_model.py --input models/best.onnx --output models/best_int8.onnx
```

Run with INT8 model:
```bash
MODEL_PATH=models/best_int8.onnx cargo run --release
```

## Benchmark results (fill in)

| Model | FPS | RSS (MB) | Notes |
| --- | --- | --- | --- |
| FP32 | TBD | TBD | baseline |
| INT8 | TBD | TBD | dynamic quantization |
