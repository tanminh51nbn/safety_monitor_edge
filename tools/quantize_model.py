import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def build_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_int8{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8 (dynamic).")
    parser.add_argument("--input", required=True, help="Path to input ONNX model.")
    parser.add_argument("--output", help="Path to output INT8 ONNX model.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input model not found: {input_path}")

    output_path = Path(args.output) if args.output else build_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    print(f"Saved INT8 model: {output_path}")


if __name__ == "__main__":
    main()
