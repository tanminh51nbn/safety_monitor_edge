import sys
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("Cần cài đặt onnx và onnxruntime: pip install onnx onnxruntime")
    sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Sử dụng: python quantize.py <input.onnx> <output_int8.onnx>")
        sys.exit(1)
    
    input_model = sys.argv[1]
    output_model = sys.argv[2]
    
    print(f"Đang tiến hành lượng tử hóa Dynamic Quantization INT8...\nNguồn: {input_model}\nĐích: {output_model}")
    
    try:
        quantize_dynamic(
            model_input=input_model,
            model_output=output_model,
            weight_type=QuantType.QUInt8
        )
        print("Đã hoàn tất! \nMô hình Lượng tử hóa giờ đã thu nhỏ kích thước 4 lần và đọc cực nhanh trên môi trường Nhúng.")
    except Exception as e:
        print(f"Lỗi rồi: {e}")

if __name__ == "__main__":
    main()
