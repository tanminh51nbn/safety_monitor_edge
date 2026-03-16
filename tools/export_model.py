from ultralytics import YOLO
model = YOLO("best.pt") 
model.export(format="onnx", opset=12, simplify=True)