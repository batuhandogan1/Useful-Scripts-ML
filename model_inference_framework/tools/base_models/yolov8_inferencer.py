from ultralytics import YOLO

class Yolov8_Inferencer():
    def __init__(self, weight_path, source, size, save, save_txt, conf, stream):
        self.weight_path = weight_path
        self.size = size
        self.source = source
        self.save = save
        self.save_txt = save_txt
        self.conf = conf
        self.stream = stream

    
    def yolov8_inference(self):
        model = YOLO(self.weight_path)
        model.predict(source=self.source, imgsz=self.size, save=self.save, save_txt=self.save_txt, conf=self.conf)

        
    def yolov8_inference_with_results(self):
        model = YOLO(self.weight_path)
        results = model(source=self.source, imgsz=self.size, conf=self.conf)

        return results
