from tools.base_models.yolov8_inferencer import Yolov8_Inferencer
from tools.base_models.yolov7.yolov7_inferencer import Yolov7_Inferencer

# More models will be added here      |
#                                     v
class Make_Inference(Yolov8_Inferencer, Yolov7_Inferencer):
    def __init__(self, model, weight_path, source, size, save, save_txt, conf, stream):
        self.model = model
        super().__init__(weight_path, source, size, save, save_txt, conf, stream)
        
    
    def predict(self):
        if self.model == 'yolov8':
            super().yolov8_inference()
        
        elif self.model == 'yolov7':
            super().yolov7_inference()
        
        else:
            pass
            # More models will be added here

    def get_result_info(self):
        if self.model == 'yolov8':
            results = super().yolov8_inference_with_results()

        elif self.model == 'yolov7':
            print('yolov7_inference_with_results girdim')
            results = super().yolov7_inference_with_results()

        return results