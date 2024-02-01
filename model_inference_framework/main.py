from tools.inferencer.make_inference import Make_Inference
from tools.divide2bb.divide2bb import Divide2BB
import argparse



parser = argparse.ArgumentParser(description='This script let you do inference and divide2bb tasks for multiple models.')
parser.add_argument('--task', type=str, required=True, help='Task to be done (inference, divide2bb).')
parser.add_argument('--model', type=str, required=True, help='The model in which the task will be performed (yolov8, yolov7).')
parser.add_argument('--weight_path', type=str, required=False, help='Path to .pt or .pth file')
parser.add_argument('--source', type=str, required=False, help='Image or folder location')
parser.add_argument('--size', type=int, default=640, required=False, help='Size of the image where the inference will be made (default:640)')
parser.add_argument('--save', action='store_true', required=False, help='Save result image (default:True)')
parser.add_argument('--save_txt', action='store_true', required=False, help='Save result annotations (default:True)')
parser.add_argument('--conf', type=float, default=0.25, required=False, help='Confidence ratio (default:0.25)')
parser.add_argument('--stream', action='store_true', required=False, help='Stream flag for yolov8. Use if inference dataset is big')
args = parser.parse_args()



if args.task == 'inference':
    inferencer = Make_Inference(args.model, args.weight_path, args.source, args.size, args.save, args.save_txt, args.conf, args.stream)
    inferencer.predict()

elif args.task == 'divide2bb':
    inferencer = Make_Inference(args.model, args.weight_path, args.source, args.size, args.save, args.save_txt, args.conf)
    results = inferencer.get_result_info()

    divider = Divide2BB(results)
    divider.divide_images()

else:
    pass
    # More task will be added here

