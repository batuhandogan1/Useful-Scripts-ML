import cv2
import os

class Divide2BB():
    def __init__(self, results):
        self.bboxes = []
        self.paths = []

        for result in results:
            self.bboxes.append(result.boxes.xyxy.cpu().numpy())
            self.paths.append(result.path)

    def get_bboxes(self):
        return self.bboxes
    

    def get_pathes(self):
        return self.paths
    

    def get_classes(self):
        pass


    def divide_images(self):
        for index in range(len(self.paths)):

            if self.bboxes[index].any() != None:
                for bb in range(len(self.bboxes[index])):

                    image = cv2.imread(self.paths[index])
                    cropped_image = image[int(self.bboxes[index][bb][1]) : int(self.bboxes[index][bb][3]),
                                          int(self.bboxes[index][bb][0]) : int(self.bboxes[index][bb][2])]
                    
                    if not os.path.exists(os.path.join(os.getcwd(), 'output_images')):
                        os.makedirs(os.path.join(os.getcwd(), 'output_images'))

                    image_name = os.path.split(self.paths[index])[-1]

                    cv2.imwrite('./output_images/' + str(bb) + image_name, cropped_image)