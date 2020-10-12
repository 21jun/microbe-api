
import io
import json
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18:

    def __init__(self):
        self.microbe_class_index = json.load(
            open('oasis/model/microbe_class_index.json'))

        PATH = "save/resnet18_colab.model"
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, 15, bias=True)
        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()

    def transform_image(self, image_bytes):
        my_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.5, 0.5, 0.5],
                                                [0.5, 0.5, 0.5])])
        image = Image.open(io.BytesIO(image_bytes))
        return my_transforms(image).unsqueeze(0)

    def get_prediction(self, image_bytes):
        tensor = self.transform_image(image_bytes=image_bytes)
        outputs = self.model.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())

        softmax = nn.Softmax(dim=1)
        p = softmax(outputs)
        # print(p[0])

        confidence = {}
        for prob, microbe in zip(p[0], self.microbe_class_index.values()):
            # print(str(prob.item()), microbe[1])
            confidence[microbe[1]] = "{:.8f}".format(float(prob.item()*100))

        # print(confidence)

        return self.microbe_class_index[predicted_idx], confidence
