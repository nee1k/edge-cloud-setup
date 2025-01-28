import torch
from PIL import Image
from torchvision import transforms
from torchvision import models

model_mapping = {
    "d16e72acdf64baf2f9b8e2276c927535e9278d5cd3202d2eef943c1d0a60d84d-model": "googlenet",
    "e0e46c4c246a0a9d88c9cbbbb9a4390600abe071c4d532a4b28e7cb8e20b08d6-model": "resnet152",
    "665fdecddef3647b5882051d4b852b6ec075d4530d548a954dbb6d5e69286c5d-model": "shufflenet_v2_x0_5",
    "86c48c6b1eca331e9c745e5d036480f4706a83b7c08ecc5cbf44a4fa12d19237-model": "densenet201",
    "922a64a4094a6dc421bdb875d3bd533a56352f1cdb2a3f9600534213fabc5b2f-model": "mobilenet_v3_small",
    "38db957b18993eba7120b2e55f02dde7c4cb39a5bc6bd056bfe0628a29129e11-model": "resnext50_32x4d"
}

class ModelStore:
    def __init__(self):
        self.model_index = 0
        self.current_model_id = "e0e46c4c246a0a9d88c9cbbbb9a4390600abe071c4d532a4b28e7cb8e20b08d6-model"

    def get_current_model_id(self):
        return self.current_model_id

    # loading the model
    # model = models.squeezenet1_1(weights="SqueezeNet1_1_Weights.IMAGENET1K_V1")
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'convnext_small', pretrained=True)
    # model = models.resnet50(weights="IMAGENET1K_V2")
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'convnext', pretrained=True)

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x0_5', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    # model = models.regnet_y_128gf(weights="IMAGENET1K_SWAG_E2E_V1")
    # MobileNet_V3_Small

    model.eval()

    def change_model(self, model_name):
        if model_name == 'regnet':
            self.model = models.regnet_y_128gf(weights="IMAGENET1K_SWAG_E2E_V1")
            print("Regnet requested...")
        else:
            self.model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        self.model.eval()

    def load_next_model(self):
        """
        Loads the next model in the model mapping.
        If all models have been loaded, it starts from the beginning.
        """

        # calculate the next model id and model name
        model_keys = list(model_mapping.keys())
        new_model_index = (self.model_index + 1) % len(model_keys)

        # get the next model id and name
        new_model_id = model_keys[new_model_index]
        new_model_name = model_mapping[new_model_id]

        # Print the model information
        print(f"Loading model with UUID: {new_model_id} -> Model: {new_model_name}")
        self.change_model(new_model_name)

        # Update the index for the next call
        self.model_index = new_model_index
        self.current_model_id = new_model_id

        return new_model_name, new_model_id

    def load_model(self, model_name):
        self.change_model(model_name)


model_store = ModelStore()


# retrieving the class label
with open("imagenet_classes.txt", "r") as f:
    labels = [s.strip() for s in f.readlines()]


def pre_process(filename):
    """
    Pre-processes the image to allow the image to be fed into the PyTorch model.
    :param filename: Path to the image file.
    :return: Pre-processed image tensor.
    """
    input_image = Image.open(filename).convert("RGB")  # Ensure the image is in RGB format
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def predict(input):
    """
    Predicting the class for a given pre-processed input
    :param input:
    :return: prediction class
    """
    with torch.no_grad():
        output = model_store.model(input)
    prob = torch.nn.functional.softmax(output[0], dim=0)

    # retrieve top probability for the input
    high_prob, pred_label = torch.topk(prob, 1)

    return str((labels[pred_label[0]])), high_prob[0].item()
