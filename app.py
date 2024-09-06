import gradio as gr
import torch
from torchvision import transforms
from torchvision.models import vit_b_16
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the class labels
class_labels = {
    0: 'Plum pox virus (PPV)',
    1: 'Prune dwarf virus (PDV)',
    2: 'Prunus necrotic ringspot virus (PNRSV)'
}

# Load the model
def load_model():
    try:
        model = vit_b_16(pretrained=False)
        num_classes = 3
        model.heads = torch.nn.Linear(model.heads[0].in_features, num_classes)
        model.load_state_dict(torch.load('vit_model.pth', map_location=torch.device('cpu')))
        model.eval()
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

model = load_model()

# Define the prediction function
def predict(image):
    if image is None:
        logging.warning("No image provided")
        return "No image provided"

    try:
        logging.info("Starting prediction process")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0)
        logging.info(f"Image transformed. Shape: {image.shape}")

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
        
        result = class_labels.get(class_idx, "Unknown class")
        logging.info(f"Prediction result: {result}")
        return result

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return f"An error occurred: {str(e)}"

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Text(),
    title="Prunus Viral Disease Classifier",
    description="Upload an image of a prunus leaf to classify the viral disease.",
)

# Launch the app
iface.launch()