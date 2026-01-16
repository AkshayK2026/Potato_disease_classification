import torch

def predict(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        print("DEBUG probs:", probs) 
        confidence, pred = torch.max(probs, 1)

    return class_names[pred.item()], confidence.item()
