import pickle

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

def load_model():
    with open("model/potato_model.pkl", "rb") as f:
        model = pickle.load(f)
    model.eval()
    return model
