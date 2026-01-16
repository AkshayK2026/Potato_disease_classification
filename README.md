# 🥔 Potato Leaf Disease Classification System

This project is an end-to-end deep learning application that classifies potato leaf diseases using a pretrained CNN model.  
It uses **PyTorch** for model training, **FastAPI** for backend API, and **Streamlit** for the user interface.

---

## 🚀 Features
- Classifies potato leaf images into disease categories
- Uses transfer learning with a pretrained CNN (ResNet-18)
- Image preprocessing and data augmentation
- REST API using FastAPI
- Interactive web UI using Streamlit

---

## 🧠 Model
- Architecture: ResNet-18 (pretrained)
- Framework: PyTorch
- Dataset: PlantVillage (Potato leaf images)

---



1.Create and Activate Virtual Environment
python -m venv env
env\Scripts\activate

2.Install Dependencies
#pip install -r requirements.txt

3.How to Run the Project
Run FastAPI Backend
#uvicorn backend.main:app --reload
Run Streamlit Frontend
streamlit run frontend/app.py


