import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from model_service import predict_disease

app = Flask(__name__)
CORS(app)

# 🔹 IMPORTANT: This list must match the order of your training labels!
# Based on common datasets used for this project:
DISEASE_NAMES = [
    "Anxiety Disorder", "Depression", "Asthma", "Heart Disease", "GERD", 
    "Migraine", "Diabetes", "Digestive System Disorder", "Urinary Tract Infection",
    "Fungal infection", "Allergy", "Drug Reaction", "Peptic ulcer disease", 
    "AIDS", "Gastroenteritis", "Hypertension", "Malaria", "Chicken pox", 
    "Dengue", "Typhoid", "Hepatitis A", "Hepatitis B", "Hepatitis C", 
    "Tuberculosis", "Common Cold", "Pneumonia", "Arthritis", "Acne", "Psoriasis"
    # Note: Ensure this list matches the number of output classes in your model (dense_2)
]

@app.route('/')
def home():
    return "Health Prediction API is running."
     

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Extract data from React request
        data = request.json.get('features')

        # 2. Validation
        if not data or len(data) != 230:
            return jsonify({
                "error": f"Expected 230 features, got {len(data) if data else 0}"
            }), 400
            
        # 3. Get Prediction from Model
        # predict_disease returns the array of probabilities (e.g., [[0.1, 0.8, 0.1...]])
        # or the class index depending on your model_service implementation.
        raw_prediction = predict_disease(data) 
        
        # 4. Process the results
        # Assuming your model_service returns probabilities from MODEL.predict()
        # If it already returns the index, you can skip the np.argmax
        class_index = int(np.argmax(raw_prediction))
        confidence = float(np.max(raw_prediction))

        # 5. Map index to Disease Name
        if class_index < len(DISEASE_NAMES):
            predicted_disease = DISEASE_NAMES[class_index]
        else:
            predicted_disease = f"Unknown Condition (Class {class_index})"

        # 6. Return response to React
        return jsonify({
            "disease": predicted_disease,
            "confidence": confidence,
            "class_index": class_index
        })

    except Exception as e:
        print(f"🔥 Backend Crash: {e}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)