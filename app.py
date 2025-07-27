# app.py - Integration for Multiple Models (NN, Decision Tree, SVM, Logistic Regression)

from flask import Flask, request, render_template
import pickle
import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
import os
import warnings
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

stemmer = PorterStemmer()

class MultiModelIntegrator:
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.vectorizer = None
        self.label_encoder = None
        self.scaler = None
        self.feature_names = []
        self.class_names = []
        self.model_weights = {}
        self.is_loaded = False
        
    def load_individual_models(self):
        """Load individual models separately"""
        model_files = {
            'neural_network': 'models/neural_network.pickle',
            'decision_tree': 'models/decision_tree.pickle', 
            'svm': 'models/svm_model.pickle',
            'logistic_regression': 'models/logistic_regression.pickle'
        }
        
        loaded_models = {}
        
        for model_name, model_path in model_files.items():
            try:
                if os.path.exists(model_path):
                    # Try pickle first
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        print(f"✅ Loaded {model_name} with pickle")
                    except:
                        # Try joblib
                        model = joblib.load(model_path)
                        print(f"✅ Loaded {model_name} with joblib")
                    
                    loaded_models[model_name] = model
                else:
                    print(f"⚠️ Model file not found: {model_path}")
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
        
        self.models = loaded_models
        return len(loaded_models) > 0
    
    def load_ensemble_model(self):
        """Load ensemble/voting classifier if you saved it as one model"""
        ensemble_paths = [
            'models/ensemble_model.pickle',
            'models/voting_classifier.pickle',
            'models/combined_model.pickle',
            'models/fitted_model.pickle2'  # Your original path
        ]
        
        for path in ensemble_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.ensemble_model = pickle.load(f)
                    print(f"✅ Loaded ensemble model from {path}")
                    return True
                except Exception as e:
                    print(f"❌ Error loading ensemble from {path}: {e}")
        
        return False
    
    def load_preprocessing_components(self):
        """Load vectorizer, scaler, label encoder etc."""
        
        # Load vectorizer (TF-IDF or CountVectorizer)
        vectorizer_paths = [
            'models/tfidf_vectorizer.pickle',
            'models/count_vectorizer.pickle',
            'models/vectorizer.pickle'
        ]
        
        for path in vectorizer_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.vectorizer = pickle.load(f)
                    print(f"✅ Loaded vectorizer from {path}")
                    break
                except Exception as e:
                    print(f"❌ Error loading vectorizer from {path}: {e}")
        
        # Load label encoder
        encoder_paths = [
            'models/label_encoder.pickle',
            'models/encoder.pickle'
        ]
        
        for path in encoder_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.label_encoder = pickle.load(f)
                    print(f"✅ Loaded label encoder from {path}")
                    break
                except Exception as e:
                    print(f"❌ Error loading label encoder from {path}: {e}")
        
        # Load scaler (for neural network)
        scaler_paths = [
            'models/scaler.pickle',
            'models/standard_scaler.pickle'
        ]
        
        for path in scaler_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    print(f"✅ Loaded scaler from {path}")
                    break
                except Exception as e:
                    print(f"❌ Error loading scaler from {path}: {e}")
        
        # Load class names if saved separately
        if os.path.exists('models/class_names.pickle'):
            try:
                with open('models/class_names.pickle', 'rb') as f:
                    self.class_names = pickle.load(f)
                print("✅ Loaded class names")
            except Exception as e:
                print(f"❌ Error loading class names: {e}")
    
    def create_fallback_vectorizer(self):
        """Create a basic vectorizer if none was found"""
        if self.vectorizer is None:
            print("🔧 Creating fallback TF-IDF vectorizer...")
            
            # Create sample medical vocabulary
            medical_terms = [
                'pain', 'ache', 'hurt', 'fever', 'headache', 'nausea', 'dizzy', 'tired',
                'cough', 'cold', 'flu', 'stomach', 'chest', 'back', 'joint', 'muscle',
                'breathing', 'shortness', 'breath', 'heart', 'blood', 'pressure'
            ]
            
            # Create basic corpus for fitting
            sample_texts = [
                "I have severe headache and feel dizzy",
                "My stomach hurts with nausea",
                "I have chest pain and breathing problems",
                "Back pain and muscle aches",
                "High fever with chills"
            ]
            
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Fit on sample texts
            self.vectorizer.fit(sample_texts)
            print("✅ Fallback vectorizer created")
    
    def initialize_models(self):
        """Initialize all model components"""
        print("🚀 Initializing Multi-Model System...")
        
        # Try to load ensemble model first
        if self.load_ensemble_model():
            print("✅ Using ensemble model")
        else:
            # Load individual models
            if self.load_individual_models():
                print("✅ Using individual models")
            else:
                print("❌ No models found!")
                return False
        
        # Load preprocessing components
        self.load_preprocessing_components()
        
        # Create fallback vectorizer if needed
        if self.vectorizer is None:
            self.create_fallback_vectorizer()
        
        self.is_loaded = True
        return True
    
    def preprocess_text(self, text):
        """Preprocess input text for prediction"""
        # Basic text cleaning
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Vectorize text
        if self.vectorizer:
            vectorized = self.vectorizer.transform([text])
            
            # Scale if scaler available (typically for neural network)
            if self.scaler:
                vectorized = self.scaler.transform(vectorized.toarray())
            
            return vectorized
        else:
            # Fallback to bag of words if no vectorizer
            return self.create_bag_of_words(text)
    
    def create_bag_of_words(self, text):
        """Fallback bag of words approach"""
        # Tokenize
        tokens = nltk.word_tokenize(text)
        tokens = [stemmer.stem(token.lower()) for token in tokens if token.isalnum()]
        
        # Create basic feature vector (this is a simple fallback)
        medical_keywords = [
            'pain', 'ache', 'hurt', 'fever', 'headache', 'nausea', 'dizzy', 'tired',
            'cough', 'cold', 'flu', 'stomach', 'chest', 'back', 'joint', 'muscle'
        ]
        
        features = []
        for keyword in medical_keywords:
            features.append(1 if keyword in tokens else 0)
        
        return np.array(features).reshape(1, -1)
    
    def predict_with_ensemble(self, text):
        """Predict using ensemble model"""
        try:
            # Preprocess text
            processed_input = self.preprocess_text(text)
            
            # Make prediction
            prediction = self.ensemble_model.predict(processed_input)[0]
            
            # Get confidence if available
            if hasattr(self.ensemble_model, 'predict_proba'):
                probabilities = self.ensemble_model.predict_proba(processed_input)[0]
                confidence = max(probabilities)
            else:
                confidence = 1.0
            
            # Decode prediction if label encoder exists
            if self.label_encoder:
                predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            else:
                predicted_class = str(prediction)
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"Ensemble prediction error: {e}")
            return "Error in prediction", 0.0
    
    def predict_with_individual_models(self, text):
        """Predict using individual models and combine results"""
        try:
            # Preprocess text
            processed_input = self.preprocess_text(text)
            
            predictions = {}
            confidences = {}
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    # Special handling for different model types
                    if model_name == 'neural_network' and self.scaler:
                        # Neural networks often need scaled input
                        input_data = processed_input.toarray() if hasattr(processed_input, 'toarray') else processed_input
                        if self.scaler:
                            input_data = self.scaler.transform(input_data)
                        pred = model.predict(input_data)[0]
                    else:
                        pred = model.predict(processed_input)[0]
                    
                    predictions[model_name] = pred
                    
                    # Get confidence if available
                    if hasattr(model, 'predict_proba'):
                        if model_name == 'neural_network' and self.scaler:
                            proba = model.predict_proba(input_data)[0]
                        else:
                            proba = model.predict_proba(processed_input)[0]
                        confidences[model_name] = max(proba)
                    else:
                        confidences[model_name] = 1.0
                        
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    continue
            
            if not predictions:
                return "No valid predictions", 0.0
            
            # Combine predictions (voting approach)
            # Method 1: Simple majority voting
            prediction_counts = {}
            for pred in predictions.values():
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            # Get most common prediction
            final_prediction = max(prediction_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate average confidence
            avg_confidence = np.mean(list(confidences.values()))
            
            # Decode prediction if label encoder exists
            if self.label_encoder:
                try:
                    predicted_class = self.label_encoder.inverse_transform([final_prediction])[0]
                except:
                    predicted_class = str(final_prediction)
            else:
                predicted_class = str(final_prediction)
            
            return predicted_class, avg_confidence
            
        except Exception as e:
            print(f"Individual models prediction error: {e}")
            return "Error in prediction", 0.0
    
    def predict(self, text):
        """Main prediction method"""
        if not self.is_loaded:
            return "Model not loaded", 0.0
        
        # Use ensemble if available, otherwise use individual models
        if self.ensemble_model:
            return self.predict_with_ensemble(text)
        elif self.models:
            return self.predict_with_individual_models(text)
        else:
            return "No models available", 0.0

# Initialize the multi-model integrator
model_integrator = MultiModelIntegrator()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symptoms = request.form['symptoms']
        if not symptoms.strip():
            return render_template('index.html', 
                                 prediction="Please enter your symptoms", 
                                 symptoms=symptoms)
        
        # Use the multi-model integrator for prediction
        predicted_class, confidence = model_integrator.predict(symptoms)
        
        # Format the prediction
        if predicted_class != "Error in prediction" and predicted_class != "No models available":
            formatted_prediction = predicted_class.replace('_', ' ').title()
        else:
            formatted_prediction = predicted_class
        
        if confidence < 0.3:  # Adjust threshold as needed
            formatted_prediction = "Unable to determine - please consult a healthcare professional"
        
        return render_template('index.html', 
                             prediction=formatted_prediction, 
                             symptoms=symptoms,
                             confidence=f"{confidence:.2%}")
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return render_template('index.html', 
                             prediction="Error processing your request. Please try again.", 
                             symptoms=request.form.get('symptoms', ''))

@app.route('/health')
def health():
    return {
        "status": "healthy", 
        "ensemble_model_loaded": model_integrator.ensemble_model is not None,
        "individual_models_loaded": len(model_integrator.models),
        "vectorizer_loaded": model_integrator.vectorizer is not None,
        "models_available": list(model_integrator.models.keys())
    }

@app.route('/model_info')
def model_info():
    """Endpoint to get information about loaded models"""
    return {
        "ensemble_model": str(type(model_integrator.ensemble_model)) if model_integrator.ensemble_model else None,
        "individual_models": {name: str(type(model)) for name, model in model_integrator.models.items()},
        "vectorizer": str(type(model_integrator.vectorizer)) if model_integrator.vectorizer else None,
        "label_encoder": str(type(model_integrator.label_encoder)) if model_integrator.label_encoder else None,
        "scaler": str(type(model_integrator.scaler)) if model_integrator.scaler else None,
        "is_loaded": model_integrator.is_loaded
    }

if __name__ == '__main__':
    # Initialize the multi-model system
    if not model_integrator.initialize_models():
        print("❌ Failed to initialize models. Please check your model files.")
        print("📂 Expected model files:")
        print("   - models/neural_network.pickle")
        print("   - models/decision_tree.pickle") 
        print("   - models/svm_model.pickle")
        print("   - models/logistic_regression.pickle")
        print("   OR")
        print("   - models/ensemble_model.pickle (if you saved as ensemble)")
        exit(1)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)