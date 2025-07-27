#!/usr/bin/env python3
"""
Setup script for Medical Symptom Classifier
Run this script to set up the application environment
"""

import os
import sys
import subprocess
import nltk
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from nltk.stem.porter import PorterStemmer

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Error installing requirements")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✅ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ['models', 'templates', 'static']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created {directory}/ directory")
        else:
            print(f"📁 {directory}/ directory already exists")

def create_model():
    """Create and train the ML model"""
    print("🤖 Creating and training ML model...")
    
    stemmer = PorterStemmer()
    
    # Complete intents data (same as in app.py)
    intents = {
        'intents': [
            {'tag': 'abdominal_pain', 'patterns': ['I have abdominal pain', 'My abdomen hurts', 'I have pain in stomach', 'I feel pain in abdomen', 'stomach ache', 'belly pain']},
            {'tag': 'abnormal_menstruation', 'patterns': ['I have a heavy period', 'Heavy flow on my period', 'Period lasts longer than usual', 'My period is really painful', 'I have strong menstrual pain', 'Menstrual cramps are strong', 'irregular periods']},
            {'tag': 'acidity', 'patterns': ['I have acid reflux', 'I have acidity problems', 'I have heartburn', 'burning sensation in chest', 'sour taste in mouth']},
            {'tag': 'allergic_reaction', 'patterns': ['I have allergic reaction', 'I am allergic to something', 'skin rash from allergy', 'allergic symptoms', 'hives', 'itchy skin']},
            {'tag': 'arthritis', 'patterns': ['I have joint pain', 'My joints are stiff', 'arthritis pain', 'joint inflammation', 'stiff joints in morning']},
            {'tag': 'back_pain', 'patterns': ['I have back pain', 'My back hurts', 'lower back pain', 'upper back pain', 'spine pain', 'backache']},
            {'tag': 'bladder_discomfort', 'patterns': ['I have bladder pain', 'difficulty urinating', 'painful urination', 'bladder discomfort', 'burning while urinating']},
            {'tag': 'breathlessness', 'patterns': ['I have difficulty breathing', 'I feel breathless', 'shortness of breath', 'hard to breathe', 'breathing problems']},
            {'tag': 'bronchial_asthma', 'patterns': ['I have asthma', 'asthma attack', 'wheezing', 'chest tightness', 'bronchial problems']},
            {'tag': 'bruising', 'patterns': ['I have bruises', 'unexplained bruising', 'bruises on body', 'easy bruising', 'black and blue marks']},
            {'tag': 'burns', 'patterns': ['I have burns', 'burned skin', 'burn injury', 'thermal burn', 'skin burn']},
            {'tag': 'cervical_spondylosis', 'patterns': ['neck pain', 'cervical pain', 'stiff neck', 'neck stiffness', 'cervical spondylosis']},
            {'tag': 'chest_pain', 'patterns': ['I have chest pain', 'chest hurts', 'pain in chest', 'chest discomfort', 'sharp chest pain']},
            {'tag': 'chicken_pox', 'patterns': ['I have chicken pox', 'chickenpox symptoms', 'itchy blisters', 'pox marks', 'varicella']},
            {'tag': 'chills', 'patterns': ['I have chills', 'feeling cold', 'shivering', 'body chills', 'cold shivers']},
            {'tag': 'common_cold', 'patterns': ['I have cold', 'runny nose', 'sneezing', 'cold symptoms', 'nasal congestion', 'stuffy nose']},
            {'tag': 'constipation', 'patterns': ['I have constipation', 'difficulty passing stool', 'hard stool', 'infrequent bowel movements', 'blocked bowels']},
            {'tag': 'cough', 'patterns': ['I have cough', 'persistent cough', 'dry cough', 'wet cough', 'coughing', 'throat irritation']},
            {'tag': 'cramps', 'patterns': ['I have cramps', 'muscle cramps', 'stomach cramps', 'leg cramps', 'painful cramps']},
            {'tag': 'dengue', 'patterns': ['I have dengue', 'dengue fever', 'high fever with body pain', 'dengue symptoms', 'breakbone fever']},
            {'tag': 'diabetes', 'patterns': ['I have diabetes', 'high blood sugar', 'diabetic symptoms', 'frequent urination and thirst', 'diabetes mellitus']},
            {'tag': 'diarrhea', 'patterns': ['I have diarrhea', 'loose stools', 'frequent bowel movements', 'watery stool', 'stomach upset']},
            {'tag': 'dischromic_patches', 'patterns': ['I have skin patches', 'discolored skin', 'dark patches on skin', 'skin discoloration', 'pigmentation issues']},
            {'tag': 'dizziness', 'patterns': ['I feel dizzy', 'dizziness', 'lightheaded', 'vertigo', 'spinning sensation', 'balance problems']},
            {'tag': 'drug_reaction', 'patterns': ['I have drug reaction', 'medication side effects', 'allergic to medicine', 'drug allergy', 'adverse drug reaction']},
            {'tag': 'fatigue', 'patterns': ['I feel tired', 'fatigue', 'exhausted', 'weakness', 'lack of energy', 'feeling drained']},
            {'tag': 'fever', 'patterns': ['I have fever', 'high temperature', 'feverish', 'body heat', 'elevated temperature']},
            {'tag': 'fungal_infection', 'patterns': ['I have fungal infection', 'skin fungus', 'athlete foot', 'fungal rash', 'yeast infection']},
            {'tag': 'gastroenteritis', 'patterns': ['I have stomach flu', 'gastroenteritis', 'stomach infection', 'intestinal flu', 'stomach bug']},
            {'tag': 'gerd', 'patterns': ['I have GERD', 'acid reflux disease', 'chronic heartburn', 'gastroesophageal reflux', 'stomach acid problems']},
            {'tag': 'headache', 'patterns': ['I have headache', 'head pain', 'migraine', 'tension headache', 'severe headache', 'head hurts']},
            {'tag': 'heart_attack', 'patterns': ['I think I am having heart attack', 'chest pain radiating to arm', 'severe chest pain', 'heart attack symptoms', 'cardiac pain']},
            {'tag': 'hemorrhoids', 'patterns': ['I have hemorrhoids', 'piles', 'anal pain', 'rectal bleeding', 'swollen veins in rectum']},
            {'tag': 'hepatitis_a', 'patterns': ['I have hepatitis A', 'liver infection', 'jaundice', 'yellow eyes', 'hepatitis symptoms']},
            {'tag': 'hepatitis_b', 'patterns': ['I have hepatitis B', 'chronic liver disease', 'hepatitis B symptoms', 'liver inflammation']},
            {'tag': 'hepatitis_c', 'patterns': ['I have hepatitis C', 'chronic hepatitis', 'liver disease', 'hepatitis C symptoms']},
            {'tag': 'hepatitis_d', 'patterns': ['I have hepatitis D', 'delta hepatitis', 'hepatitis D symptoms']},
            {'tag': 'hepatitis_e', 'patterns': ['I have hepatitis E', 'acute hepatitis', 'hepatitis E symptoms']},
            {'tag': 'hypertension', 'patterns': ['I have high blood pressure', 'hypertension', 'elevated blood pressure', 'BP is high']},
            {'tag': 'hyperthyroidism', 'patterns': ['I have hyperthyroidism', 'overactive thyroid', 'thyroid problems', 'rapid heartbeat', 'weight loss']},
            {'tag': 'hypoglycemia', 'patterns': ['I have low blood sugar', 'hypoglycemia', 'sugar drop', 'low glucose']},
            {'tag': 'hypothyroidism', 'patterns': ['I have hypothyroidism', 'underactive thyroid', 'thyroid deficiency', 'slow metabolism']},
            {'tag': 'impetigo', 'patterns': ['I have impetigo', 'skin sores', 'crusty skin infection', 'bacterial skin infection']},
            {'tag': 'jaundice', 'patterns': ['I have jaundice', 'yellow skin', 'yellow eyes', 'liver problems', 'bilirubin high']},
            {'tag': 'malaria', 'patterns': ['I have malaria', 'malaria symptoms', 'fever with chills', 'mosquito bite fever']},
            {'tag': 'migraine', 'patterns': ['I have migraine', 'severe headache', 'migraine headache', 'throbbing head pain', 'light sensitivity']},
            {'tag': 'osteoarthritis', 'patterns': ['I have osteoarthritis', 'joint wear and tear', 'bone on bone pain', 'arthritis in joints']},
            {'tag': 'paralysis', 'patterns': ['I have paralysis', 'cannot move limbs', 'loss of movement', 'paralyzed', 'weakness in limbs']},
            {'tag': 'peptic_ulcer', 'patterns': ['I have peptic ulcer', 'stomach ulcer', 'ulcer pain', 'burning stomach pain']},
            {'tag': 'pneumonia', 'patterns': ['I have pneumonia', 'lung infection', 'chest infection', 'breathing difficulty with fever']},
            {'tag': 'psoriasis', 'patterns': ['I have psoriasis', 'scaly skin', 'skin plaques', 'red patches with scales']},
            {'tag': 'tuberculosis', 'patterns': ['I have tuberculosis', 'TB symptoms', 'persistent cough with blood', 'lung TB']},
            {'tag': 'typhoid', 'patterns': ['I have typhoid', 'typhoid fever', 'prolonged fever', 'enteric fever']},
            {'tag': 'urinary_tract_infection', 'patterns': ['I have UTI', 'urinary tract infection', 'burning urination', 'frequent urination']},
            {'tag': 'varicose_veins', 'patterns': ['I have varicose veins', 'swollen veins', 'bulging veins in legs', 'vein problems']},
            {'tag': 'red_spots_over_body', 'patterns': ['I have red spots on my body', 'I have small patches on the body', 'red rash all over', 'spotted skin']}
        ]
    }
    
    # Process data
    all_words = []
    tags = []
    xy = []
    
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))
    
    ignore_words = ['?', '.', '!', ',', "'", '"', ':', ';']
    all_words = sorted(set([stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]))
    tags = sorted(set(tags))
    
    def bag_of_words(tokenized_sentence, all_words):
        tokenized_sentence = [stemmer.stem(w.lower()) for w in tokenized_sentence if isinstance(w, str)]
        bag = np.zeros(len(all_words), dtype=np.float32)
        for idx, w in enumerate(all_words):
            if w in tokenized_sentence:
                bag[idx] = 1.0
        return bag
    
    # Create training data
    X_train = []
    y_train = []
    
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        
        label = tags.index(tag)
        y_train.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Create and train model
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=1000,
        alpha=0.01,
        solver='adam',
        random_state=42
    )
    
    print("🏋️ Training model...")
    model.fit(X_train, y_train)
    
    # Save model
    with open('fitted_model.pickle', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/fitted_model.pickle2', 'wb') as f:
        pickle.dump(model, f)
    
    print("✅ Model trained and saved successfully!")
    return True

def create_html_template():
    """Create the HTML template file"""
    print("🌐 Creating HTML template...")
    
    if not os.path.exists('templates/index.html'):
        # The HTML content is already provided in the artifacts above
        print("✅ Please copy the HTML template to templates/index.html")
    else:
        print("📄 HTML template already exists")

def run_tests():
    """Run basic tests to ensure everything works"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import flask
        import nltk
        import numpy
        import sklearn
        print("✅ All required packages imported successfully")
        
        # Test model loading
        if os.path.exists('fitted_model.pickle'):
            with open('fitted_model.pickle', 'rb') as f:
                model = pickle.load(f)
            print("✅ Model loaded successfully")
        else:
            print("❌ Model file not found")
            return False
            
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Medical Symptom Classifier...")
    print("=" * 50)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return
    
    # Step 3: Download NLTK data
    if not download_nltk_data():
        print("❌ Setup failed at NLTK data download")
        return
    
    # Step 4: Create and train model
    if not create_model():
        print("❌ Setup failed at model creation")
        return
    
    # Step 5: Create HTML template
    create_html_template()
    
    # Step 6: Run tests
    if not run_tests():
        print("❌ Setup completed but tests failed")
        return
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Copy the HTML template to templates/index.html")
    print("2. Run the application with: python app.py")
    print("3. Or for production: gunicorn app:app")
    print("\n📝 Don't forget to set up your deployment configuration!")

if __name__ == "__main__":
    main()