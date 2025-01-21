# Plant Disease Detection System for Sustainable Agriculture 🌿🔍

An AI-powered system designed to detect and classify plant diseases using advanced machine learning and computer vision techniques. The project aims to support sustainable agriculture by enabling early disease detection, minimizing crop loss, and reducing the overuse of chemical pesticides.  

---

## 📝 About the Project  
This project leverages artificial intelligence, specifically Convolutional Neural Networks (CNNs), to identify plant diseases from images of leaves. It provides a practical solution to the challenges faced by farmers in diagnosing and managing crop diseases.  
### Features  
- Disease detection for multiple plant species and diseases.  
- User-friendly interface for farmers to upload leaf images for diagnosis.  
- Promotes sustainable agricultural practices by reducing chemical pesticide usage.  
- Real-time diagnosis with affordable deployment options.  

---

## 🛠️ Tech Stack  
- **Programming Language**: Python  
- **Frameworks and Libraries**: TensorFlow, Keras, OpenCV, Streamlit
- **Model**: CNN
- **Deployment Platform**: Streamlit Cloud  
---

## How It Works??
1. **Image Upload**: Users upload an image of a plant leaf through the system.  
2. **Preprocessing**: The image is resized, normalized, and enhanced for better analysis.  
3. **Feature Extraction**: Key features such as color, texture, and patterns are extracted.  
4. **Disease Classification**: The pre-trained CNN model analyzes the image and identifies the disease.  
5. **Results**: The system provides the diagnosis.  
---

## ⚙️ Setup Instructions  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/plant-disease-detection.git  
   cd plant-disease-detection

2. Install dependencies:
 ```bash
  pip install -r requirements.txt
 ```

3. Run the app locally:
 ```bash
  streamlit run app.py
 ```

## 📊 Dataset  

- **Details**:  
  - Contains labeled images of healthy and diseased leaves across various crops.  
  - Includes 87,000+ images for training and testing.
    
- **Preprocessing**:  
  - Images are resized to uniform dimensions.  
  - Data augmentation techniques (e.g., rotation, flipping) are applied.  

## 🙏 Acknowledgments  
- **Mentors**: Special thanks to Mr. P. Raja, Master Trainer, Edunet Foundation.  
- **Organizations**: Gratitude to AICTE, Microsoft, SAP, and Edunet Foundation for facilitating the internship.

 
