# Brain Stroke Detection Using MRI/CT Scan

An automated deep learning system that analyzes brain MRI and CT scans to classify them as either **Normal** or **Stroke**. This project was developed as our 3rd-year mini-project.

We used an ensemble of two fine-tuned Convolutional Neural Networks (**VGG19** and **EfficientNetB4**) to achieve high accuracy and reliable predictions, assisting in early stroke detection.

---

## 📸 Demo & Results

Here is the performance of our ensemble model:

**Confusion Matrix**  
![Confusion Matrix](graphs/confusion_matrix.png)

**Classification Report**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 0.96 | 0.96 | 0.96 | 2966 |
| **Stroke** | 0.94 | 0.95 | 0.94 | 1968 |
| **Overall Accuracy** | | | **0.95** | 4934 |

---

## 🛠️ Technology Stack
- **Python 3.9+**
- **TensorFlow / Keras** (Deep Learning)
- **NumPy, Matplotlib, Scikit-learn, Pillow**

---

## 🚀 How to Run the Project (Cloning Steps)

If you'd like to run this project on your local machine, follow these steps:

### 1. Clone the Repository
Open your terminal and run the following command to download the project:
```bash
git clone https://github.com/Sarvesh2905/STROKE-DETECTION-USING-CT-MRI-SCAN.git
cd STROKE-DETECTION-USING-CT-MRI-SCAN
```

### 2. Set Up a Virtual Environment (Highly Recommended)
Create a clean environment so dependencies don't clash with your system.
```bash
# Create the virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Required Dependencies
Install everything needed for the models and notebook to run:
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
The dataset is excluded from this repository because of its large size.
- Download the **"Brain Stroke CT Image Dataset"** from Kaggle (or your data source).
- Extract it into a `dataset/` folder in the project root so it looks like this:
```
dataset/
├── Normal/
└── Stroke/
```

### 5. Run the Notebook
Launch Jupyter to explore the code, train models, or run evaluations:
```bash
jupyter notebook StrokeDetection.ipynb
```

---

## 🧠 Model Architecture

The system works by extracting features through two different networks and combining their strengths:

1. **VGG19**: Fine-tuned on the dataset (Weight: 0.55)
2. **EfficientNetB4**: Fine-tuned on the dataset (Weight: 0.45)
3. **Ensemble**: A weighted average of both predictions is taken to output the final result (**Normal** or **Stroke**).

---

## 👨‍💻 Authors

**SARVESH P**  
**[Teammate 1 Name]** *(Note: Please add your teammate's name here)*  
**[Teammate 2 Name]** *(Note: Please add your teammate's name here)*  

3rd Year B.Tech — Information Technology  
Dr. N.G.P Institute of Technology, Coimbatore

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
