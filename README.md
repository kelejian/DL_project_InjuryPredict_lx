# **Pre-Crash Occupant Injury Prediction Model based on Knowledge Distillation**

A PyTorch implementation of a pre-crash occupant injury prediction model that leverages knowledge distillation to balance accuracy and computational efficiency.

## **1\. Project Overview**

This project addresses a key challenge in intelligent driving: achieving fast and accurate occupant injury risk prediction in the pre-crash phase. Traditional post-crash models are accurate but too slow for real-time use, while pre-crash models are fast but lack precision.

To bridge this gap, we propose a **Knowledge Distillation (KD)** framework:

* **Teacher Model**: A complex, high-precision Temporal Convolutional Network (TCN) and Multi-Layer Perceptron (MLP) model. It's trained on detailed post-crash data (waveforms \+ scalar features) to serve as the source of "knowledge."  
* **Student Model**: A lightweight, MLP-only model. It uses only pre-crash scalar features, making it computationally efficient and suitable for real-time deployment.

Through distillation, the Student Model learns from both the ground-truth labels and the rich feature representations learned by the Teacher Model. This significantly improves the Student's predictive performance on critical metrics like the Head Injury Criterion (HIC) and Abbreviated Injury Scale (AIS) without increasing its architectural complexity.

## **2\. Key Features**

* **High Efficiency & Accuracy**: Balances prediction accuracy with the low-latency requirements of pre-crash systems.  
* **Knowledge Distillation**: Implements a teacher-student framework to transfer knowledge from a complex model to a lightweight one.  
* **Modular Design**: Clear and separated scripts for training the teacher, the student (with and without KD), and evaluation.  
* **Comprehensive Evaluation**: Provides detailed performance metrics, including MAE, RMSE, R² for regression (HIC) and Accuracy, G-mean, and a confusion matrix for classification (AIS).

## **3\. Project Structure**

.  
├── runs/                 \# Stores logs, model weights, and results for all training runs  
├── data/                 \# Stores the raw dataset files  
│   ├── data\_crashpulse.npy   \# Crash acceleration waveform data (N, 2, 150\)  
│   └── data\_features.npy     \# Scalar features for crash conditions and occupants (N, 9\)  
├── utils/                \# Utility modules and helper functions  
│   ├── models.py             \# Defines the Teacher and Student model architectures  
│   ├── dataset\_prepare.py    \# Handles dataset preprocessing and loading  
│   ├── weighted\_loss.py      \# Custom weighted loss function  
│   └── combined\_loss.py      \# (Alternative) combined loss function  
├── train\_teacher.py      \# Script: Train the Teacher Model  
├── train\_student\_w\_KD.py \# Script: Train the Student Model with Knowledge Distillation  
├── train\_student\_wo\_KD.py\# Script: Train the Student Model independently (as a baseline)  
├── eval\_model.py         \# Script: Evaluate the performance of a trained model  
└── README.md             \# This README file

## **4\. Getting Started**

Follow these steps to set up the environment and prepare the data.

### **Prerequisites**

This project is built with Python and PyTorch. Ensure you have a compatible environment. The key dependencies are:

* **torch**: The core deep learning framework.  
* **torch\_geometric**: Used for rapid construction of MLP layers.  
* **numpy, pandas**: For data handling and processing.  
* **scikit-learn, imbalanced-learn**: For model evaluation and metrics calculation.  
* **matplotlib**: For plotting and visualizing results.

### **Installation & Data Preparation**

1. **Clone the repository (optional, if you haven't already):**  
   git clone \<your-repository-url\>  
   cd \<repository-directory\>

2. **Install dependencies:**  
   pip install torch torch\_geometric numpy pandas scikit-learn matplotlib imbalanced-learn

3. Prepare the Dataset:  
   Place the raw data files, data\_crashpulse.npy and data\_features.npy, into the ./data/ directory. Then, run the preprocessing script:  
   python utils/dataset\_prepare.py

   This will generate train\_dataset.pt, val\_dataset.pt, and test\_dataset.pt in the ./data/ directory for faster loading during training.

## **5\. Usage: Training and Evaluation**

Follow the sequence below to train and evaluate the models.

### **Step 1: Train the Teacher Model**

The Teacher Model must be trained first as it provides the knowledge for distillation.

python train\_teacher.py

Training artifacts, including logs, model weights (teacher\_best\_mae.pth), and hyperparameters, will be saved to a new timestamped directory in ./runs/ (e.g., TeacherModel\_Train\_\<timestamp\>).

### **Step 2: Train the Student Model**

You can train the Student Model either with or without knowledge distillation.

#### **With Knowledge Distillation (Recommended)**

1. **Update Teacher Model Path**: Open train\_student\_w\_KD.py and set the teacher\_run\_dir variable to the path of your trained Teacher Model from the previous step.  
   \# In train\_student\_w\_KD.py (around line 196\)  
   teacher\_run\_dir \= ".\\\\runs\\\\TeacherModel\_Train\_XXXXXXXX" \# \<-- IMPORTANT: Change this

2. **Start Training**:  
   python train\_student\_w\_KD.py

   Results will be saved to a new directory like StudentModel\_Distill\_\<timestamp\> in ./runs/.

#### **Without Knowledge Distillation (Baseline)**

To train a baseline Student Model for comparison:

python train\_student\_wo\_KD.py

Results will be saved to a new directory like StudentModel\_Baseline\_\<timestamp\> in ./runs/.

### **Step 3: Evaluate a Trained Model**

Use the eval\_model.py script to evaluate any trained model on the test set. You need to specify the run directory and the weight file.

python eval\_model.py \--run\_dir \<path\_to\_model\_directory\> \--weight\_file \<model\_weight\_filename.pth\>

**Examples:**

* **Evaluate the Teacher Model:**  
  python eval\_model.py \-r ./runs/TeacherModel\_Train\_01121600 \-w teacher\_best\_mae.pth

* **Evaluate the Student Model with KD:**  
  python eval\_model.py \-r ./runs/StudentModel\_Distill\_01122045 \-w student\_best\_mae.pth

* **Evaluate the Baseline Student Model:**  
  python eval\_model.py \-r ./runs/StudentModel\_Baseline\_01122201 \-w student\_best\_mae.pth

## **6\. Results**

The evaluation script (eval\_model.py) will:

1. Print key performance metrics to the console for both HIC prediction (MAE, RMSE, R²) and AIS classification (Accuracy, G-mean, Confusion Matrix).  
2. Save a detailed evaluation\_report.md file in the specified run directory.  
3. Generate and save a scatter plot (HIC\_prediction\_scatter.png) comparing predicted vs. actual HIC values.
