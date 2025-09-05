## 1. Project Introduction

This project aims to address a key challenge in the field of intelligent driving: how to quickly and accurately predict occupant injury risk before a collision occurs (the pre-crash stage).

Traditional post-crash injury prediction models can achieve high accuracy by leveraging detailed data generated during a collision (e.g., acceleration waveforms) with complex deep learning networks like TCNs. However, their high computational cost makes them unsuitable for the real-time requirements of pre-crash scenarios. Conversely, pre-crash models are computationally fast but are limited to using only basic scalar information (e.g., impact speed, angle), which restricts their accuracy.

To resolve this trade-off, this project proposes a training framework based on **Knowledge Distillation (KD)**:
1.  **Teacher Model**: A complex, high-accuracy model based on a Temporal Convolutional Network (TCN) and a Multi-Layer Perceptron (MLP). It is trained on detailed post-crash data (vehicle crash waveforms + scalar features) and serves as the source of "knowledge."
2.  **Student Model**: A lightweight model based solely on an MLP. It uses only pre-crash scalar features as input, making it simple, computationally efficient, and suitable for real-time prediction.

During training, the Student Model learns not only from the ground-truth injury labels but also from the "knowledge" extracted by the Teacher Model's intermediate encoder and decoder layers via a **distillation loss**. This allows the Student Model to significantly improve its predictive performance (especially on the critical Head Injury AIS scale) without any additional computational overhead, successfully balancing prediction accuracy and computational efficiency.

## 2. Project Structure

\`\`\`
.
├── runs/                 # Stores logs, model weights, and evaluation results for all training runs
├── data/                 # Stores the raw dataset
│   ├── data_crashpulse.npy   # Collision acceleration waveform data (N, 2, 150)
│   └── data_features.npy     # Collision condition and occupant scalar feature data (N, 9)
├── utils/                # Utility modules and helper functions
│   ├── models.py             # Defines the Teacher and Student models
│   ├── dataset_prepare.py    # Dataset preprocessing and loading
│   ├── weighted_loss.py      # Custom weighted loss function
│   └── combined_loss.py      # (Alternative) combined loss function
├── train_teacher.py      # Script: Train the Teacher Model
├── train_student_w_KD.py # Script: Train the Student Model with Knowledge Distillation
├── train_student_wo_KD.py# Script: Train the Student Model independently (as a baseline)
├── eval_model.py         # Script: Evaluate the performance of a trained model
└── README.md             # Project documentation (this file)
\`\`\`

## 3. Dependencies

This project is implemented using Python and PyTorch. You will need to install the following main dependencies:

\`\`\`bash
pip install torch torch_geometric numpy pandas scikit-learn matplotlib imbalanced-learn
\`\`\`

- \`torch\`: The core deep learning framework.
- \`torch_geometric\`: Used for quickly building MLP layers.
- \`numpy\`, \`pandas\`: For data manipulation.
- \`scikit-learn\`, \`imbalanced-learn\`: For model evaluation and metrics.
- \`matplotlib\`: For visualizing results.

## 4. Usage Guide

Please follow the steps below to run this project.

### Step 1: Prepare the Dataset

First, run the \`dataset_prepare.py\` script to preprocess the raw data, split it into training, validation, and test sets, and save them as \`.pt\` files for faster loading in subsequent runs.

1.  Ensure the raw data files, \`data_crashpulse.npy\` and \`data_features.npy\`, are located in the \`./data/\` directory.
2.  Execute the following command:
    \`\`\`bash
    python utils/dataset_prepare.py
    \`\`\`
    After successful execution, files like \`train_dataset.pt\`, \`val_dataset.pt\`, and \`test_dataset.pt\` will be generated in the \`./data/\` directory.

### Step 2: Train the Teacher Model

The Teacher Model serves as the foundation for knowledge distillation and must be trained first.

-   Execute the following command to train the Teacher Model:
    \`\`\`bash
    python train_teacher.py
    \`\`\`
-   Logs, model weights (\`teacher_best_*.pth\`), and hyperparameter records will be saved in a new directory inside \`./runs/\`, named with the pattern \`TeacherModel_Train_<timestamp>\`.

### Step 3: Train the Student Model

We provide two methods for training the Student Model: with knowledge distillation (recommended) and without (for baseline comparison).

#### 3.1 With Knowledge Distillation (with KD)

1.  **Update Teacher Model Path**: Open the \`train_student_w_KD.py\` file and locate line 196. Change the value of the \`teacher_run_dir\` variable to the path of the directory containing the Teacher Model you trained in **Step 2**.
    \`\`\`python
    # train_student_w_KD.py L196
    teacher_run_dir = ".\\\\runs\\\\TeacherModel_Train_XXXXXXXX" # <--- Change this to your teacher model's run directory
    \`\`\`
2.  Execute the following command to start training:
    \`\`\`bash
    python train_student_w_KD.py
    \`\`\`
-   The results will be saved in a new directory inside \`./runs/\`, named with the pattern \`StudentModel_Distill_<timestamp>\`.

#### 3.2 Without Knowledge Distillation (without KD / Baseline)

This script trains a baseline Student Model to validate the effectiveness of knowledge distillation.

-   Simply execute the following command:
    \`\`\`bash
    python train_student_wo_KD.py
    \`\`\`
-   The results will be saved in a new directory inside \`./runs/\`, named with the pattern \`StudentModel_Baseline_<timestamp>\`.

### Step 4: Evaluate Models

Use the \`eval_model.py\` script to evaluate the performance of any trained model on the test set.

-   Execute the following command, specifying the model's run directory with \`-r\` (or \`--run_dir\`) and the specific weight file with \`-w\` (or \`--weight_file\`).
    \`\`\`bash
    python eval_model.py --run_dir <path_to_model_run_directory> --weight_file <model_weight_file.pth>
    \`\`\`

-   **Examples**:
    -   To evaluate the **Teacher Model** (assuming the best MAE weight is \`teacher_best_mae.pth\`):
        \`\`\`bash
        python eval_model.py -r ./runs/TeacherModel_Train_01121600 -w teacher_best_mae.pth
        \`\`\`
    -   To evaluate the **Student Model with KD** (assuming the best MAE weight is \`student_best_mae.pth\`):
        \`\`\`bash
        python eval_model.py -r ./runs/StudentModel_Distill_01122045 -w student_best_mae.pth
        \`\`\`
    -   To evaluate the **baseline Student Model**:
        \`\`\`bash
        python eval_model.py -r ./runs/StudentModel_Baseline_01122201 -w student_best_mae.pth
        \`\`\`

-   The evaluation script will automatically calculate HIC prediction metrics (**MAE, RMSE, R²**) and AIS classification metrics (**Accuracy, Confusion Matrix, G-mean**). The detailed results will be saved to a \`.md\` file in the corresponding \`run_dir\`, along with a scatter plot of predicted vs. true values.

