# AI-Based Sign Language Translator

## Project Overview

The **AI-Based Sign Language Translator** is a computer vision and deep learning system designed to recognize and translate sign language gestures into text.
The goal of this project is to bridge the communication gap between sign language users and people who do not understand sign language.

The system uses **hand landmark detection and machine learning models** to interpret hand gestures and convert them into readable output.

## Features

* Real-time hand gesture detection
* Sign language recognition using trained ML models
* Translation of gestures into readable text
* Modular architecture for dataset expansion
* Visualization of training performance (confusion matrix, accuracy graphs)

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* Matplotlib
* PyTorch / TensorFlow (depending on model implementation)

---

## Project Structure

```
AI-Based-Sign-Language-Translator
│
├── Model1.py                # Model training script
├── Testing_model.py         # Model testing / prediction
├── hand_landmarker.task     # MediaPipe hand landmark model
├── requirements.txt         # Required Python libraries
├── README.md                # Project documentation
└── .gitignore               # Ignored files
```

---

## Installation

### 1 Clone the repository

```  git clone https://github.com/Prithvi565/AI-Based-Sign-Language-Translator.git  ``` 
```  cd AI-Based-Sign-Language-Translator  ```

### 2 Install dependencies

```  pip install -r requirements.txt  ```

---

## Dataset

Due to GitHub size limitations, the dataset is stored externally.

Download dataset from:

https://drive.google.com/file/d/1Ud-hAAwWwEq-ZjGzFBpSxZMahN3H_T7H/view?usp=drive_link

After downloading, place the dataset inside:

```  datasets/  ```

---

## Pretrained Model

The trained model is not stored directly in the repository due to GitHub file size limitations.

Download the pretrained model from:

Model Download Link:
https://drive.google.com/file/d/1WhexAq9bkK9j56VoB4JJh4J970J6apUx/view?usp=drive_link

---

## How to Run the Project

### Train the model

```  python Model1.py  ```

### Test / Predict gestures

```  python Testing_model.py  ```

---

## Future Improvements

* Real-time webcam translation
* Sentence-level sign translation
* Integration with speech output
* Web or mobile interface
* Support for multiple sign languages
* Improved dataset and model accuracy

---

## Contributors

* Prithvi Singh Chauhan
* Nitin Kumar Vishwakarma
* Prajwal Tiwari

---

## License

This project is for academic and research purposes.
