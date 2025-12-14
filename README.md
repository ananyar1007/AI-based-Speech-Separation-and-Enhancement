#  Lend Me Your Ear  
### Realtime Speech Separation and Enhancement using Deep Neural Networks  
**Anisha Raghu and Ananya Raghu**

---

## Overview
Hearing impairment makes it difficult to understand speech in noisy or crowded environments, even with hearing aids. This project investigates whether a **single deep learning model** can simultaneously perform:

- **Speech separation** (removing interfering speakers)
- **Speech enhancement** (suppressing background noise)
- **Real-time inference** with low latency

Using a modified **Conv-TasNet** architecture, we demonstrate a unified model that improves signal quality in the presence of both speaker interference and noise, and runs in real time on a laptop. The training code is adapted from:  
https://github.com/tky823/DNN-based_source_separation/tree/main


## Motivation
This project was inspired by real-world challenges faced by individuals with hearing loss, including our grandmother, who struggles to isolate voices in noisy environments despite using hearing aids. Existing solutions typically address either speaker separation or noise suppression, but not both together in real time.

---

## Key Contributions
- Designed a **single Conv-TasNet–based model** for joint speech separation and enhancement  
- Created a **custom training dataset** combining multiple speakers and noise at varying strengths  
- Implemented **causal, streaming versions** of Conv1D, normalization, and transpose Conv1D layers  
- Achieved **real-time performance** on a laptop with ~40 ms latency  

---

## Model Architecture
The model is based on **Conv-TasNet**, consisting of three main components:

1. **Encoder**  
   Converts raw time-domain audio into a learned representation using 1D convolutions.

2. **Separator**  
   A time-dilated convolutional network that generates masks to isolate the desired speaker.

3. **Decoder**  
   Reconstructs the time-domain signal using transpose 1D convolutions.

---

## Datasets
We trained and evaluated our models using:

- **LibriSpeech** – clean speech dataset for speaker separation  
- **LibriMix + WHAM!** – speech corrupted with background noise for enhancement  
- **Custom dataset** – mixtures of two speakers plus noise at varying relative strengths, enabling joint training for both tasks  

---

## Training Details
- Trained for **20 epochs** using **SGD**
- Based on an open-source PyTorch Conv-TasNet implementation
- Training performed on a **GPU-enabled desktop**

---

## Streaming & Real-Time Inference
Most speech separation models are non-causal and rely on future audio samples. To enable real-time use:

- Converted the model to a **causal, streaming architecture**
- Processed audio in **fixed-size chunks** while maintaining internal state
- Built a real-time demo using the `sounddevice` Python library for microphone and speaker I/O
Code for the realtime demo is found in the realtime demo folder.
---

## Results
Performance was evaluated using **Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)**.

- **~3 dB improvement** for interfering speakers  
- **~6.8–7 dB improvement** for background noise  
- Jointly trained model **outperformed task-specific models** on all test conditions  

