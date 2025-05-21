# AI-POWERED SYNTHETIC DATA GENERATION FOR FLOWER CLASSIFICATION  
## A Comparative Analysis Between CPU and GPU Using Unreal Engine

##  Description

This research focuses on solving the limitations of real-world data availability and inefficient training workflows in image classification. The project leverages Unreal Engine and NVIDIA NDDS to generate synthetic flower datasets and compares the performance of deep learning models trained on real, synthetic, and hybrid datasets. Evaluation is performed across CPU (Intel Core i5) and GPU (Jetson Orin Nano) platforms to assess the efficiency and scalability of deep learning workflows under resource-constrained conditions.

## Solution

The solution consists of two parts:  
1. Synthetic Data Pipeline: Using Unreal Engine and NDDS, thousands of annotated flower images are generated with varying lighting, angles, and textures. This overcomes the scarcity and imbalance of real-world datasets.  
2. Performance Evaluation Framework: Deep learning models are trained and tested on real, synthetic, and hybrid datasets using both CPU and GPU hardware. Metrics such as accuracy, training time, and power consumption are recorded and compared. The Jetson Orin Nano completed training in approximately 2 hours, compared to over 11 hours on the Intel Core i5 CPU.

## Instructions

1. Clone this repository to your local machine.  
2. Add real and synthetic flower image datasets in appropriate folders.  
3. Run the code to train and evaluate the models.  
4. Review the outputs for metrics like accuracy, precision, and training time.  
