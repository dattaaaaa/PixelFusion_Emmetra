# Assignment3
# HDR Image Processing

## Overview
This project implements an HDR (High Dynamic Range) image processing pipeline that combines three differently exposed images into a single HDR image. The processed HDR image is tone-mapped into a viewable format, and evaluation metrics such as **Dynamic Range** and **Contrast-to-Noise Ratio (CNR)** are calculated to assess the quality of the result.

## Features
- **HDR Merging Methods**:
  - Debevec
  - Robertson
  - Mertens Fusion
- **Tone Mapping Techniques**:
  - Reinhard
  - Drago
  - Mantiuk
- **Evaluation Metrics**:
  - Dynamic Range
  - Contrast-to-Noise Ratio (CNR)

## Folder Structure

```
HDR-Image-Processing
├── hdr_processing.py    # Main Python script 
├── test_data            # Test images for HDR processing 
├── Report               # Project documentation and report 
├── requirements.txt     # Python dependencies
├── README.md            # Instructions for running the demo
├── Results              # resultant images of all possible combinations
```

## Setup and Installation

### Prerequisites
- Python 3.8 or higher installed on your system.
  
### Steps
## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/ChillaleNaveen/Assignment3.git
   cd Assignment3
2. pip install -r requirements.txt #Install the required Python dependencies.
3. streamlit run src/hdr_processing.py #Run the Streamlit application.

