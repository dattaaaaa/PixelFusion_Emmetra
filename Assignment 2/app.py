import streamlit as st
import numpy as np
import cv2
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from pipe import ISPPipeline, DenoiseSharpenPipeline

def _safe_dncnn_denoise(pipeline, image):
    """
    Safely apply DnCNN denoising with fallback
    """
    try:
        # Check if DnCNN model is loaded
        if pipeline.dncnn is None:
            print("DnCNN model not loaded. Falling back to original image.")
            return image
        
        # Ensure image is in correct format for DnCNN
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Preprocess image for DnCNN
        img_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(pipeline.device)
        
        # Apply denoising
        with torch.no_grad():
            denoised_tensor = pipeline.dncnn(img_tensor)
        
        # Convert back to numpy
        denoised = denoised_tensor.squeeze().cpu().numpy().transpose((1, 2, 0))
        denoised = (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
        
        return denoised
    
    except Exception as e:
        print(f"Error in DnCNN denoising: {e}")
        return image

def convert_image_for_display(image):
    """Convert image for Streamlit display"""
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
    
    return image

def generate_comparison_pdf(processed_images, metrics_results):
    """Generate a PDF report comparing denoising and sharpening methods"""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    import tempfile
    import os
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2, height-50, "Image Signal Processing Comparison")
    
    # Images and metrics
    y_position = height - 100
    with tempfile.TemporaryDirectory() as tmpdirname:
        for name, img in processed_images.items():
            # Save image temporarily
            plt.figure(figsize=(6,4))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            plt.title(name)
            plt.axis('off')
            
            # Save to a temporary file
            temp_img_path = os.path.join(tmpdirname, f"{name}_image.png")
            plt.savefig(temp_img_path, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Add image to PDF
            c.drawImage(temp_img_path, width/4, y_position-3*inch, width=2*inch, height=2*inch)
            
            # Add metrics
            c.setFont("Helvetica", 10)
            metrics = metrics_results.get(name, {})
            metrics_text = f"SNR: {metrics.get('SNR', 'N/A'):.2f}, Edge Strength: {metrics.get('Edge Strength', 'N/A'):.2f}"
            c.drawString(width/4, y_position-3.2*inch, metrics_text)
            
            y_position -= 3.5*inch
            
            # New page if running out of space
            if y_position < 100:
                c.showPage()
                y_position = height - 100
        
        c.save()
    
    buffer.seek(0)
    return buffer

def main():
    st.set_page_config(page_title="Advanced Image Signal Processing", layout="wide")
    
    st.title("🖼️ Advanced Image Signal Processing")
    
    # Sidebar for configuration
    st.sidebar.header("Image Processing Options")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload RAW Image", 
        type=['raw'], 
        help="Upload a 12-bit RAW image in GRBG Bayer pattern"
    )
    
    # Processing methods selection
    st.sidebar.subheader("Processing Methods")
    denoising_methods = st.sidebar.multiselect(
        "Select Denoising Methods",
        [
            "Gaussian Denoising", 
            "Median Denoising", 
            "Bilateral Denoising", 
            "DnCNN Denoising"
        ],
        default=["Gaussian Denoising", "Median Denoising"]
    )
    
    # Sharpening methods
    sharpening_methods = st.sidebar.multiselect(
        "Select Sharpening Methods",
        [
            "Unsharp Mask", 
            "Laplacian Sharpening"
        ],
        default=["Laplacian Sharpening"]
    )
    
    # ROI selection
    st.sidebar.subheader("Region of Interest")
    roi_x = st.sidebar.slider("ROI X Position", 0, 1920, 200)
    roi_y = st.sidebar.slider("ROI Y Position", 0, 1280, 200)
    roi_width = st.sidebar.slider("ROI Width", 100, 800, 400)
    roi_height = st.sidebar.slider("ROI Height", 100, 800, 400)
    
    if uploaded_file is not None:
        # Initialize pipelines
        isp_pipeline = ISPPipeline()
        denoiser_pipeline = DenoiseSharpenPipeline()
        
        # Save uploaded file temporarily
        with open("temp_upload.raw", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Read RAW image
        raw_image = isp_pipeline.read_raw("temp_upload.raw")
        
        # Basic image processing steps
        demosaiced = isp_pipeline.demosaic(raw_image)
        wb_image = isp_pipeline.white_balance(demosaiced)
        gamma_image = isp_pipeline.apply_gamma(wb_image)
        
        # Denoising methods
        denoising_methods_dict = {
            "Gaussian Denoising": lambda img: cv2.GaussianBlur(img, (5, 5), 1.0),
            "Median Denoising": lambda img: cv2.medianBlur(img, 5),
            "Bilateral Denoising": lambda img: cv2.bilateralFilter(img, 9, 75, 75),
            "DnCNN Denoising": lambda img: _safe_dncnn_denoise(denoiser_pipeline, img)
        }
        
        # Sharpening methods
        sharpening_methods_dict = {
            "Unsharp Mask": lambda img: cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (5, 5), 1.0), -0.5, 0),
            "Laplacian Sharpening": lambda img: cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        }
        
        # Process images
        processed_images = {}
        metrics_results = {}
        
        # Apply denoising methods
        for method_name in denoising_methods:
            denoised_img = denoising_methods_dict[method_name](gamma_image)
            processed_images[method_name] = convert_image_for_display(denoised_img)
            
            # Compute metrics
            roi = (roi_x, roi_y, roi_width, roi_height)
            snr, edge_strength = denoiser_pipeline.compute_metrics(processed_images[method_name], roi)
            metrics_results[method_name] = {
                'SNR': snr,
                'Edge Strength': edge_strength
            }
        
        # Apply sharpening methods
        for method_name in sharpening_methods:
            sharpened_img = sharpening_methods_dict[method_name](gamma_image)
            processed_images[method_name] = convert_image_for_display(sharpened_img)
            
            # Compute metrics
            roi = (roi_x, roi_y, roi_width, roi_height)
            snr, edge_strength = denoiser_pipeline.compute_metrics(processed_images[method_name], roi)
            metrics_results[method_name] = {
                'SNR': snr,
                'Edge Strength': edge_strength
            }
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Processed Images")
            for name, img in processed_images.items():
                st.subheader(name)
                st.image(img, channels="RGB")
        
        with col2:
            st.header("Image Quality Metrics")
            metrics_df = pd.DataFrame.from_dict(metrics_results, orient='index')
            st.dataframe(metrics_df)
        
        # Generate detailed visualization of metrics
        st.header("Metrics Visualization")
        plt.figure(figsize=(10, 6))
        metrics_df.plot(kind='bar', rot=45)
        plt.title("Comparison of Image Processing Methods")
        plt.tight_layout()
        st.pyplot(plt)
        
        # Comparative Analysis
        st.header("Comparative Analysis")
        analysis_text = """
        ### Denoising Method Comparison
        - **Gaussian Filtering**: Provides smooth noise reduction but may blur image details.
        - **Median Filtering**: Effective at removing salt-and-pepper noise while preserving edges.
        - **Bilateral Filtering**: Reduces noise while preserving edge information.
        - **DnCNN Denoising**: AI-based approach that can adaptively remove noise.
        
        ### Sharpening Method Comparison
        - **Unsharp Mask**: Enhances image details by subtracting a blurred version of the image.
        - **Laplacian Sharpening**: Edge enhancement through high-frequency detail amplification.
        """
        st.markdown(analysis_text)
        
        # Download options
        st.header("Download Results")
        
        # Metrics CSV
        results_csv = metrics_df.to_csv(index=True)
        st.download_button(
            label="Download Metrics CSV",
            data=results_csv,
            file_name="image_processing_metrics.csv",
            mime="text/csv"
        )
        
        # PDF Report
        pdf_buffer = generate_comparison_pdf(processed_images, metrics_results)
        st.download_button(
            label="Download Detailed PDF Report",
            data=pdf_buffer,
            file_name="image_processing_comparison.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()