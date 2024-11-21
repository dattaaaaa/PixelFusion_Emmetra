import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import io

# HDR Merging Methods
hdr_methods = {
    "Debevec": cv.createMergeDebevec,
    "Robertson": cv.createMergeRobertson,
    "Mertens Fusion (No HDR)": cv.createMergeMertens,
}

# Tone Mapping Methods
tone_mapping_methods = {
    "Reinhard": cv.createTonemapReinhard,
    "Drago": cv.createTonemapDrago,
    "Mantiuk": cv.createTonemapMantiuk,
}

# Streamlit app title
st.title("HDR Image Processing")

# Instructions
st.write("""
### Upload 3 exposure images for HDR processing
Select your preferred HDR merging and tone mapping techniques. Enter custom EV values if required.
""")

# Dropdown menus for HDR merging and tone mapping
selected_hdr_method = st.selectbox("Select HDR Merging Technique", list(hdr_methods.keys()))
selected_tone_mapping = st.selectbox("Select Tone Mapping Technique", list(tone_mapping_methods.keys()))

# File upload
uploaded_files = st.file_uploader("Choose 3 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 3:
    # Read images
    img_list = [np.array(Image.open(file)) for file in uploaded_files]

    # Ensure all images have the same dimensions
    if not all(img.shape == img_list[0].shape for img in img_list):
        st.error("All images must have the same resolution.")
        st.stop()

    # Custom EV values input
    st.write("Enter the EV values for the 3 images (e.g., +2, 0, -2):")
    ev_values = st.text_input("EV Values (comma-separated)", "+2, 0, -2")
    try:
        ev_values = [float(val) for val in ev_values.split(',')]
        if len(ev_values) != 3:
            st.error("Please enter exactly 3 EV values.")
            st.stop()
    except ValueError:
        st.error("Please enter valid numeric values for EVs.")
        st.stop()

    # Calculate exposure times based on EVs
    def ev_to_exposure_time(ev, base_exposure=0.01):
        """
        Convert EV to exposure time using the formula:
        Exposure Time = Base Exposure * 2^(-EV)
        """
        return base_exposure * (2 ** -ev)

    base_exposure_time = 0.01  # Assume a base exposure of 0.01 seconds
    exposure_times = np.array([ev_to_exposure_time(ev, base_exposure_time) for ev in ev_values], dtype=np.float32)

    # Convert images to uint8 for HDR processing (the OpenCV HDR methods expect uint8 format)
    img_list_uint8 = [np.clip(img, 0, 255).astype(np.uint8) for img in img_list]

    # HDR merging
    with st.spinner("Processing HDR..."):
        if selected_hdr_method in ["Debevec", "Robertson"]:
            hdr_creator = hdr_methods[selected_hdr_method]()
            hdr_image = hdr_creator.process(img_list_uint8, times=exposure_times)
        else:
            hdr_creator = hdr_methods[selected_hdr_method]()
            hdr_image = hdr_creator.process(img_list_uint8)

    # Tone mapping
    tone_mapper = tone_mapping_methods[selected_tone_mapping]()
    tone_mapped = tone_mapper.process(hdr_image)

    # Normalize and convert to 8-bit for display
    tone_mapped = cv.normalize(tone_mapped, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    result_image = (tone_mapped * 255).astype('uint8')

    # Display result
    st.image(result_image, caption=f"{selected_hdr_method} with {selected_tone_mapping}", channels="RGB")

    # Evaluation Metrics for Dynamic Range and CNR
    def calculate_dynamic_range(image):
        """Calculates the dynamic range of an image."""
        image_float = image.astype(np.float32)
        min_val = np.min(image_float)
        max_val = np.max(image_float)
        return max_val, min_val

    def calculate_cnr(image, roi1, roi2):
        """Calculates the Contrast-to-Noise Ratio (CNR) between two regions of interest."""
        roi1_img = image[roi1[0]:roi1[1], roi1[2]:roi1[3]]  # (y1, y2, x1, x2)
        roi2_img = image[roi2[0]:roi2[1], roi2[2]:roi2[3]]  # (y1, y2, x1, x2)

        mean_roi1 = np.mean(roi1_img)
        mean_roi2 = np.mean(roi2_img)

        contrast = abs(mean_roi1 - mean_roi2)
        noise = np.std(roi2_img)  # Assume ROI2 is the noise region
        cnr = contrast / noise
        return cnr

    # Calculate Dynamic Range for HDR Image
    hdr_max, hdr_min = calculate_dynamic_range(result_image)
    st.write(f"Dynamic Range of HDR Image: Max = {hdr_max}, Min = {hdr_min}")

    # Calculate CNR for HDR Image (using example ROIs for bright and dark regions)
    roi_bright = (0, 100, 0, 100)  # (y1, y2, x1, x2) for a bright area
    roi_dark = (200, 300, 200, 300)  # (y1, y2, x1, x2) for a dark area

    cnr_value = calculate_cnr(result_image, roi_bright, roi_dark)
    st.write(f"Contrast-to-Noise Ratio (CNR) of HDR Image: {cnr_value}")

    # Download button
    def get_image_download_link(img):
        """Generates a download link for an image."""
        buffered = io.BytesIO()
        img_pil = Image.fromarray(img)
        img_pil.save(buffered, format="JPEG")
        return buffered.getvalue()

    st.download_button(
        label=f"Download {selected_hdr_method} with {selected_tone_mapping}",
        data=get_image_download_link(result_image),
        file_name=f"{selected_hdr_method}_{selected_tone_mapping}.jpg",
        mime="image/jpeg",
    )

else:
    st.warning("Please upload exactly 3 images for HDR processing.")
