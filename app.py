import os
import zipfile
import uuid
import shutil
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import time
from src.cnnclassifier.utils.common import load_json


from src.cnnclassifier.pipelines.prediction import PredictionPipeline
CLASS_NAMES = {v: k.split('_')[0] for k, v in load_json("classification_order.json").items()}

st.set_page_config(
    page_title="Chest-Cancer Classifier",
    page_icon="🩻",
    layout="wide",
)

with st.sidebar:
    selection = option_menu(
        menu_title="Menu",
        options=["Home (1-image)", "Batch Predict", "About Project"],
        icons=["house", "stack", "info-circle"],
        menu_icon="list",
        default_index=1,
    )

def run_single_prediction(uploaded):
    """Save uploaded file to tmp, run model, return dict."""
    tmp_name = f"tmp_{uuid.uuid4().hex}.png"
    with open(tmp_name, "wb") as f:   # save
        f.write(uploaded.read())

    pipe = PredictionPipeline(tmp_name)
    result = pipe.predict()
    os.remove(tmp_name)
    return result


if selection == "Home (1-image)":
    st.title("🩻 Chest-Cancer Classification")
    st.caption("Upload a single Chest CT or X-ray image (png/jpg)…")
    st.divider()

    uploaded_img = st.file_uploader("Choose image", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Running inference…"):
                time.sleep(2)     
                res = run_single_prediction(uploaded_img)

            st.success(f"**Predicted class:** {res['class']}")
            st.write("### Probabilities")
            label_map = list(CLASS_NAMES.values())
            df_probs = pd.DataFrame(
                [res["probs"]], 
                columns=label_map
            )
            st.bar_chart(df_probs)

elif selection == "Batch Predict":
    st.title("📂 Batch Prediction")
    st.markdown(
        "Upload a **ZIP file** containing PNG/JPG/JPEG images (images may be nested in sub-folders)."
    )
    st.divider()

    zip_file = st.file_uploader("Upload images.zip", type="zip")

    if zip_file is not None:

        tmp_dir = f"batch_{uuid.uuid4().hex}"
        os.makedirs(tmp_dir, exist_ok=True)

        tmp_zip = os.path.join(tmp_dir, "uploaded.zip")
        with open(tmp_zip, "wb") as f:
            f.write(zip_file.read())

        shutil.unpack_archive(tmp_zip, tmp_dir)

        image_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(tmp_dir)
            for f in files
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_paths:
            st.error("❌ No PNG/JPG images found in the ZIP (including sub-folders).")
        else:
            st.info(f"Found **{len(image_paths)}** images – running predictions…")
            results, prog = [], st.progress(0)

            for i, path in enumerate(image_paths, 1):
                res = PredictionPipeline(path).predict()
                results.append(
                    {"image": os.path.relpath(path, tmp_dir), **res}
                )
                prog.progress(i / len(image_paths))

            prog.empty()
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)

            st.download_button(
                "📥 Download results CSV",
                res_df.to_csv(index=False).encode(),
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        shutil.rmtree(tmp_dir)


elif selection == "About Project":
    st.title("ℹ️ About the Chest-Cancer Classifier")
    st.markdown(
        """
### Overview  
This web-app lets clinicians or researchers quickly classify CT-scan slices / X-ray images into **four lung-cancer categories** using a VGG-16 transfer-learned model.

**Classes:**  
1. Adenocarcinoma Cancer  
2. Normal  
3. Large-cell Carcinoma  
4. Squamous-cell Carcinoma  

### Pipeline Highlights  
* ✅ **Data versioning** with **DVC**  
* ✅ **Experiment tracking** with **MLflow** hosted on **DagsHub**  
* ✅ Model saved in `.keras` format for stability  
* ✅ Streamlit frontend for **single & batch inference**
        """
    )

    st.divider()
    st.subheader("🔗 Project Architecture Pipeline")
    with st.expander("PIPELINE"):
        st.image("dag.png", caption="Model Pipeline", use_container_width =True)

    st.divider()
    st.subheader("🧠 VGG16- Model Summary")
    with st.expander("SUMMARY"):
        st.image("artifacts/model.png", caption="Model Summary", use_container_width =True)

    st.divider()
    st.markdown(
        """
### 👨‍💻 Author  
**Bhavin Karangia**  
📌 [GitHub Repo](https://github.com/bhavin2004/Chest-Cancer-CNN-Classifier-with-MLFLOW-and-DVC)
        """
    )
