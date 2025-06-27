import os
import zipfile
import uuid
import shutil
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from src.cnnclassifier.utils.common import load_json
from src.cnnclassifier.pipelines.prediction import PredictionPipeline

CLASS_NAMES = {v: k.split('_')[0] for k, v in load_json("classification_order.json").items()}

st.set_page_config(page_title="Chest-Cancer Classifier", page_icon="🩻", layout="wide")

with st.sidebar:
    selection = option_menu(
        menu_title="Menu",
        options=["Home (1-image)", "Batch Predict", "About Project"],
        icons=["house", "stack", "info-circle"],
        menu_icon="list",
        default_index=0,
    )

def run_single_prediction(uploaded):
    tmp_name = f"tmp_{uuid.uuid4().hex}.png"
    with open(tmp_name, "wb") as f:
        f.write(uploaded.read())
    pipe = PredictionPipeline(tmp_name)
    result = pipe.predict()
    os.remove(tmp_name)
    return result

if selection == "Home (1-image)":
    st.title("🩻 Chest-Cancer Classification")
    st.caption("Upload a chest CT/X-ray **or** pick a sample image by class.")
    st.divider()


    class_to_imgs = {}
    for root, _, files in os.walk("samples"):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.endswith("test.zip"):
                class_name = os.path.basename(os.path.dirname(os.path.join(root, f))).split('_')[0]
                class_to_imgs.setdefault(class_name, []).append(os.path.join(root, f))

    class_choices = ["-- None --"] + sorted(class_to_imgs.keys())
    chosen_class = st.selectbox("Choose a sample class:", class_choices)

    sample_choices = ["-- None --"]
    if chosen_class != "-- None --":
        sample_choices += [os.path.basename(p) for p in class_to_imgs[chosen_class]]

    chosen_sample = st.selectbox("Pick a sample image:", sample_choices)

    uploaded_img = st.file_uploader("Or upload your image", type=["png", "jpg", "jpeg"])

    final_image = None
    image_caption = ""

    if chosen_class != "-- None --" and chosen_sample != "-- None --":
        final_image = next(p for p in class_to_imgs[chosen_class] if os.path.basename(p) == chosen_sample)
        image_caption = f"Sample: {chosen_class}"
    elif uploaded_img:
        final_image = uploaded_img
        image_caption = "Uploaded Image"

    # ── 2-column layout for Image and Result
    if final_image:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.image(final_image, caption=image_caption, use_container_width=True)

        with col2:
            with st.spinner("Running inference…"):

                res = run_single_prediction(open(final_image, "rb") if isinstance(final_image, str) else uploaded_img)

            st.success(f"**Predicted class:** {res['class']}")
            st.write("### Probabilities")
            df_probs = pd.DataFrame([res["probs"]], columns=list(CLASS_NAMES.values()))
            st.bar_chart(df_probs)

# ──────────────── BATCH PREDICTION ────────────────
elif selection == "Batch Predict":
    st.title("📂 Batch Prediction")
    st.markdown("Upload a **ZIP file** with PNG/JPG images (can include sub-folders).")
    st.divider()

    # 🔽 Sample ZIP
    with open("samples/test.zip", "rb") as f:
        st.download_button(
            label="⬇️ Download Sample ZIP",
            data=f.read(),
            file_name="test.zip",
            mime="application/zip",
            use_container_width=True,
        )

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
            st.error("❌ No PNG/JPG images found in the ZIP.")
        else:
            st.info(f"Found **{len(image_paths)}** images – running predictions…")
            results, prog = [], st.progress(0)

            for i, path in enumerate(image_paths, 1):
                res = PredictionPipeline(path).predict()
                results.append({"image": os.path.relpath(path, tmp_dir), **res})
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
    st.markdown("""
### Overview  
This app classifies CT/X-ray images into **4 lung cancer categories** using a VGG-16-based CNN.

**Classes:**  
1. Adenocarcinoma Cancer  
2. Normal  
3. Large-cell Carcinoma  
4. Squamous-cell Carcinoma  

### Highlights  
- ✅ Streamlit-based UI  
- ✅ Batch & Single prediction  
- ✅ VGG-16 Model  
- ✅ DVC + MLflow  

    """)

    st.link_button("Check MLFLOW Expirements",'https://dagshub.com/bhavin2004/Chest-Cancer-CNN-Classifier-with-MLFLOW-and-DVC.mlflow')

    st.markdown("""
### Visuals
    """)

    st.subheader("🔗 Pipeline Architecture")
    with st.expander("Pipeline Diagram"):
        st.image("dag.png", caption="Pipeline Overview", use_container_width=True)

    st.subheader("🧠 VGG16 Model Summary")
    with st.expander("Model Diagram"):
        st.image("artifacts/model.png", caption="Model Summary", use_container_width=True)

    st.divider()
    st.markdown("""
### 👨‍💻 Author  
**Bhavin Karangia**  
📌 [GitHub Repo](https://github.com/bhavin2004/Chest-Cancer-CNN-Classifier-with-MLFLOW-and-DVC)
    """)
