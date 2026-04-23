import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import os

# 显示图片（图片在上，标题在下）
st.markdown("""
    <img src="https://github.com/Dandan-debug/2M-EC/raw/main/endometrial.svg" width="100" alt="Endometrial Cancer Model Image" style="display: block; margin: 0 auto 20px;">
    <h1 style="font-weight: bold; font-size: 50px; text-align: center; margin: 0;">
        2M-EC Predictive Platform
    </h1>
""", unsafe_allow_html=True)

# 显示描述文本
st.markdown("""
    <p style='text-align: left; font-size: 16px; margin-bottom: 28px;'>
        The 2M-EC (Bimodal Multilevel Endometrial Cancer) model is designed for patient-centered minimally invasive ENDOM screening with high sensitivity and precise diagnosis.<br><br>
        Utilizes multiple models to calculate cancer risk probabilities:<br>
        • <b>CP</b> (Clinical-Pathological Model): Minimally invasive screening model for early-stage endometrial cancer<br>
        • <b>UCP</b> (Ultra-precision Clinical-Piological Model): Precision screening model for endometrial cancer<br><br>
        Input data includes:<br>
        • Patient clinical information: demographics, medical history, ultrasonographic imaging, and tumor markers (HE4 and CA125)<br>
        • Multi-source biospecimen omics data: endometrial metabolic omics, cervical metabolic omics, and plasma molecular omics<br><br>
        Risk calculation:<br>
        • High-risk probability = Highest cancer probability across models<br>
        • Low-risk probability = 1 - Highest cancer probability<br><br>
        Please select either the CP or UCP model based on your requirements.
    </p>
""", unsafe_allow_html=True)


# 获取 APP.py 所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载标准器和模型
scalers = {
    'C': joblib.load(os.path.join(BASE_DIR, 'scaler_standard_C.pkl')),
    'P': joblib.load(os.path.join(BASE_DIR, 'scaler_standard_P.pkl')),
    'U': joblib.load(os.path.join(BASE_DIR, 'scaler_standard_U.pkl'))
}

models = {
    'C': joblib.load(os.path.join(BASE_DIR, 'xgboost_C.pkl')),
    'P': joblib.load(os.path.join(BASE_DIR, 'xgboost_P.pkl')),
    'U': joblib.load(os.path.join(BASE_DIR, 'xgboost_U.pkl'))
}

# 定义特征名称
display_features_to_scale = [
    'Age (years)',
    'Endometrial thickness (mm)',
    'HE4 (pmol/L)',
    'Menopause (1=yes)',
    'HRT (Hormone Replacement Therapy, 1=yes)',
    'Endometrial heterogeneity (1=yes)',
    'Uterine cavity occupation (1=yes)',
    'Uterine cavity occupying lesion with rich blood flow (1=yes)',
    'Uterine cavity fluid (1=yes)'
]

original_features_to_scale = [
    'CI_age', 'CI_endometrial thickness', 'CI_HE4', 'CI_menopause',
    'CI_HRT', 'CI_endometrial heterogeneity',
    'CI_uterine cavity occupation',
    'CI_uterine cavity occupying lesion with rich blood flow',
    'CI_uterine cavity fluid'
]

additional_features = {
    'C': ['CM4160.0','CM727.0','CM889.0','CM7441.0','CM995.0','CM7440.0','CM7439.0','CM734.0',
          'CM1857.0','CM6407.0','CM2920.0','CM729.0','CM628.0'],

    'P': ['PM816.0','PM846.0','PM120.0','PP408.0','PM883.0','PM801.0','PM578.0',
          'PP48.0','PM504.0','PP317.0','PM722.0','PM86.0','PP63.0','PP405.0',
          'PM574.0','PP434.0','PM163.0','PP81.0','PM461.0','PM571.0','PM88.0','PP378.0',
          'PM867.0','PP286.0','PM409.0','PP497.0','PM900.0','PM836.0','PP393.0',
          'PP653.0','PP456.0','PP75.0','PP488.0','PM887.0','PP640.0','PP344.0',
          'PM584.0','PM396.0','PM681.0','PP332.0','PM328.0','PM882.0','PM548.0',
          'PM832.0','PM232.0','PM285.0','PM104.0','PM379.0','PM782.0'],

    'U': ['UM7578.0', 'UM510.0', 'UM507.0', 'UM670.0', 'UM351.0',
          'UM5905.0', 'UM346.0', 'UM355.0', 'UM8899.0', 'UM1152.0',
          'UM5269.0', 'UM6437.0', 'UM5906.0', 'UM7622.0', 'UM8898.0',
          'UM2132.0', 'UM3513.0', 'UM790.0', 'UM8349.0', 'UM2093.0',
          'UM4210.0', 'UM3935.0', 'UM4256.0']
}

# 模型选择
selected_models = st.multiselect(
    "Select the model(s) to be used (you can select one or more)",
    options=['U', 'C', 'P'],
    default=['U']
)

# ── 质谱数据输入方式选择 ──────────────────────────────────────────────
st.markdown("### Mass Spectrometry Data Input")
input_method = st.radio(
    "How would you like to input mass spectrometry data?",
    options=["Manual input", "Upload file (CSV / Excel)"],
    horizontal=True
)

# ── 下载模板按钮 ──────────────────────────────────────────────────────
if selected_models:
    all_ms_features = []
    for mk in selected_models:
        for f in additional_features[mk]:
            if f not in all_ms_features:
                all_ms_features.append(f)

    template_df = pd.DataFrame(columns=all_ms_features)
    template_df.loc[0] = [0.0] * len(all_ms_features)   # 示例行（全0）

    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="📥 Download template CSV for selected model(s)",
        data=csv_template,
        file_name="ms_template.csv",
        mime="text/csv"
    )

# ── 用户输入字典 ──────────────────────────────────────────────────────
user_input = {}

# 临床信息（始终手动输入）
st.markdown("### Clinical Information")
for i, feature in enumerate(display_features_to_scale):
    if "1=yes" in feature:
        user_input[original_features_to_scale[i]] = st.selectbox(f"{feature}:", options=[0, 1])
    else:
        user_input[original_features_to_scale[i]] = st.number_input(f"{feature}:", min_value=0.0, value=0.0)

# ── 质谱数据输入 ──────────────────────────────────────────────────────
ms_data_ready = False

if input_method == "Upload file (CSV / Excel)":
    uploaded_file = st.file_uploader(
        "Upload your mass spectrometry data file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                ms_df = pd.read_csv(uploaded_file)
            else:
                ms_df = pd.read_excel(uploaded_file)

            st.success(f"✅ File uploaded successfully: {ms_df.shape[0]} row(s), {ms_df.shape[1]} column(s)")
            st.dataframe(ms_df.head())

            # 取第一行数据填入 user_input
            row = ms_df.iloc[0]

            missing_cols = []
            for model_key in selected_models:
                for feature in additional_features[model_key]:
                    if feature in row.index:
                        user_input[feature] = float(row[feature])
                    else:
                        missing_cols.append(feature)

            if missing_cols:
                st.warning(f"⚠️ The following columns are missing from the file and will default to 0:\n{missing_cols}")
                for f in missing_cols:
                    user_input[f] = 0.0

            ms_data_ready = True

        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")

else:
    # 手动输入质谱数据
    st.markdown("### Mass Spectrometry Features")
    for model_key in selected_models:
        st.markdown(f"**Model {model_key} features:**")
        for feature in additional_features[model_key]:
            user_input[feature] = st.number_input(
                f"{feature} ({model_key}):", min_value=0.0, format="%.9f"
            )
    ms_data_ready = True

# ── 预测按钮 ──────────────────────────────────────────────────────────
if st.button("Submit"):
    if not ms_data_ready and input_method == "Upload file (CSV / Excel)":
        st.error("Please upload a valid mass spectrometry data file before submitting.")
    else:
        model_predictions = {}

        for model_key in selected_models:
            model_input_df = pd.DataFrame([user_input])
            model_features = original_features_to_scale + additional_features[model_key]
            model_input_df = model_input_df[model_features]
            model_input_df[original_features_to_scale] = scalers[model_key].transform(
                model_input_df[original_features_to_scale]
            )
            predicted_proba = models[model_key].predict_proba(model_input_df)[0]
            predicted_class = models[model_key].predict(model_input_df)[0]
            model_predictions[model_key] = {
                'proba': predicted_proba,
                'class': predicted_class
            }

        if len(selected_models) == 1:
            st.write("Error")

        elif len(selected_models) == 2 and set(selected_models) != {'C', 'P'}:
            st.write("Error")

        elif len(selected_models) == 2 and set(selected_models) == {'C', 'P'}:
            has_positive = any(model_predictions[mk]['class'] == 1 for mk in selected_models)
            max_proba = max(model_predictions[mk]['proba'][1] for mk in selected_models)
            if has_positive:
                st.write(f"ENDOM screening：{max_proba * 100:.2f}%- high risk")
            else:
                st.write(f"ENDOM screening：{max_proba * 100:.2f}%- low risk")

        elif len(selected_models) == 3:
            positive_count = sum(model_predictions[mk]['class'] == 1 for mk in selected_models)
            max_proba = max(model_predictions[mk]['proba'][1] for mk in selected_models)
            if positive_count >= 2:
                st.write(f"ENDOM diagnosis：{max_proba * 100:.2f}%- high risk")
            else:
                low_risk_proba = (1 - max_proba) * 100
                st.write(f"ENDOM diagnosis：{low_risk_proba:.2f}%- low risk")

        else:
            st.write("Error")
