import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_text

# -------------------------------
# Header & hướng dẫn
# -------------------------------
st.set_page_config(page_title="CHIEN THAN Gradient Boosting Demo", layout="wide")
st.markdown("<h2 style='text-align:center;color:#0066cc'>CHIEN THAN Gradient Boosting Demo</h2>", unsafe_allow_html=True)
st.info("How to Use: Select data → Configure target → Set boosting parameters → Enter new point → Run prediction!")

# -------------------------------
# Layout 2 cột
# -------------------------------
left, right = st.columns([1, 1.2])

with left:
    st.markdown("### 📂 Upload Your Data")
    uploaded_file = st.file_uploader("Drop File Here (CSV/Excel)", type=["csv", "xlsx", "xls"])

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)

        st.write("Preview dữ liệu:", df.head())

        # Chọn cột target
        st.markdown("### 🎯 Target Column")
        target_col = st.selectbox("Select target column", options=df.columns)
        st.write(f"Bạn đã chọn target: **{target_col}**")

        # -------------------------------
        # Cấu hình tham số boosting
        # -------------------------------
        st.markdown("### ⚙️ Set Boosting Parameters")

        n_estimators = st.slider("Number of trees (n_estimators)", 10, 500, 100, step=10)
        learning_rate = st.slider("Learning rate", 0.01, 1.0, 0.1)
        max_depth = st.slider("Max depth of trees", 1, 10, 3)
        subsample = st.slider("Subsample (fraction of samples per tree)", 0.1, 1.0, 1.0)
        random_state = st.number_input("Random state (for reproducibility)", value=42, step=1)

        st.caption(f"⚙️ Bạn đã chọn: n_estimators={n_estimators}, learning_rate={learning_rate}, "
                   f"max_depth={max_depth}, subsample={subsample}, random_state={random_state}")

        # Xác định các cột feature
        feature_cols = [c for c in df.columns if c != target_col]
        num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

        # Ép kiểu dữ liệu về string để tránh lỗi encoder
        for col in cat_cols:
            df[col] = df[col].astype(str)

        # Form nhập điểm mới
        st.markdown("### ✍️ Enter New Point for Prediction")
        new_point = {}
        with st.form("new_point_form"):
            for col in num_cols:
                default_val = float(df[col].median()) if pd.notnull(df[col].median()) else 0.0
                new_point[col] = st.number_input(f"{col} (numeric)", value=default_val)

            for col in cat_cols:
                options = sorted([str(x) for x in df[col].dropna().unique().tolist()])[:100]
                new_point[col] = st.selectbox(f"{col} (categorical)", options=options if options else [""], index=0)

            submitted = st.form_submit_button("Run prediction!")

with right:
    if uploaded_file is not None and submitted:
        # Chuẩn bị dữ liệu
        X = df[feature_cols]
        y = df[target_col]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            ]
        )

        # Chọn model
        is_classification = y.nunique() <= 20 and pd.api.types.is_integer_dtype(y)
        if is_classification:
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=random_state
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=random_state
            )

        pipe = Pipeline(steps=[("prep", preprocessor), ("gb", model)])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state,
            stratify=y if is_classification else None
        )
        pipe.fit(X_train, y_train)

        # Đánh giá
        y_pred_test = pipe.predict(X_test)
        if is_classification:
            acc = accuracy_score(y_test, y_pred_test)
            st.success(f"✅ Test accuracy: {acc:.4f}")
        else:
            r2 = r2_score(y_test, y_pred_test)
            st.success(f"✅ Test R²: {r2:.4f}")

        # Dự đoán cho điểm mới
        new_df = pd.DataFrame([new_point])
        pred_new = pipe.predict(new_df)[0]
        st.info(f"🔮 Prediction for new point: {pred_new}")

        # Trực quan hóa tiến trình boosting
        gb_model = pipe.named_steps["gb"]
        X_test_transformed = pipe.named_steps["prep"].transform(X_test)

        st.markdown("### 📈 Boosting Progress: How Prediction Evolves")
        progress_scores = []
        if is_classification:
            for stage_pred in gb_model.staged_predict(X_test_transformed):
                progress_scores.append(accuracy_score(y_test, stage_pred))
            st.line_chart(pd.DataFrame({"Accuracy": progress_scores}))
        else:
            for stage_pred in gb_model.staged_predict(X_test_transformed):
                progress_scores.append(r2_score(y_test, stage_pred))
            st.line_chart(pd.DataFrame({"R² Score": progress_scores}))

        # Hiển thị cây tại iteration
        st.markdown("### 🌳 Select Iteration to Visualize")
        iter_idx = st.slider("Iteration", 1, n_estimators, 1)

        try:
            if is_classification:
                if len(gb_model.estimators_.shape) == 2:
                    tree = gb_model.estimators_[0, iter_idx - 1]
                else:
                    tree = gb_model.estimators_[0][iter_idx - 1]
            else:
                tree = gb_model.estimators_[iter_idx - 1]

            num_names = num_cols
            cat_names = []
            if len(cat_cols) > 0:
                enc = pipe.named_steps["prep"].named_transformers_["cat"]
                cat_names = enc.get_feature_names_out(cat_cols).tolist()
            feature_names = num_names + cat_names

            tree_text = export_text(tree, feature_names=feature_names, spacing=2)
            st.code(tree_text)
        except Exception as e:
            st.warning(f"Không thể hiển thị cây cho iteration {iter_idx}: {e}")



# streamlit run web2.py --server.port 1502