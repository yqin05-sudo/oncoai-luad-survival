
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

APP_DIR = Path(__file__).parent
TRAIN_PATH = APP_DIR / "data" / "luad.data.csv"
TEMPLATE_PATH = APP_DIR / "sample_data" / "batch_template.csv"

REQUIRED_COLS = [
    "age", "gender", "smoking", "stage_group", "surgery", "chemo", "radiation"
]
MODEL_COLS = REQUIRED_COLS.copy()
ALL_DATA_COLS = ["time_month", "event_binary"] + MODEL_COLS

LABELS = {
    "en": {
        "title": "OncoAI: Lung Cancer Survival Prediction System",
        "subtitle": "AI-powered clinical decision support tool for personalized survival prediction in lung adenocarcinoma",
        "individual_tab": "Individual prediction",
        "batch_tab": "Batch prediction",
        "explorer_tab": "Model explorer",
        "patient_header": "Patient information",
        "model_header": "Model summary",
        "risk_header": "Individual risk evaluation",
        "surv_table_header": "Survival probability at specified time points",
        "curve_header": "Personalized survival curve",
        "risk_explain_header": "Risk explanation",
        "notes_header": "Notes",
        "run_btn": "Run survival prediction",
        "batch_run_btn": "Run batch prediction",
        "download_curve": "Download survival curve (PNG)",
        "download_pred": "Download prediction (CSV)",
        "download_batch": "Download results CSV",
        "download_template": "Download batch template CSV",
        "download_cox": "Download Cox summary (CSV)",
        "download_forest_png": "Download forest plot (PNG)",
        "download_forest_csv": "Download forest plot table (CSV)",
    },
    "cn": {
        "title": "OncoAI：肺腺癌生存预测系统",
        "subtitle": "用于肺腺癌个体化生存预测的 AI 临床决策支持工具",
        "individual_tab": "单病人预测",
        "batch_tab": "批量预测",
        "explorer_tab": "模型浏览",
        "patient_header": "患者资料",
        "model_header": "模型说明",
        "risk_header": "个体风险评估",
        "surv_table_header": "指定时间点生存概率",
        "curve_header": "个体化生存曲线",
        "risk_explain_header": "风险解释",
        "notes_header": "说明",
        "run_btn": "生成生存预测",
        "batch_run_btn": "运行批量预测",
        "download_curve": "下载生存曲线 (PNG)",
        "download_pred": "下载预测结果 (CSV)",
        "download_batch": "下载批量结果 CSV",
        "download_template": "下载批量模板 CSV",
        "download_cox": "下载 Cox 结果 (CSV)",
        "download_forest_png": "下载 Forest plot (PNG)",
        "download_forest_csv": "下载 Forest 表格 (CSV)",
    },
}


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    alias = {
        "case_id": ["case_id", "id", "patient_id"],
        "age": ["age", "age_at_diagnosis"],
        "gender": ["gender", "sex"],
        "smoking": ["smoking", "smoking_status"],
        "stage_group": ["stage_group", "stage"],
        "surgery": ["surgery", "has_surgery"],
        "chemo": ["chemo", "chemotherapy"],
        "radiation": ["radiation", "radiotherapy", "rt"],
    }
    rename_map = {}
    for std, candidates in alias.items():
        if std in out.columns:
            continue
        for cand in candidates:
            if cand in out.columns:
                rename_map[cand] = std
                break
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def to_yes_no(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.Series(np.where(series.fillna(0).astype(float) > 0, "Yes", "No"), index=series.index)
    x = series.astype(str).str.strip().str.lower()
    out = np.where(x.isin(["1", "yes", "y", "true", "t"]), "Yes", "No")
    return pd.Series(out, index=series.index)


@st.cache_data(show_spinner=False)
def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH)
    df["gender"] = pd.Categorical(df["gender"], categories=["female", "male"], ordered=False)
    df["smoking"] = pd.Categorical(df["smoking"], categories=["Never", "Ever"], ordered=False)
    df["stage_group"] = pd.Categorical(df["stage_group"], categories=["I", "II", "III/IV"], ordered=True)
    for col in ["surgery", "chemo", "radiation"]:
        df[col] = pd.Categorical(df[col], categories=["No", "Yes"], ordered=False)
    return df


@st.cache_resource(show_spinner=False)
def fit_model():
    train = load_training_data()
    formula = "age + C(gender) + C(smoking) + C(stage_group) + C(surgery) + C(chemo) + C(radiation)"
    cph = CoxPHFitter()
    cph.fit(train[ALL_DATA_COLS], duration_col="time_month", event_col="event_binary", formula=formula)
    train_lp = cph.predict_log_partial_hazard(train[MODEL_COLS]).rename("lp")
    lp_median = float(np.median(train_lp))
    return cph, train, train_lp, lp_median


def percentile_of_score(score: float, reference: pd.Series) -> float:
    return round(float((reference <= score).mean() * 100), 1)


def risk_category_percentile(percentile: float, lang: str = "en") -> str:
    if lang == "cn":
        if percentile < 33:
            return "低风险 / Low risk"
        if percentile < 66:
            return "中等风险 / Intermediate risk"
        return "高风险 / High risk"
    else:
        if percentile < 33:
            return "Low risk"
        if percentile < 66:
            return "Intermediate risk"
        return "High risk"


def risk_group_by_median(lp: float, lp_median: float) -> str:
    return "High risk" if lp > lp_median else "Low risk"


def clean_single_patient(raw: dict, train: pd.DataFrame) -> pd.DataFrame:
    row = pd.DataFrame([raw])
    row["age"] = pd.to_numeric(row["age"], errors="coerce")
    row["gender"] = pd.Categorical(row["gender"], categories=train["gender"].cat.categories)
    row["smoking"] = pd.Categorical(row["smoking"], categories=train["smoking"].cat.categories)
    row["stage_group"] = pd.Categorical(row["stage_group"], categories=train["stage_group"].cat.categories, ordered=True)
    for col in ["surgery", "chemo", "radiation"]:
        row[col] = pd.Categorical(row[col], categories=train[col].cat.categories)
    return row[MODEL_COLS]


def clean_batch(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    if "case_id" not in out.columns:
        out["case_id"] = [f"C{i:03d}" for i in range(1, len(out) + 1)]
    missing = [c for c in MODEL_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["gender"] = out["gender"].astype(str).str.strip().str.lower()
    out["gender"] = out["gender"].replace({"f": "female", "m": "male"})
    out["gender"] = pd.Categorical(out["gender"], categories=train["gender"].cat.categories)

    out["smoking"] = out["smoking"].astype(str).str.strip()
    out["smoking"] = out["smoking"].replace({"never": "Never", "ever": "Ever", "Never smoker": "Never", "Ever smoker": "Ever"})
    out["smoking"] = pd.Categorical(out["smoking"], categories=train["smoking"].cat.categories)

    out["stage_group"] = out["stage_group"].astype(str).str.strip()
    out["stage_group"] = pd.Categorical(out["stage_group"], categories=train["stage_group"].cat.categories, ordered=True)

    for col in ["surgery", "chemo", "radiation"]:
        out[col] = to_yes_no(out[col])
        out[col] = pd.Categorical(out[col], categories=train[col].cat.categories)

    return out


def survival_at_times(cph: CoxPHFitter, row: pd.DataFrame, times: list[float]) -> pd.DataFrame:
    sf = cph.predict_survival_function(row, times=times)
    # one column only
    vals = sf.iloc[:, 0].values
    return pd.DataFrame({"time_month": times, "survival_probability": vals})


def survival_curve(cph: CoxPHFitter, row: pd.DataFrame) -> pd.DataFrame:
    sf = cph.predict_survival_function(row)
    out = sf.reset_index()
    out.columns = ["time", "survival_probability"]
    return out


def bootstrap_survival_ci(train: pd.DataFrame, row: pd.DataFrame, formula: str, n_boot: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_model, _, _, _ = fit_model()
    timeline = base_model.baseline_survival_.index.values.astype(float)
    curves = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(train), len(train))
        sample = train.iloc[idx].copy()
        try:
            cph = CoxPHFitter()
            cph.fit(sample[ALL_DATA_COLS], duration_col="time_month", event_col="event_binary", formula=formula)
            pred = cph.predict_survival_function(row, times=timeline).iloc[:, 0].values
            curves.append(pred)
        except Exception:
            continue
    if not curves:
        point = base_model.predict_survival_function(row, times=timeline).iloc[:, 0].values
        return pd.DataFrame({"time": timeline, "surv": point, "lower": point, "upper": point})
    arr = np.vstack(curves)
    point = base_model.predict_survival_function(row, times=timeline).iloc[:, 0].values
    lower = np.quantile(arr, 0.025, axis=0)
    upper = np.quantile(arr, 0.975, axis=0)
    return pd.DataFrame({"time": timeline, "surv": point, "lower": lower, "upper": upper})


def pretty_label(x: str) -> str:
    replacements = {
        "age": "Age (per year)",
        "C(gender)[T.male]": "Male vs Female",
        "C(smoking)[T.Ever]": "Ever vs Never",
        "C(stage_group)[T.II]": "Stage II vs I",
        "C(stage_group)[T.III/IV]": "Stage III/IV vs I",
        "C(surgery)[T.Yes]": "Surgery: Yes vs No",
        "C(chemo)[T.Yes]": "Chemo: Yes vs No",
        "C(radiation)[T.Yes]": "Radiation: Yes vs No",
    }
    return replacements.get(str(x), str(x))


def draw_survival_curve(curve_df: pd.DataFrame, point_df: pd.DataFrame | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    if {"lower", "upper"}.issubset(curve_df.columns):
        ax.fill_between(curve_df["time"], curve_df["lower"], curve_df["upper"], alpha=0.2)
    ax.plot(curve_df["time"], curve_df["surv"], linewidth=2)
    if point_df is not None and not point_df.empty:
        ax.scatter(point_df["time_month"], point_df["survival_probability"], s=40, color="red")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.0)
    ax.set_title("Personalized survival curve (with 95% CI)")
    return fig


def draw_forest_plot(cph: CoxPHFitter) -> tuple[plt.Figure, pd.DataFrame]:
    forest = cph.summary.reset_index().copy()
    if "covariate" not in forest.columns:
        forest.rename(columns={forest.columns[0]: "covariate"}, inplace=True)
    forest["Variable"] = forest["covariate"].apply(pretty_label)
    forest["HR"] = forest["exp(coef)"]
    forest["CI_lower"] = forest["exp(coef) lower 95%"]
    forest["CI_upper"] = forest["exp(coef) upper 95%"]
    forest["p_value"] = forest["p"]
    forest = forest.sort_values("HR").reset_index(drop=True)

    n = len(forest)
    fig_h = max(4.5, 0.55 * n + 1.0)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(n)
    ax.errorbar(
        forest["HR"],
        y,
        xerr=[forest["HR"] - forest["CI_lower"], forest["CI_upper"] - forest["HR"]],
        fmt="o",
        capsize=3,
    )
    ax.axvline(1, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(forest["Variable"])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_title("Forest plot of Cox model")
    return fig, forest[["Variable", "HR", "CI_lower", "CI_upper", "p_value"]]


def km_by_group(data: pd.DataFrame, group_col: str, title: str) -> tuple[plt.Figure, float | None]:
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    groups = list(data[group_col].dropna().astype(str).unique())
    for g in groups:
        mask = data[group_col].astype(str) == g
        kmf.fit(
            data.loc[mask, "time_month"],
            data.loc[mask, "event_binary"],
            label=g,
        )
        kmf.plot_survival_function(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    p_val = None
    if len(groups) == 2:
        g1, g2 = groups
        m1 = data[group_col].astype(str) == g1
        m2 = data[group_col].astype(str) == g2
        p_val = logrank_test(
            data.loc[m1, "time_month"],
            data.loc[m2, "time_month"],
            event_observed_A=data.loc[m1, "event_binary"],
            event_observed_B=data.loc[m2, "event_binary"],
        ).p_value
    return fig, p_val


def render_individual_tab(lang: str):
    d = LABELS[lang]
    cph, train, train_lp, lp_median = fit_model()
    formula = "age + C(gender) + C(smoking) + C(stage_group) + C(surgery) + C(chemo) + C(radiation)"

    col_left, col_right = st.columns([1, 1.25], gap="large")

    with col_left:
        st.subheader(d["patient_header"])
        age = st.slider("Age (years)" if lang == "en" else "年龄（岁）", int(train["age"].min()), int(train["age"].max()), 75)
        gender = st.selectbox("Gender" if lang == "en" else "性别", list(train["gender"].cat.categories), index=1)
        smoking = st.selectbox("Smoking history" if lang == "en" else "吸烟史", list(train["smoking"].cat.categories), index=0)
        stage_group = st.selectbox("Tumor stage group" if lang == "en" else "肿瘤分期", list(train["stage_group"].cat.categories), index=2)
        surgery = st.radio("Surgery" if lang == "en" else "手术", ["No", "Yes"], horizontal=True)
        chemo = st.radio("Chemotherapy" if lang == "en" else "化疗", ["No", "Yes"], horizontal=True, index=1)
        radiation = st.radio("Radiotherapy" if lang == "en" else "放疗", ["No", "Yes"], horizontal=True, index=1)

        st.markdown("#### " + ("Prediction time points (months)" if lang == "en" else "预测时间点（单位：月）"))
        tcols = st.columns(3)
        t1 = tcols[0].number_input("T1", value=12, min_value=1, step=1)
        t2 = tcols[1].number_input("T2", value=36, min_value=1, step=1)
        t3 = tcols[2].number_input("T3", value=60, min_value=1, step=1)

        st.markdown("#### " + ("Typical cases (auto-fill)" if lang == "en" else "典型病例一键填充"))
        if st.button("Typical low-risk case" if lang == "en" else "低风险典型病例", use_container_width=True):
            st.session_state["preset"] = {"age": 55, "gender": "female", "smoking": "Never", "stage_group": "I", "surgery": "Yes", "chemo": "No", "radiation": "No"}
        if st.button("Typical high-risk case" if lang == "en" else "高风险典型病例", use_container_width=True):
            st.session_state["preset"] = {"age": 75, "gender": "male", "smoking": "Ever", "stage_group": "III/IV", "surgery": "No", "chemo": "Yes", "radiation": "Yes"}

        if "preset" in st.session_state:
            st.info(("Preset loaded. Adjust values if needed, then run prediction." if lang == "en" else "已加载典型病例。可调整后再运行预测。"))

        run = st.button(d["run_btn"], use_container_width=True, type="primary")

        st.markdown("#### " + d["notes_header"])
        st.write(
            "This model is built on the TCGA-LUAD cohort and provides statistical predictions of overall survival for lung adenocarcinoma patients. Results are for research and educational purposes only and should not be used as direct clinical decisions."
            if lang == "en" else
            "本模型基于 TCGA-LUAD 队列构建，用于肺腺癌总体生存的统计预测。结果仅供科研与教学参考，不应直接替代临床决策。"
        )

    with col_right:
        st.subheader(d["model_header"])
        st.markdown(
            f"""
            - Model type: Cox proportional hazards
            - Dataset: TCGA-LUAD, sample size ≈ n = {len(train)}
            - Key variables: age, gender, smoking, stage_group, surgery, chemotherapy, radiotherapy
            - Outcome: overall survival (months)
            """
            if lang == "en" else
            f"""
            - 模型类型：Cox proportional hazards
            - 数据集：TCGA-LUAD，样本量约 n = {len(train)}
            - 主要变量：年龄、性别、吸烟、分期、手术、化疗、放疗
            - 结局：总体生存（月）
            """
        )

        if run:
            if "preset" in st.session_state:
                age = st.session_state["preset"]["age"]
                gender = st.session_state["preset"]["gender"]
                smoking = st.session_state["preset"]["smoking"]
                stage_group = st.session_state["preset"]["stage_group"]
                surgery = st.session_state["preset"]["surgery"]
                chemo = st.session_state["preset"]["chemo"]
                radiation = st.session_state["preset"]["radiation"]

            row = clean_single_patient(
                {
                    "age": age,
                    "gender": gender,
                    "smoking": smoking,
                    "stage_group": stage_group,
                    "surgery": surgery,
                    "chemo": chemo,
                    "radiation": radiation,
                },
                train,
            )

            lp = float(cph.predict_log_partial_hazard(row).iloc[0])
            percentile = percentile_of_score(lp, train_lp)
            category = risk_category_percentile(percentile, lang=lang)
            hr_vs_median = float(np.exp(lp - lp_median))
            times = sorted({int(t1), int(t2), int(t3)})
            pred_table = survival_at_times(cph, row, times)
            curve_ci = bootstrap_survival_ci(train, row, formula=formula, n_boot=30, seed=42)
            fig = draw_survival_curve(curve_ci, pred_table)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader(d["risk_header"])
                st.write(f"Linear predictor (Risk Score): **{lp:.3f}**")
                st.write(f"Percentile in TCGA-LUAD cohort: **{percentile:.1f}-th percentile**")
                st.write(f"Risk category: **{category}**")
                st.write(f"HR vs cohort median LP: **{hr_vs_median:.2f}**")

                st.subheader(d["risk_explain_header"])
                if percentile < 33:
                    st.write("Low risk: the patient's risk score is lower than most TCGA-LUAD patients." if lang == "en" else "低风险：该患者风险评分低于大多数 TCGA-LUAD 患者。")
                elif percentile < 66:
                    st.write("Intermediate risk: the patient's risk is in the middle range of the cohort." if lang == "en" else "中等风险：该患者风险处于 TCGA-LUAD 队列中间水平。")
                else:
                    st.write("High risk: the patient's risk score is higher than most TCGA-LUAD patients." if lang == "en" else "高风险：该患者风险高于多数 TCGA-LUAD 患者。")

            with c2:
                st.subheader(d["surv_table_header"])
                show = pred_table.copy()
                show["survival_probability"] = (show["survival_probability"] * 100).round(1).astype(str) + "%"
                st.dataframe(show, hide_index=True, use_container_width=True)

            st.subheader(d["curve_header"] + " (with 95% CI)")
            st.pyplot(fig)
            pred_download = pred_table.copy()
            pred_download["risk_score"] = lp
            pred_download["risk_percentile"] = percentile
            pred_download["risk_category"] = category

            dc1, dc2 = st.columns(2)
            with dc1:
                st.download_button(d["download_curve"], data=fig_to_png_bytes(fig), file_name="individual_survival_curve.png", mime="image/png")
            with dc2:
                st.download_button(d["download_pred"], data=df_to_csv_bytes(pred_download), file_name="individual_prediction.csv", mime="text/csv")
        else:
            st.info("Click the prediction button to generate results." if lang == "en" else "点击按钮后生成预测结果。")


def render_batch_tab(lang: str):
    d = LABELS[lang]
    cph, train, train_lp, lp_median = fit_model()
    st.subheader("TCGA-LUAD Batch Risk Prediction (Cohort Mode)")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        uploaded = st.file_uploader("Upload cohort CSV" if lang == "en" else "上传队列 CSV", type=["csv"], key="batch_upload")
        st.download_button(d["download_template"], TEMPLATE_PATH.read_bytes(), file_name="batch_template.csv", mime="text/csv")
        has_outcomes = st.checkbox("My file includes outcomes (time_month, event_binary)", value=False)
        run = st.button(d["batch_run_btn"], type="primary")
    with right:
        if uploaded is None:
            preview = pd.read_csv(TEMPLATE_PATH)
        else:
            preview = pd.read_csv(uploaded)
        st.markdown("#### Preview")
        st.dataframe(preview.head(8), use_container_width=True, hide_index=True)

    if run:
        if uploaded is None:
            df = pd.read_csv(TEMPLATE_PATH)
        else:
            uploaded.seek(0)
            df = pd.read_csv(uploaded)
        try:
            batch = clean_batch(df, train)
        except Exception as e:
            st.error(str(e))
            return

        lp = cph.predict_log_partial_hazard(batch[MODEL_COLS]).astype(float)
        percentiles = [percentile_of_score(v, train_lp) for v in lp]
        out = batch.copy()
        raw = normalize_columns(df)
        existing_case_ids = None
        if "case_id" in raw.columns:
            existing_case_ids = raw["case_id"].astype(str).tolist()
        elif "case_id" in out.columns:
            existing_case_ids = out["case_id"].astype(str).tolist()
        else:
            existing_case_ids = [f"C{i:03d}" for i in range(1, len(out) + 1)]

        out = out.drop(columns=["case_id"], errors="ignore").reset_index(drop=True)
        out.insert(0, "case_id", existing_case_ids[:len(out)])
        out["lp"] = lp.values
        out["risk_percentile"] = percentiles
        out["risk_group"] = np.where(out["lp"] > lp_median, "High risk", "Low risk")
        out["hr_vs_median"] = np.exp(out["lp"] - lp_median)

        surv_123660 = {
            t: cph.predict_survival_function(batch[MODEL_COLS], times=[t]).T.iloc[:, 0].values
            for t in [12, 36, 60]
        }
        for t, vals in surv_123660.items():
            out[f"surv_{t}m"] = vals

        if has_outcomes and all(col in normalize_columns(df).columns for col in ["time_month", "event_binary"]):
            raw = normalize_columns(df)
            out["time_month"] = pd.to_numeric(raw["time_month"], errors="coerce")
            out["event_binary"] = pd.to_numeric(raw["event_binary"], errors="coerce")

        st.markdown("#### Summary")
        st.write(f"N = {len(out)}")
        st.write(f"High risk = {(out['risk_group'] == 'High risk').sum()}")
        st.write(f"Low risk = {(out['risk_group'] == 'Low risk').sum()}")
        st.write(f"LP median cutoff = {lp_median:.4f}")

        g1, g2 = st.columns(2)
        with g1:
            st.dataframe(out.head(20), use_container_width=True, hide_index=True)
        with g2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(out["lp"], bins=30)
            ax.axvline(lp_median)
            ax.set_title("LP distribution")
            st.pyplot(fig)

        if has_outcomes and {"time_month", "event_binary"}.issubset(out.columns):
            clean = out.dropna(subset=["time_month", "event_binary"]).copy()
            if clean["risk_group"].nunique() == 2:
                km_fig, km_p = km_by_group(clean, "risk_group", "KM by batch risk group")
                st.pyplot(km_fig)
                if km_p is not None:
                    st.write(f"Log-rank p-value: {km_p:.4f}")

        st.download_button(d["download_batch"], data=df_to_csv_bytes(out), file_name="luad_batch_results.csv", mime="text/csv")


def render_explorer_tab(lang: str):
    d = LABELS[lang]
    cph, train, train_lp, lp_median = fit_model()
    st.subheader("Model explorer")
    st.write(cph.summary)
    st.download_button(d["download_cox"], data=df_to_csv_bytes(cph.summary.reset_index()), file_name="cox_summary.csv", mime="text/csv")

    fig_forest, forest_table = draw_forest_plot(cph)
    st.pyplot(fig_forest)
    st.download_button(d["download_forest_png"], data=fig_to_png_bytes(fig_forest), file_name="forest_plot.png", mime="image/png")
    st.dataframe(forest_table, use_container_width=True, hide_index=True)
    st.download_button(d["download_forest_csv"], data=df_to_csv_bytes(forest_table), file_name="forest_plot_table.csv", mime="text/csv")

    st.markdown("### Kaplan–Meier quick views")
    age_df = train.copy()
    median_age = float(age_df["age"].median())
    age_df["age_group"] = np.where(age_df["age"] > median_age, "Above median", "Below median")
    age_fig, age_p = km_by_group(age_df, "age_group", f"KM by age median ({median_age:.1f})")
    st.pyplot(age_fig)
    if age_p is not None:
        st.write(f"Age log-rank p-value: {age_p:.4f}")

    stage_fig, _ = km_by_group(train.copy(), "stage_group", "KM by stage_group")
    st.pyplot(stage_fig)
    chemo_fig, chemo_p = km_by_group(train.copy(), "chemo", "KM by chemo")
    st.pyplot(chemo_fig)
    if chemo_p is not None:
        st.write(f"Chemo log-rank p-value: {chemo_p:.4f}")


def main():
    st.set_page_config(page_title="OncoAI: AI-Powered Lung Cancer Survival Prediction System", layout="wide")
    lang = st.sidebar.radio("Language / 语言", ["English", "中文"], index=0)
    lang_key = "en" if lang == "English" else "cn"
    d = LABELS[lang_key]

    st.title(d["title"])
    st.caption(d["subtitle"])
    cph, train, train_lp, lp_median = fit_model()

    tabs = st.tabs([d["individual_tab"], d["batch_tab"], d["explorer_tab"]])
    with tabs[0]:
        render_individual_tab(lang_key)
    with tabs[1]:
        render_batch_tab(lang_key)
    with tabs[2]:
        render_explorer_tab(lang_key)


    st.markdown("---")
    if lang_key == "en":
        st.caption("For research and educational purposes only. Not intended for clinical decision-making.")
    else:
        st.caption("仅用于科研和教学目的，不可替代临床判断。")

if __name__ == "__main__":
    main()
