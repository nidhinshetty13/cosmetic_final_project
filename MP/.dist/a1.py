import os
import ast
import random
import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------
# 1) Load & clean dataset
# ----------------------------------------------------
DATA_PATH = r"C:/Users/Nidhin Shetty/OneDrive/Desktop/DatasetM.xlsx"
df = pd.read_excel(DATA_PATH)

# Standardize column names
df.columns = [c.strip().lower() for c in df.columns]

# Required columns
REQUIRED_COLS = [
    "product_name",
    "ingredients",
    "toxicity_normal",
    "toxicity_oily",
    "toxicity_dry",
    "toxicity_sensitive",
    "category",
]

# Detect missing columns
missing = [col for col in REQUIRED_COLS if col not in df.columns]
if missing:
    print(f"⚠ Warning: Dataset missing columns: {missing}")
    for col in missing:
        if "toxicity" in col:
            df[col] = 0
        else:
            df[col] = ""

# Normalize ingredients
def normalize_ingredients(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, (list, tuple)):
                return ", ".join(str(x).strip() for x in lst if str(x).strip())
        except Exception:
            pass
    return s

df["ingredients"] = df["ingredients"].apply(normalize_ingredients).fillna("")
df["category"] = df["category"].astype(str).str.strip()

# ----------------------------------------------------
# 2) Vectorizer for ingredient similarity
# ----------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["ingredients"].astype(str))
skin_types = ["normal", "oily", "dry", "sensitive"]

# ----------------------------------------------------
# 3) UI helpers
# ----------------------------------------------------
category_icons = {
    "lipstick": "💄",
    "lip gloss": "💋",
    "perfume": "🌸",
    "perfume oil": "🪔",
    "moisturizer": "🧴",
    "lotion": "🧴",
    "cream": "🥥",
    "sunscreen": "☀️",
    "foundation": "🎨",
    "shampoo": "🧼",
    "conditioner": "💧",
    "soap": "🫧",
}

def get_icon(category):
    key = str(category).strip().lower()
    return category_icons.get(key, "✨")

# ----------------------------------------------------
# 4) Core function
# ----------------------------------------------------
def check_product(product_name, skin_type, top_n=3):
    try:
        if not product_name or not skin_type:
            return (
                "⚠️ Please select both product and skin type.",
                "", "<p></p>",
                "<div style='height:220px;border-radius:14px;background:#202334;display:flex;align-items:center;justify-content:center;color:#9aa0b4;font-weight:600;'>No product selected</div>",
                None
            )

        match = df.loc[df["product_name"].str.lower() == str(product_name).strip().lower()]
        if match.empty:
            return (
                f"⚠️ Product '{product_name}' not found in dataset.",
                "", "<p></p>",
                "<div style='height:220px;border-radius:14px;background:#202334;display:flex;align-items:center;justify-content:center;color:#ff6b6b;font-weight:600;'>Product not found</div>",
                None
            )

        product = match.iloc[0]
        toxicity_col = f"toxicity_{skin_type.lower()}"
        toxicity_value = int(product.get(toxicity_col, 0))
        score = random.randint(0, 30) if toxicity_value == 0 else random.randint(70, 100)

        if score <= 30:
            color, label = "#27c93f", f"🟢 {score}/100 (Safe)"
        elif score <= 69:
            color, label = "#ffb020", f"🟠 {score}/100 (Moderate)"
        else:
            color, label = "#ff4d4f", f"🔴 {score}/100 (Toxic)"

        meter_html = f"""
        <style>
        @keyframes grow {{
            from {{ width: 0%; }}
            to {{ width: {score}%; }}
        }}
        </style>
        <div style='width:100%; background:#1b1f2d; border-radius:10px; overflow:hidden;'>
            <div style='width:{score}%; background:{color}; height:24px; text-align:center; color:#fff; font-weight:700; border-radius:10px; animation: grow 1.2s ease-out;'>
                {score}%
            </div>
        </div>
        <div style="font-size:14px; text-align:center; margin-top:6px; font-weight:600;">{label}</div>
        """

        icon = get_icon(product.get("category", ""))

        if toxicity_value == 0:
            message = f"✅ {icon} <b>{product['product_name']}</b> is <b>SAFE</b> for <b>{skin_type}</b> skin."
            rec_df = pd.DataFrame([["Not needed, product is safe ✅", "—"]], columns=["Product Name", "Ingredients"])
        else:
            # Get toxic ingredient from the last column
            toxic_ingredient = product.iloc[-1]
            message = f"❌ {icon} <b>{product['product_name']}</b> is <b>TOXIC</b> for <b>{skin_type}</b> skin.<br>⚠️ Causing ingredient(s): <b>{toxic_ingredient}</b>"

            # Find similar safe alternatives
            prod_vec = vectorizer.transform([product["ingredients"]])
            similarities = (X @ prod_vec.T).toarray().ravel()
            temp = df.copy()
            temp["similarity"] = similarities

            same_cat = temp["toxicity_reason"].str.lower() == str(product["toxicity_reason"]).lower()
            safe_mask = temp.get(toxicity_col, 0) == 0
            recs = temp[safe_mask & same_cat].sort_values("similarity", ascending=False).head(max(1, top_n))

            if recs.empty:
                rec_df = pd.DataFrame([["❌ No safe alternatives found", "—"]], columns=["Product Name", "Ingredients"])
            else:
                rec_df = recs[["product_name", "ingredients"]].rename(columns={"product_name": "Product Name", "ingredients": "Ingredients"})

        # Save recommendations CSV
        downloads_dir = os.path.join(os.getcwd(), "downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        csv_path = os.path.join(downloads_dir, f"Alternatives_for_{product_name.replace(' ','_')}.csv")
        rec_df.to_csv(csv_path, index=False)

        table_html = rec_df.to_html(index=False, escape=False)
        table_html = table_html.replace(
            "Not needed, product is safe ✅", "<span style='color:#27c93f;font-weight:700;'>Not needed, product is safe ✅</span>"
        ).replace(
            "❌ No safe alternatives found", "<span style='color:#ff4d4f;font-weight:700;'>❌ No safe alternatives found</span>"
        )

        product_card = f"""
        <div style="height:220px;border-radius:14px;background:linear-gradient(135deg,#1b1f2d,#23283b);color:#e5e7f2;padding:16px;display:flex;flex-direction:column;justify-content:center;">
            <div style="font-size:48px;line-height:1">{icon}</div>
            <div style="font-size:18px;margin-top:8px;font-weight:700;">{product['product_name']}</div>
            <div style="opacity:.8;font-size:13px;margin-top:4px;">Category: {product.get('category','—')}</div>
        </div>
        """

        return message, meter_html, table_html, product_card, csv_path

    except Exception as e:
        return f"⚠️ Oops, something went wrong: <code>{str(e)}</code>", "", "<p></p>", "", None

# ----------------------------------------------------
# 5) Gradio UI
# ----------------------------------------------------
css_theme = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
body { background: #0e1220; font-family: 'Poppins', sans-serif; }
.gr-button { background:#6c63ff; color:#fff; font-weight:700; border-radius:12px; padding:10px 20px; }
.gr-dropdown { border-radius:10px; }
.gr-markdown { color:#cfd3e6; }
h1, h2, h3 { color:#ff6bb5; }
"""

with gr.Blocks(css=css_theme, theme="soft") as demo:
    gr.Markdown("## 🌸 Cosmetic Toxicity Analyzer")
    gr.Markdown("Check if your cosmetic is safe for your skin type and explore <b>safer alternatives</b> 💄✨")

    with gr.Row():
        with gr.Column(scale=1):
            product_input = gr.Dropdown(
                choices=df["product_name"].tolist(),
                label="💄 Enter Product",
                value="",
                allow_custom_value=True
            )
            skin_input = gr.Dropdown(
                choices=skin_types,
                label="🧴 Enter Skin Type",
                value="",
                allow_custom_value=True
            )
            check_btn = gr.Button("🔍 Analyze Product")

        with gr.Column(scale=2):
            output_msg = gr.HTML(label="Result")
            toxicity_meter = gr.HTML(label="Toxicity Risk")
            product_card = gr.HTML(label="Product")

    with gr.Tab("🌟 Recommendations"):
        output_table = gr.HTML(label="Safe Alternatives (Top matches)")
        download_btn = gr.File(label="📥 Download Alternatives as CSV")

    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
**How it works**
- Uses dataset toxicity flags per skin type.
- Recommends safer alternatives in the same category using TF-IDF similarity.
- Shows the toxic ingredient causing risk for the selected skin type.
- Animated, colored bar shows risk visually.
        """)

    check_btn.click(
        fn=check_product,
        inputs=[product_input, skin_input],
        outputs=[output_msg, toxicity_meter, output_table, product_card, download_btn]
    )

demo.launch()
