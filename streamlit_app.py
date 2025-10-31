# --- START OF FILE app.py ---

import streamlit as st
import pandas as pd
import json
import roboflow
import os
import tempfile
import time
from datetime import datetime
# import pytz # No longer needed
import shutil
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import io # To handle image bytes and in-memory Excel file

# --- Page Config ---
st.set_page_config(layout="wide")

# --- Logo and Title ---
logo_path = "Data/logo.png"
try:
    logo = Image.open(logo_path)
    st.image(logo, width=200) # Adjust width as needed
except FileNotFoundError:
    st.error(f"Logo file not found at: {logo_path}")
except Exception as e:
    st.error(f"Error loading logo: {e}")

st.title("Intelligent Bottle Detection Compliance Engine")

# --- Paths ---
bev_master_file_path = "Data/master_file.xlsx"
json_folder = "JSON_Outputs" # Optional debugging folder
report_folder = "Report" # Local save folder
outlet_master_folder = "Data"

# Create necessary folders if they don't exist
for folder in ["Data", report_folder, json_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        # st.info(f"Created folder: {folder}") # Less verbose

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Choose image files (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload one or more cooler images. Ensure filenames follow the format: `OUTLETCODE_AnyOtherInfo.jpg` (e.g., `12345_20231027.jpg`)"
)

# --- Roboflow Config ---
# Consider using Streamlit secrets for API keys
# api_key = st.secrets["roboflow_api_key"]
api_key = "YxeULFmRqt8AtNbwzXrT" # Replace with your key or use secrets
model_id = "cooler-image"
model_version = "1"

# --- Calculation Functions (Identical to previous version) ---
def calculate_chilled_uf_score(pack_compliance_output):
    keywords = ['Coke_small', 'Limca_small', 'thumpsup_small',
                'Sprite_small', 'thumpsup_medium', 'Sprite_Big']
    chilled_scores = {}
    if 'Image_id' not in pack_compliance_output.columns: return pd.Series(chilled_scores)
    for image_id in pack_compliance_output['Image_id'].dropna().unique():
        image_data = pack_compliance_output[pack_compliance_output['Image_id'] == image_id]
        if 'class' in image_data.columns:
            unique_matches = set(image_data[image_data['class'].isin(keywords)]['class'])
            chilled_scores[image_id] = len(unique_matches)
        else: chilled_scores[image_id] = 0
    return pd.Series(chilled_scores)

def calculate_purity_rcs(pack_compliance_output):
    keywords = ["Pepsi", "Mountain", "7up", "Slice", "Sting"]
    purity_scores = {}
    required_cols = ['Image_id', 'shelf', 'class']
    if not all(col in pack_compliance_output.columns for col in required_cols): return pd.Series(purity_scores)
    for image_id in pack_compliance_output['Image_id'].dropna().unique():
        image_data = pack_compliance_output[pack_compliance_output['Image_id'] == image_id].copy()
        image_data['shelf'] = pd.to_numeric(image_data['shelf'], errors='coerce')
        image_data.dropna(subset=['shelf'], inplace=True)
        top_shelves_data = image_data[image_data['shelf'].isin([1, 2])]
        total_top_shelves = len(top_shelves_data)
        top_points = 0
        if total_top_shelves > 0:
            top_keywords_count = top_shelves_data['class'].astype(str).str.startswith(tuple(keywords)).sum()
            top_percentage = (top_keywords_count / total_top_shelves) * 100
            if top_percentage == 0: top_points = 10
            elif top_percentage < 10: top_points = 8
            else: top_points = 0
        other_shelves_data = image_data[image_data['shelf'] > 2]
        total_other_shelves = len(other_shelves_data)
        other_points = 0
        if total_other_shelves > 0:
            other_keywords_count = other_shelves_data['class'].astype(str).str.startswith(tuple(keywords)).sum()
            other_percentage = (other_keywords_count / total_other_shelves) * 100
            if 30 <= other_percentage <= 35: other_points = 1
            elif 26 <= other_percentage <= 29: other_points = 2
            elif 20 <= other_percentage <= 25: other_points = 3
            elif 15 <= other_percentage <= 19: other_points = 4
            elif other_percentage <= 14: other_points = 5
            else: other_points = 0
        purity_scores[image_id] = top_points + other_points
    return pd.Series(purity_scores)

# --- Outlet Data Population (Identical to previous version) ---
def populate_outlet_data(report_df, outlet_master_folder):
    columns_to_fill = {"OTYP": "MainChannelType", "OUTLET_NM": "OutletName", "ASM": "ASM_Name", "SE": "RSE_Name", "PSR": "PSR_Desc"}
    for col in columns_to_fill.keys():
        if col not in report_df.columns: report_df[col] = pd.NA
    if 'OUTLET CODE' not in report_df.columns:
        st.error("'OUTLET CODE' column missing. Cannot map outlet data. Ensure filenames include Outlet Code.")
        return report_df
    report_df['OUTLET CODE'] = report_df['OUTLET CODE'].astype(str)
    try:
        outlet_master_files = [f for f in os.listdir(outlet_master_folder) if f.startswith("OutletMaster_") and f.endswith(".csv")]
        if not outlet_master_files:
            st.warning(f"No 'OutletMaster_*.csv' files found in: {outlet_master_folder}")
            return report_df
        outlet_master_combined = pd.DataFrame()
        required_master_cols = ["Outletid"] + list(columns_to_fill.values())
        for file in outlet_master_files:
            file_path = os.path.join(outlet_master_folder, file)
            try:
                cols_in_file = pd.read_csv(file_path, nrows=0).columns.tolist()
                cols_to_read = [col for col in required_master_cols if col in cols_in_file]
                if "Outletid" not in cols_to_read: continue
                temp_df = pd.read_csv(file_path, usecols=cols_to_read)
                temp_df['Outletid'] = temp_df['Outletid'].astype(str)
                outlet_master_combined = pd.concat([outlet_master_combined, temp_df], ignore_index=True)
            except Exception as e: st.error(f"Error reading {file}: {e}")
        if outlet_master_combined.empty:
             st.warning("No valid data loaded from OutletMaster files.")
             return report_df
        outlet_master_combined.drop_duplicates(subset='Outletid', keep='first', inplace=True)
        outlet_map = outlet_master_combined.set_index('Outletid').to_dict('index')
        for report_col, master_col in columns_to_fill.items():
            if master_col in outlet_master_combined.columns:
                 report_df[report_col] = report_df['OUTLET CODE'].map(lambda oid: outlet_map.get(oid, {}).get(master_col))
            else:
                 if report_col not in report_df.columns: report_df[report_col] = pd.NA
    except Exception as e: st.error(f"Error during outlet data population: {e}")
    return report_df

# --- Core Data Processing Functions (Identical to previous version) ---
def size_classification(name):
    name_lower = str(name).lower();
    if 'small' in name_lower: return "ic"
    elif 'medium' in name_lower: return "otg"
    elif 'big' in name_lower or 'large' in name_lower: return "fc"
    else: return ""

def follows_order(ideal_order, current_order):
    if not isinstance(current_order, list): return 0
    ideal_index, current_index = 0, 0
    while current_index < len(current_order) and ideal_index < len(ideal_order):
        try: found_at = ideal_order[ideal_index:].index(current_order[current_index]); ideal_index += found_at + 1; current_index += 1
        except ValueError: return 0
    return 1

def expected_shelf_op(shelf):
    try:
        shelf_num = int(shelf)
        if shelf_num in [1, 2]: return "ic"
        elif shelf_num in [3, 4]: return "otg"
        elif shelf_num >= 5: return "fc"
        else: return ""
    except: return ""

def get_pack_order_comp_and_shelves(predictions_list, image_id, bev_master):
    """Processes predictions for a single image for pack order and shelf count."""
    if not predictions_list: return pd.DataFrame(), 0
    try:
        df = pd.DataFrame(predictions_list)
        required_cols = ['x', 'y', 'width', 'height', 'class', 'class_id']
        if not all(col in df.columns for col in required_cols): return pd.DataFrame(), 0

        df = df.sort_values(by=['y', 'x']).reset_index(drop=True)
        df['Image_id'] = image_id
        df['y_diff'] = df['y'].diff().fillna(0)
        threshold = df['height'].mean() * 0.5 if df['height'].mean() > 0 else 50
        df['new_bin'] = (df['y_diff'] > threshold).cumsum()
        df['shelf'] = (df['new_bin'] + 1).astype(int)
        max_shelf = df['shelf'].max() if not df.empty else 0

        df = df.drop(columns=['y_diff', 'new_bin'], errors='ignore')
        df = df.sort_values(by=['shelf', 'x'])
        df['actual size (json op)'] = df['class'].apply(size_classification)
        df['class_id'] = pd.to_numeric(df['class_id'], errors='coerce') # Adapt if class_id is string
        df = pd.merge(df, bev_master[['class_id', 'flavour_type']], on='class_id', how='left')
        df['flavour_type'] = df['flavour_type'].fillna('Unknown')
        df['expected size'] = df['shelf'].apply(expected_shelf_op)
        df['pack_order_check'] = df.apply(
            lambda row: 1 if row['actual size (json op)'] != row['expected size'] and row['expected size'] != "" else 0, axis=1)
        return df, max_shelf
    except Exception as e:
        st.error(f"Error processing pack order for {image_id}: {e}")
        return pd.DataFrame(), 0

def get_brand_order_comp(df_poc_group):
    """Calculates brand order compliance for a single image DataFrame group."""
    ideal_order = ['Cola', 'Flavour', 'Energy Drink', 'Stills', 'Mixers', 'Water']
    if df_poc_group.empty or 'shelf' not in df_poc_group.columns or 'flavour_type' not in df_poc_group.columns: return pd.DataFrame()
    try:
        image_id = df_poc_group['Image_id'].iloc[0]
        shelf_flavour_mapping = df_poc_group.groupby('shelf')['flavour_type'].apply(
            lambda x: [item for item in x if pd.notna(item) and item != 'Unknown']
        ).to_dict()
        comparison_result = []
        for shelf, flavours in shelf_flavour_mapping.items():
            if not flavours: continue
            result = {'Shelf': shelf, 'Flavour List': flavours, 'Ideal Order': ideal_order,
                      'brand_order_check': follows_order(ideal_order, flavours)}
            comparison_result.append(result)
        if not comparison_result: return pd.DataFrame()
        comparison_df = pd.DataFrame(comparison_result)
        comparison_df['Image_id'] = image_id
        return comparison_df
    except Exception as e:
        st.error(f"Error calculating brand order for {image_id}: {e}")
        return pd.DataFrame()

# --- Function to create Excel file in memory ---
def to_excel(df_dict):
    """Writes multiple dataframes to an Excel file in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df_data in df_dict.items():
             if isinstance(df_data, pd.DataFrame) and not df_data.empty:
                 df_data.to_excel(writer, sheet_name=sheet_name, index=False)
    processed_data = output.getvalue()
    return processed_data

# --- Main Processing Logic ---
if uploaded_files:
    if st.button("Process Uploaded Images and Generate Report"):
        # --- Load Master Data ---
        try:
            bev_master = pd.read_excel(bev_master_file_path)
            if 'class_id' not in bev_master.columns or 'flavour_type' not in bev_master.columns:
                 st.error(f"Master file '{bev_master_file_path}' missing required columns: 'class_id', 'flavour_type'")
                 st.stop()
            # bev_master['class_id'] = bev_master['class_id'].astype(int) # Ensure type consistency if needed
        except FileNotFoundError: st.error(f"Master file not found: {bev_master_file_path}"); st.stop()
        except Exception as e: st.error(f"Error reading master file: {e}"); st.stop()

        # --- Initialize Roboflow ---
        try:
            st.info("Connecting to Roboflow...")
            rf = roboflow.Roboflow(api_key=api_key)
            project = rf.workspace().project(model_id)
            model = project.version(model_version).model
            st.success("Connected to Roboflow model.")
        except Exception as e: st.error(f"Failed to connect to Roboflow: {e}"); st.stop()

        # --- Font Loading for Drawing (Silent Fallback) ---
        # FIX: Load font here, fallback silently without warning
        try:
            main_font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            # No warning here, just use default
            main_font = ImageFont.load_default()


        # --- Data Storage for Aggregation ---
        all_pack_compliance_dfs = []
        all_brand_compliance_dfs = []
        processed_image_ids = []

        # --- Process Each Uploaded Image ---
        st.info(f"Processing {len(uploaded_files)} uploaded images...")
        progress_bar_vis = st.progress(0)
        image_display_placeholder = st.container() # Display results progressively

        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            image_id = os.path.splitext(filename)[0]
            processed_image_ids.append(image_id)

            outlet_code = filename.split('_')[0]
            #if not outlet_code.isdigit():
            #    st.warning(f"Filename '{filename}' may not follow 'OUTLETCODE_...' format.", icon="⚠️")

            with image_display_placeholder:
                st.subheader(f"Processing: {filename}")

            tmp_file_path = None # Initialize path
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # 1. Perform Inference
                with st.spinner(f"Running detection on {filename}..."):
                    predictions_response = model.predict(tmp_file_path, confidence=40, overlap=30).json()
                    raw_predictions = predictions_response.get("predictions", [])

                    # Optional: Save raw JSON output
                    try:
                         output_json_path = os.path.join(json_folder, f"OP_{filename}.json")
                         with open(output_json_path, 'w') as json_file: json.dump(raw_predictions, json_file, indent=4)
                    except Exception as json_e: st.warning(f"Could not save JSON for {filename}: {json_e}", icon="⚠️")

                # 2. Load Original Image
                original_image = Image.open(uploaded_file).convert("RGB")

                # 3. Create Detected Image
                detected_image = original_image.copy()
                draw = ImageDraw.Draw(detected_image)
                if raw_predictions:
                    for pred in raw_predictions:
                        x_center, y_center = pred.get('x', 0), pred.get('y', 0)
                        width, height = pred.get('width', 0), pred.get('height', 0)
                        confidence, label = pred.get('confidence', 0), pred.get('class', 'Unknown')
                        x1, y1 = x_center - width / 2, y_center - height / 2
                        x2, y2 = x_center + width / 2, y_center + height / 2
                        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                        text = f"{label}: {confidence:.2f}"
                        text_pos = (x1 + 2, y1 - 14)
                        if text_pos[1] < 2: text_pos = (x1 + 2, y1 + 2)
                        try: # Draw text background
                            text_bbox = draw.textbbox(text_pos, text, font=main_font) # Use loaded font
                            bg_coords = (text_bbox[0]-1, text_bbox[1]-1, text_bbox[2]+1, text_bbox[3]+1)
                            draw.rectangle(bg_coords, fill="lime")
                        except Exception: pass
                        draw.text(text_pos, text, fill="black", font=main_font) # Use loaded font

                # 4. Calculate Compliance and Shelf Count
                pack_comp_df_single, max_shelf = get_pack_order_comp_and_shelves(raw_predictions, image_id, bev_master)
                if not pack_comp_df_single.empty:
                    all_pack_compliance_dfs.append(pack_comp_df_single)
                    brand_comp_df_single = get_brand_order_comp(pack_comp_df_single)
                    if not brand_comp_df_single.empty: all_brand_compliance_dfs.append(brand_comp_df_single)

                # 5. Calculate Detection Summary
                detection_summary = defaultdict(list)
                if raw_predictions:
                    for pred in raw_predictions: detection_summary[pred.get('class', 'Unknown')].append(pred.get('confidence', 0))

                # --- Display Results for Current Image ---
                with image_display_placeholder:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_image, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(detected_image, caption="Detected Image", use_container_width=True)

                    st.metric(label="Number of Shelves Detected", value=max_shelf if max_shelf > 0 else "N/A")

                    st.write(f"**Detected Items Summary:**")
                    if detection_summary:
                        sorted_items = sorted(detection_summary.items())
                        summary_md = ""
                        for class_name, conf_list in sorted_items:
                            count = len(conf_list)
                            avg_conf = sum(conf_list) / count if count > 0 else 0
                            summary_md += f"- **{class_name}:** Count = `{count}` (Avg. Confidence: `{avg_conf:.2f}`)\n"
                        st.markdown(summary_md)
                    else: st.write("No objects detected in this image.")
                    st.divider()

            except Exception as e:
                st.error(f"Failed to process {filename}: {e}")
            finally: # Clean up temporary file
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

            progress_bar_vis.progress((i + 1) / len(uploaded_files))

        progress_bar_vis.empty()
        st.success("Image processing and visualization complete.")
        st.divider()

        # --- Generate Final Compliance Report ---
        st.header("Compliance Report")

        if not all_pack_compliance_dfs:
             st.warning("No valid data collected. Cannot generate compliance report.")
        else:
            with st.spinner("Generating final compliance scores and report..."):
                # Concatenate Data
                pack_compliance_output = pd.concat(all_pack_compliance_dfs, ignore_index=True)
                brand_compliance_df = pd.concat(all_brand_compliance_dfs, ignore_index=True) if all_brand_compliance_dfs else pd.DataFrame()

                # Aggregate Scores (Identical logic to previous version)
                if 'pack_order_check' in pack_compliance_output.columns:
                    pack_order_check = pack_compliance_output.groupby('Image_id')['pack_order_check'].sum().reset_index()
                    pack_order_check['pack_order_score'] = pack_order_check['pack_order_check'].apply(lambda x: 0 if x > 0 else 2)
                else: pack_order_check = pd.DataFrame({'Image_id': processed_image_ids}).assign(pack_order_check=0, pack_order_score=0)
                if 'brand_order_check' in brand_compliance_df.columns:
                    brand_check_agg = brand_compliance_df.groupby('Image_id')['brand_order_check'].agg(['sum', 'size']).reset_index()
                    brand_check_agg['brand_order_score'] = brand_check_agg.apply(lambda row: 3 if row['sum'] == row['size'] and row['size'] > 0 else 0, axis=1)
                    brand_order_check = brand_check_agg[['Image_id', 'sum', 'brand_order_score']].rename(columns={'sum': 'brand_order_check_raw_sum'})
                else: brand_order_check = pd.DataFrame({'Image_id': processed_image_ids}).assign(brand_order_check_raw_sum=0, brand_order_score=0)

                # Merge and Calculate Final Scores
                final_op = pd.DataFrame({'Image_id': processed_image_ids})
                final_op['Image_id'] = final_op['Image_id'].astype(str)
                pack_order_check['Image_id'] = pack_order_check['Image_id'].astype(str)
                brand_order_check['Image_id'] = brand_order_check['Image_id'].astype(str)
                final_op = pd.merge(final_op, pack_order_check, on='Image_id', how='left')
                final_op = pd.merge(final_op, brand_order_check, on='Image_id', how='left')
                chilled_scores = calculate_chilled_uf_score(pack_compliance_output)
                purity_scores = calculate_purity_rcs(pack_compliance_output)
                final_op['Chilled_UF_Scoring_RCS'] = final_op['Image_id'].map(chilled_scores.astype(str))
                final_op['Purity_RCS'] = final_op['Image_id'].map(purity_scores.astype(str))
                score_cols = ['pack_order_check', 'pack_order_score', 'brand_order_check_raw_sum', 'brand_order_score', 'Chilled_UF_Scoring_RCS', 'Purity_RCS']
                for col in score_cols:
                    if col in final_op.columns: final_op[col] = final_op[col].fillna(0)
                    else: final_op[col] = 0 # Ensure column exists

                # Add/Rename Columns & Populate Outlet Data
                final_op['MONTH'] = datetime.now().strftime("%B")
                final_op['OUTLET CODE'] = final_op['Image_id'].astype(str).apply(lambda x: x.split('_')[0] if '_' in x else x)
                final_op['Visible_Accessible_RCS'] = 0 # Default value
                for col in ['OTYP', 'OUTLET_NM', 'DL1_ADD1', 'VPO_CLASS', 'SM', 'ASM', 'SE', 'PSR', 'RT']:
                     if col not in final_op.columns: final_op[col] = '' if col != 'VPO_CLASS' else pd.NA
                final_op = final_op.rename(columns={'brand_order_check_raw_sum': 'Brand_Order_Compliance_Check', 'brand_order_score': 'Brand_Order_Compliance_RCS', 'pack_order_check': 'Pack_Order_Compliance_Test', 'pack_order_score': 'Pack_Order_Compliance_RCS'})
                final_op = populate_outlet_data(final_op, outlet_master_folder)

                # Calculate Total Score
                total_score_cols = ['Visible_Accessible_RCS', 'Purity_RCS', 'Chilled_UF_Scoring_RCS', 'Brand_Order_Compliance_RCS', 'Pack_Order_Compliance_RCS']
                final_op['Total_Equipment_Score'] = 0
                for col in total_score_cols:
                    if col in final_op.columns: final_op[col] = pd.to_numeric(final_op[col], errors='coerce').fillna(0); final_op['Total_Equipment_Score'] += final_op[col]

                # Define final column order for report
                excel_columns_order = ['MONTH', 'OTYP', 'OUTLET_NM', 'DL1_ADD1', 'VPO_CLASS', 'OUTLET CODE', 'SM', 'ASM', 'SE', 'PSR', 'RT', 'Visible_Accessible_RCS', 'Purity_RCS', 'Chilled_UF_Scoring_RCS', 'Brand_Order_Compliance_Check', 'Brand_Order_Compliance_RCS', 'Pack_Order_Compliance_Test', 'Pack_Order_Compliance_RCS', 'Total_Equipment_Score']
                for col in excel_columns_order: # Ensure columns exist
                     if col not in final_op.columns: final_op[col] = pd.NA

                # Display Final Report Table
                st.dataframe(final_op[excel_columns_order])

                # --- Prepare DataFrames for Excel Export ---
                excel_data_dict = {
                    'Compliance Scores': final_op[excel_columns_order]
                }
                # Include detail sheets only if data exists
                if not pack_compliance_output.empty:
                    excel_data_dict['Pack Order Details'] = pack_compliance_output
                if not brand_compliance_df.empty:
                     brand_display_cols = ['Image_id', 'Shelf', 'Flavour List', 'brand_order_check']
                     valid_brand_cols = [col for col in brand_display_cols if col in brand_compliance_df.columns]
                     if valid_brand_cols:
                         excel_data_dict['Brand Order Details'] = brand_compliance_df[valid_brand_cols]

                # --- Create Excel file in memory ---
                excel_bytes = to_excel(excel_data_dict)

                # --- Download Button ---
                report_filename = f"COMPLIANCE_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                st.download_button(
                    label="📥 Download Compliance Report (Excel)",
                    data=excel_bytes,
                    file_name=report_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Optional: Save locally as well
                try:
                    local_report_path = os.path.join(report_folder, report_filename)
                    with open(local_report_path, 'wb') as f:
                        f.write(excel_bytes)
                    st.info(f"Report also saved locally to: {local_report_path}")
                except Exception as e:
                    st.error(f"Error saving report locally: {e}")


# --- Footer/Info ---
if not uploaded_files:
    st.info("Upload images using the button above to start processing.")

# --- END OF FILE app.py ---
