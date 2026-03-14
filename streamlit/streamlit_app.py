"""
=============================================================
  Vehicle Insurance Fraud Detection - Streamlit Frontend
=============================================================
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Insurance Fraud Detector",
    page_icon="🚗",
    layout="wide"
)

# ─── Page Header ─────────────────────────────────────────────
st.title("🚗 Vehicle Insurance Fraud Detection")
st.markdown("Fill in the claim details below and click **Predict** to check if the claim is fraudulent.")
st.divider()

# ─── Valid Values ─────────────────────────────────────────────
@st.cache_data
def get_valid_values():
    try:
        res = requests.get(f"{API_URL}/valid-values", timeout=5)
        if res.status_code == 200:
            return res.json()
    except:
        pass
    # Fallback hardcoded values
    return {
        "policy_state": ["CA", "FL", "GA", "IL", "MI", "NC", "NY", "OH", "PA", "TX"],
        "insured_sex": ["FEMALE", "MALE", "OTHER"],
        "insured_education_level": ["College", "High School", "Masters", "PhD"],
        "insured_occupation": ["Clerk", "Doctor", "Engineer", "Lawyer", "Manager", "Sales", "Teacher", "Technician"],
        "insured_hobbies": ["camping", "chess", "hiking", "movies", "paintball", "reading", "yachting"],
        "incident_type": ["Multi-vehicle Collision", "Parked Car", "Single Vehicle Collision", "Vehicle Theft"],
        "collision_type": ["Front", "Rear", "Side", "Unknown"],
        "incident_severity": ["Major Damage", "Minor Damage", "Total Loss"],
        "authorities_contacted": ["Ambulance", "Fire", "Police"],
        "incident_state": ["CA", "FL", "GA", "IL", "MI", "NC", "NY", "OH", "PA", "TX"],
        "police_report_available": ["No", "Yes"],
    }

valid_vals = get_valid_values()

# ─── Input Form ───────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Policy Info")
    policy_state = st.selectbox("Policy State", valid_vals["policy_state"])
    policy_deductible = st.selectbox("Deductible ($)", [250, 300, 400, 500, 600, 750, 1000], index=2)
    policy_annual_premium = st.number_input("Annual Premium ($)", min_value=100.0, max_value=5000.0, value=1200.0, step=50.0)

    st.subheader("👤 Insured Info")
    insured_age = st.number_input("Age", min_value=18, max_value=100, value=35)
    insured_sex = st.selectbox("Sex", valid_vals["insured_sex"])
    insured_education_level = st.selectbox("Education Level", valid_vals["insured_education_level"])
    insured_occupation = st.selectbox("Occupation", valid_vals["insured_occupation"])
    insured_hobbies = st.selectbox("Hobbies", valid_vals["insured_hobbies"])

with col2:
    st.subheader("🚨 Incident Details")
    incident_type = st.selectbox("Incident Type", valid_vals["incident_type"])
    collision_type = st.selectbox("Collision Type", valid_vals["collision_type"])
    incident_severity = st.selectbox("Incident Severity", valid_vals["incident_severity"])
    authorities_contacted = st.selectbox("Authorities Contacted", valid_vals["authorities_contacted"])
    incident_state = st.selectbox("Incident State", valid_vals["incident_state"])
    incident_city = st.text_input("Incident City", value="Charlesville",
                                   help="Enter the city name exactly as in the dataset")
    incident_hour = st.slider("Incident Hour (0-23)", min_value=0, max_value=23, value=14)
    number_of_vehicles = st.number_input("Number of Vehicles Involved", min_value=1, max_value=10, value=1)

with col3:
    st.subheader("📊 Claim Details")
    bodily_injuries = st.number_input("Bodily Injuries", min_value=0, max_value=10, value=0)
    witnesses = st.number_input("Witnesses", min_value=0, max_value=10, value=1)
    police_report = st.selectbox("Police Report Available?", valid_vals["police_report_available"])
    claim_amount = st.number_input("Claim Amount ($)", min_value=100.0, max_value=100000.0, value=8000.0, step=100.0)
    total_claim_amount = st.number_input("Total Claim Amount ($)", min_value=100.0, max_value=100000.0, value=9500.0, step=100.0)

    st.subheader("🔍 Engineered Features Preview")
    claim_ratio = total_claim_amount / (policy_annual_premium + 1)
    high_claim = "Yes ⚠️" if total_claim_amount > 14500 else "No ✅"
    night = "Yes ⚠️" if (incident_hour >= 20 or incident_hour <= 5) else "No ✅"
    many_veh = "Yes ⚠️" if number_of_vehicles > 2 else "No ✅"
    no_police = "Yes ⚠️" if police_report == "No" else "No ✅"

    st.metric("Claim-to-Premium Ratio", f"{claim_ratio:.2f}")
    st.write(f"🚨 High Claim Flag: **{high_claim}**")
    st.write(f"🌙 Night Incident: **{night}**")
    st.write(f"🚗 Many Vehicles: **{many_veh}**")
    st.write(f"🚓 No Police Report: **{no_police}**")

# ─── Predict Button ───────────────────────────────────────────
st.divider()
predict_col, _ = st.columns([1, 2])

with predict_col:
    predict_btn = st.button("🔍 Predict Fraud", type="primary", use_container_width=True)

if predict_btn:
    payload = {
        "policy_state": policy_state,
        "policy_deductible": policy_deductible,
        "policy_annual_premium": policy_annual_premium,
        "insured_age": insured_age,
        "insured_sex": insured_sex,
        "insured_education_level": insured_education_level,
        "insured_occupation": insured_occupation,
        "insured_hobbies": insured_hobbies,
        "incident_type": incident_type,
        "collision_type": collision_type,
        "incident_severity": incident_severity,
        "authorities_contacted": authorities_contacted,
        "incident_state": incident_state,
        "incident_city": incident_city,
        "incident_hour_of_the_day": incident_hour,
        "number_of_vehicles_involved": number_of_vehicles,
        "bodily_injuries": bodily_injuries,
        "witnesses": witnesses,
        "police_report_available": police_report,
        "claim_amount": claim_amount,
        "total_claim_amount": total_claim_amount,
    }

    with st.spinner("Analyzing claim..."):
        try:
            res = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if res.status_code == 200:
                result = res.json()
                st.divider()
                st.subheader("📊 Prediction Result")

                if result["prediction"] == 1:
                    st.error(f"### {result['label']}")
                    st.error(f"⚠️ This claim has a **{result['fraud_probability']}** probability of being **fraudulent**.")
                else:
                    st.success(f"### {result['label']}")
                    st.success(f"✅ This claim appears **legitimate** with only **{result['fraud_probability']}** fraud probability.")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Fraud Probability", result["fraud_probability"])
                col_b.metric("Not Fraud Probability", result["not_fraud_probability"])
                col_c.metric("Model Used", result["model_used"])

            else:
                error = res.json().get("detail", "Unknown error")
                st.error(f"❌ API Error: {error}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ─── Footer ───────────────────────────────────────────────────
st.divider()
