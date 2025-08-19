import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go

# --- Data Parsing Functions ---

REQUIRED_RECORD_TYPES = {
    'HKQuantityTypeIdentifierHeartRateVariabilitySDNN',
    'HKCategoryTypeIdentifierSleepAnalysis',
    'HKQuantityTypeIdentifierActiveEnergyBurned',
    'HKQuantityTypeIdentifierVO2Max',
    'HKQuantityTypeIdentifierHeartRate'
}
def enrich_workouts_with_distance(workouts_df, records_df):
    """
    Matches running workouts with their corresponding distance records to calculate total distance.
    """
    if workouts_df.empty or records_df.empty:
        print("No workouts or records data available.")
        return workouts_df

    # Filter for only the necessary data to speed up processing
    running_workouts = workouts_df[workouts_df['workoutActivityType'].str.contains("Running", na=False)].copy()
    distance_records = records_df[records_df['type'] == 'HKQuantityTypeIdentifierDistanceWalkingRunning'].copy()
    distance_records = distance_records[distance_records['sourceName'].str.contains("Watch", na=False)].copy()
    distance_records['value'] = distance_records['value'].astype(float)
    
    if running_workouts.empty or distance_records.empty:
       
        return workouts_df # No running workouts or no distance data to match

    # This will hold the calculated distances
    calculated_distances = {}
    
    for index, workout in running_workouts.iterrows():
        
        workout_start = workout['startDate']
        workout_end = workout['endDate']
        
        # Create a boolean mask to find all distance records within this workout's timeframe
        mask = (distance_records['startDate'] >= workout_start) & (distance_records['endDate'] <= workout_end)
        
        # Sum the 'value' (distance) of all matching records
        workout_distance = distance_records.loc[mask, 'value'].astype(float).sum()
        
        if workout_distance > 0:
            calculated_distances[index] = workout_distance 

    # Update the original workouts dataframe with the new, accurate distances
    if calculated_distances:
        # Create a Series from the dictionary to update the DataFrame
        if 'totalDistance' in workouts_df.columns:
            workouts_df.drop(columns=['totalDistance'], inplace=True, errors='ignore')
        distance_series = pd.Series(calculated_distances, name='totalDistance')
        workouts_df = workouts_df.join(distance_series, how='left')
        
    return workouts_df

@st.cache_data
def parse_apple_health_xml(uploaded_file):
    record_list = []
    workout_list = []
    try:
        for event, elem in ET.iterparse(uploaded_file, events=('end',)):
            if event == 'end':
                if elem.tag == 'Workout':
                    workout_list.append(elem.attrib)
                elif elem.tag == 'Record':
                    record_list.append(elem.attrib)
                elem.clear()

        workouts_df = pd.DataFrame(workout_list)
        records_df = pd.DataFrame(record_list)
        
        # Convert date columns for both dataframes
        for df in [workouts_df, records_df]:
            if not df.empty:
                for col in ['creationDate', 'startDate', 'endDate']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns for workouts
        if not workouts_df.empty:
            for col in ['duration', 'totalDistance', 'totalEnergyBurned']:
                if col in workouts_df.columns:
                    workouts_df[col] = pd.to_numeric(workouts_df[col], errors='coerce')
        
        # For records, convert value ONLY for non-sleep-analysis records.
        if not records_df.empty:
            # This is safer: we leave 'value' as an object and convert it in each
            # metric function as needed. The only truly essential numeric is duration_seconds for sleep.
            sleep_mask = records_df['type'] == 'HKCategoryTypeIdentifierSleepAnalysis'
            if sleep_mask.any():
                records_df.loc[sleep_mask, 'duration_seconds'] = (records_df.loc[sleep_mask, 'endDate'] - records_df.loc[sleep_mask, 'startDate']).dt.total_seconds()

        workouts_df = enrich_workouts_with_distance(workouts_df, records_df)
        return {'workouts': workouts_df, 'records': records_df}
    except Exception as e:
        st.error(f"An error occurred during parsing: {e}")
        return None
    
def find_personal_bests(workouts_df):
    """
    Scans workout data to find personal best times for standard race distances.
    """
    if workouts_df.empty or 'totalDistance' not in workouts_df.columns:
        return {}

    # Define race distances in kilometers
    races = {"5K": 5, "10K": 10, "Half-Marathon": 21.0975, "Marathon": 42.195, "50K": 50, "100K": 100}
    personal_bests = {}
    tolerance = 0.99
    for name, distance_km in races.items():
        race_attempts = workouts_df[
            (workouts_df['workoutActivityType'].str.contains("Running")) &
            (workouts_df['totalDistance'] >= distance_km * tolerance)
        ].copy()
        if not race_attempts.empty:
            race_attempts['pace_s_per_km'] = (race_attempts['duration'] * 60) / race_attempts['totalDistance']
            best_attempt = race_attempts.loc[race_attempts['pace_s_per_km'].idxmin()]
            best_time_seconds = best_attempt['pace_s_per_km'] * distance_km
            personal_bests[name] = best_time_seconds
    return personal_bests

# --- New Function: Time Formatting Helpers ---
def format_time_hms(seconds):
    if pd.isna(seconds) or seconds is None: return None
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    else:
        return f"{minutes:02d}:{sec:02d}"

def format_improvement(seconds):
    if pd.isna(seconds) or seconds is None: return None
    seconds = int(seconds)
    sign = "-" if seconds < 0 else "+"
    seconds = abs(seconds)
    minutes, sec = divmod(seconds, 60)
    return f"{sign}{minutes:02d}:{sec:02d}"

# --- Metric Calculation Functions (Unchanged) ---

def calculate_body_battery(health_data):
    if 'records' not in health_data or health_data['records'].empty: return 0, "Not enough data"
    df = health_data['records']
    hrv_df = df[df['type'] == 'HKQuantityTypeIdentifierHeartRateVariabilitySDNN'].copy()
    sleep_df = df[df['type'] == 'HKCategoryTypeIdentifierSleepAnalysis'].copy()
    active_energy_df = df[df['type'] == 'HKQuantityTypeIdentifierActiveEnergyBurned'].copy()

    if sleep_df.empty or active_energy_df.empty: return 0, "Missing Sleep or Activity data."

    # --- Model Constants ---
    MAX_DAILY_ACTIVITY_BURN = 2000; MAX_ACTIVITY_DEPLETION_POINTS = 50; PASSIVE_DRAIN_PER_HOUR = 3
    RECHARGE_POINTS = {'Deep': 20, 'REM': 12, 'Light': 6, 'Awake': -15} # Points per hour

    # --- Time Setup ---
    df_timezone = df['endDate'].dt.tz if 'endDate' in df else None
    today = pd.Timestamp.now(tz=df_timezone).normalize() if pd.Timestamp.now(tz=df_timezone).normalize() == df['endDate'].max().normalize() else df['endDate'].max().normalize()
    last_night_start = today - pd.DateOffset(hours=12); last_night_end = today + pd.DateOffset(hours=12)

    # --- Step 1: The Recharge Engine ---
    last_night_sleep = sleep_df[(sleep_df['startDate'] >= last_night_start) & (sleep_df['endDate'] < last_night_end)]
    if last_night_sleep.empty: return 0, "No sleep data for last night to calculate recharge."
    
    stage_durations_s = last_night_sleep.groupby('value')['duration_seconds'].sum()
    deep_h = stage_durations_s.get('HKCategoryValueSleepAnalysisAsleepDeep', 0) / 3600
    rem_h = stage_durations_s.get('HKCategoryValueSleepAnalysisAsleepREM', 0) / 3600
    light_h = stage_durations_s.get('HKCategoryValueSleepAnalysisAsleepCore', 0) / 3600
    awake_h = stage_durations_s.get('HKCategoryValueSleepAnalysisAwake', 0) / 3600

    # Calculate raw score from sleep stages
    raw_recharge_score = ((deep_h * RECHARGE_POINTS['Deep']) + 
                          (rem_h * RECHARGE_POINTS['REM']) + 
                          (light_h * RECHARGE_POINTS['Light']) + 
                          (awake_h * RECHARGE_POINTS['Awake']))

    # Apply HRV as a quality multiplier
    hrv_last_night = hrv_df[(hrv_df['endDate'] >= last_night_start) & (hrv_df['endDate'] < last_night_end)]
    if not hrv_last_night.empty:
        avg_hrv_raw = hrv_last_night['value'].astype(float).mean()
        unit = hrv_last_night['unit'].iloc[0] if 'unit' in hrv_last_night.columns and not hrv_last_night['unit'].empty else None
        if unit == 'ms': avg_hrv = avg_hrv_raw
        elif avg_hrv_raw < 1.0: avg_hrv = avg_hrv_raw * 1000
        else: avg_hrv = avg_hrv_raw
    else: avg_hrv = 45 # Use baseline if no HRV data

    hrv_multiplier = 1 + max(-0.25, min(0.25, (avg_hrv - 45) / 45)) # +/- 25% effect
    morning_battery_level = max(20, min(100, raw_recharge_score * hrv_multiplier))

    # --- Step 2: The Depletion Engine ---
    active_energy_today = active_energy_df[active_energy_df['endDate'] >= today]
    calories_burned_today = active_energy_today['value'].astype(float).sum()
    activity_depletion = (calories_burned_today / MAX_DAILY_ACTIVITY_BURN) * MAX_ACTIVITY_DEPLETION_POINTS

    now_corrected = pd.Timestamp.now(tz=df_timezone) if pd.Timestamp.now(tz=df_timezone) == df['endDate'].max() else df['endDate'].max()

    wake_up_time = last_night_sleep['endDate'].max()
    hours_awake = max(0, (now_corrected - wake_up_time).total_seconds() / 3600)
    passive_drain = hours_awake * PASSIVE_DRAIN_PER_HOUR
    total_depletion_today = activity_depletion + passive_drain
    
    # --- Step 3: Final Score ---
    current_body_battery = max(0, min(100, morning_battery_level - total_depletion_today)) 
    
    help_text = (f"Started day at: {morning_battery_level:.0f} pts (Recharge Score: {raw_recharge_score:.0f}, HRV Multiplier: {hrv_multiplier:.2f}x). "
                 f"Depletion today: {total_depletion_today:.0f} pts.")
    return int(current_body_battery), help_text


def calculate_sleep_coach(health_data):
    if 'records' not in health_data or health_data['records'].empty:
        return 0, "Not enough data"

    df = health_data['records']
    sleep_df = df[df['type'] == 'HKCategoryTypeIdentifierSleepAnalysis'].copy()
    if sleep_df.empty:
        return 0, "No sleep analysis data found."

    # Find the last night's sleep session
    df_timezone = sleep_df['startDate'].dt.tz
    today = pd.Timestamp.now(tz=df_timezone).normalize() if pd.Timestamp.now(tz=df_timezone).normalize() == df['endDate'].max().normalize() else df['endDate'].max().normalize()
    yesterday_noon = today - pd.DateOffset(hours=12) if df_timezone else pd.Timestamp.now().normalize() - pd.DateOffset(hours=12)
    last_night_sleep = sleep_df[sleep_df['startDate'] >= yesterday_noon].copy()
    if last_night_sleep.empty:
        return 0, "No sleep data recorded for last night."

    # --- Calculate Core Sleep Metrics ---
    stage_durations = last_night_sleep.groupby('value')['duration_seconds'].sum()
    
    time_in_deep_s = stage_durations.get('HKCategoryValueSleepAnalysisAsleepDeep', 0)
    time_in_rem_s = stage_durations.get('HKCategoryValueSleepAnalysisAsleepREM', 0)
    time_in_light_s = stage_durations.get('HKCategoryValueSleepAnalysisAsleepCore', 0)
    time_awake_s = stage_durations.get('HKCategoryValueSleepAnalysisAwake', 0)
    
    total_asleep_s = time_in_deep_s + time_in_rem_s + time_in_light_s
    total_asleep_h = total_asleep_s / 3600
    
    if total_asleep_s == 0:
        return 0, "No asleep stages were recorded."

    # --- Component Scoring (out of 100) ---
    
    # 1. Total Sleep Duration Score (30 points)
    duration_score = min(1, total_asleep_h / 8.0) * 30

    # 2. Deep Sleep Score (30 points)
    deep_percentage = time_in_deep_s / total_asleep_s
    deep_target = 0.18 # Target 18%
    # Score is 1 at target, 0 if it's twice the target distance away
    deep_score = max(0, 1 - abs(deep_percentage - deep_target) / deep_target) * 30

    # 3. REM Sleep Score (20 points)
    rem_percentage = time_in_rem_s / total_asleep_s
    rem_target = 0.22 # Target 22%
    rem_score = max(0, 1 - abs(rem_percentage - rem_target) / rem_target) * 20
    
    # 4. Sleep Continuity Score (20 points)
    # Score based on total time awake during the sleep session
    continuity_score = max(0, 1 - (time_awake_s / 3600)) * 20 # Perfect score for <1 min awake, 0 for >60min

    # --- Final Score ---
    total_score = int(duration_score + deep_score + rem_score + continuity_score)

    help_text = (f"Score Breakdown:\n"
                 f"- Duration: {duration_score:.0f}/30 ({total_asleep_h:.1f}h)\n"
                 f"- Deep Sleep: {deep_score:.0f}/30 ({deep_percentage:.1%})\n"
                 f"- REM Sleep: {rem_score:.0f}/20 ({rem_percentage:.1%})\n"
                 f"- Continuity: {continuity_score:.0f}/20 ({time_awake_s/60:.0f}m awake)")

    return total_score, help_text

def calculate_race_predictor(health_data):
    if 'records' not in health_data or health_data['records'].empty: return pd.DataFrame(), "Not enough data"
    
    vo2_max_df = health_data['records'][health_data['records']['type'] == 'HKQuantityTypeIdentifierVO2Max'].copy().sort_values('endDate', ascending=False)
    vo2_max_df['value'] = vo2_max_df['value'].astype(float)
    if vo2_max_df.empty: return pd.DataFrame(), "No VO₂ Max data found."

    latest_vo2_max = vo2_max_df.iloc[0]['value']
    
    # Find Personal Bests
    personal_bests = find_personal_bests(health_data['workouts'])
    
    # Predict times based on VO2 Max
    try:
        base_5k_seconds = (13.3 - (0.28 * latest_vo2_max)) * 60
    except Exception:
        return pd.DataFrame(), "Could not calculate race times from VO₂ Max."

    races = {"5K": 5, "10K": 10, "Half-Marathon": 21.0975, "Marathon": 42.195, "50K": 50, "100K": 100}
    predictions = []

    for name, distance_km in races.items():
        predicted_improvement_sec = base_5k_seconds * (distance_km / 5)**1.06
        print(f"Predicted {name} time: {predicted_improvement_sec:.2f} seconds")
        pb_seconds = personal_bests.get(name)
        predicted_time_sec = pb_seconds + predicted_improvement_sec if pb_seconds else None
        predictions.append({
            "Distance": name,
            "Personal Best": format_time_hms(pb_seconds),
            "Predicted Improvement": format_improvement(predicted_improvement_sec),
            "Predicted Time": format_time_hms(predicted_time_sec)
        })

    prediction_df = pd.DataFrame(predictions)
    prediction_df.dropna(subset=['Personal Best'], inplace=True)  # Remove rows with no pb time
    info = f"Predictions based on a current VO₂ Max of {latest_vo2_max:.1f}."
    return prediction_df, info

def calculate_hrv_status(hrv_df):
    """
    Calculates HRV status by comparing the last 7 days to the last 60 days.
    """
    if hrv_df.empty or len(hrv_df) < 10: # Need enough data to establish a baseline
        return "No Status", 0, 0

    # For robustness, ensure timezone consistency
    df_timezone = hrv_df['endDate'].dt.tz
    today = pd.Timestamp.now(tz=df_timezone).normalize() if pd.Timestamp.now(tz=df_timezone).normalize() == hrv_df['endDate'].max().normalize() else hrv_df['endDate'].max().normalize()
    
    # Define time windows
    seven_days_ago = today - pd.Timedelta(days=7)
    sixty_days_ago = today - pd.Timedelta(days=60)
    
    # Filter for morning HRV (e.g., before 8 AM) for better accuracy
    morning_hrv = hrv_df[hrv_df['endDate'].dt.hour < 8]
    if len(morning_hrv) < 5: morning_hrv = hrv_df # Fallback if not enough morning readings

    # Calculate averages
    recent_avg = morning_hrv[morning_hrv['endDate'] >= seven_days_ago]['value'].astype(float).mean()
    baseline_avg = morning_hrv[(morning_hrv['endDate'] >= sixty_days_ago) & (morning_hrv['endDate'] < seven_days_ago)]['value'].astype(float).mean()

    if pd.isna(recent_avg) or pd.isna(baseline_avg):
        return "No Status", 0, 0
    else:
        # Determine status based on deviation from baseline
        if recent_avg < baseline_avg * 0.9: # More than 10% below baseline
            return "Strained", recent_avg, baseline_avg
        elif recent_avg > baseline_avg * 1.1: # More than 10% above might be good recovery or a sign of fatigue
            return "Unbalanced", recent_avg, baseline_avg
        else:
            return "Balanced", recent_avg, baseline_avg


def calculate_training_status(health_data):
    if 'workouts' not in health_data or health_data['workouts'].empty: return "No Status", "Not enough workout data.",0,0,0,0
    
    workouts_df = health_data['workouts'].copy()
    vo2_max_df = health_data['records'][health_data['records']['type'] == 'HKQuantityTypeIdentifierVO2Max'].copy().sort_values('endDate', ascending=False)
    hrv_df = health_data['records'][health_data['records']['type'] == 'HKQuantityTypeIdentifierHeartRateVariabilitySDNN'].copy()
    hrv_df['value'] = hrv_df['value'].astype(float)
    vo2_max_df['value'] = vo2_max_df['value'].astype(float)

    # --- Pillar 1: VO2 Max Trend ---
    vo2_max_trend = "Stable"
    if len(vo2_max_df) < 2: return "No Status", "A VO2 Max reading is required.",0,0,0,0
    latest_vo2_max = vo2_max_df.iloc[0]['value']
    four_weeks_ago = vo2_max_df.iloc[0]['endDate'] - pd.Timedelta(days=28)
    past_vo2_max_readings = vo2_max_df[vo2_max_df['endDate'] < four_weeks_ago]
    if not past_vo2_max_readings.empty:
        past_vo2_max = past_vo2_max_readings.iloc[0]['value']
        if latest_vo2_max > past_vo2_max + 0.5: vo2_max_trend = "Increasing"
        elif latest_vo2_max < past_vo2_max - 0.5: vo2_max_trend = "Decreasing"

    # --- Pillar 2: Training Load (ACWR) ---
    if 'totalEnergyBurned' in workouts_df.columns:
        energy_used = workouts_df['totalEnergyBurned'].fillna(workouts_df['duration'] / 60 * 5)
    else: energy_used = workouts_df['duration'] / 60 * 5
    workouts_df['trainingLoad'] = energy_used
    
    daily_load = workouts_df.set_index('endDate')['trainingLoad'].resample('D').sum().sort_index()
    if len(daily_load) < 14: return "No Status", "At least two weeks of training data is needed.",0,0,0,0
    
    short_term_sum = daily_load.rolling(window=7).sum().iloc[-1]
    long_term_sum = daily_load.rolling(window=28).sum().iloc[-1]
    
    if pd.isna(long_term_sum) or long_term_sum == 0: return "No Status", "Not enough long-term training data.", 0, 0,0,0

    short_term_avg = daily_load.rolling(window=7).mean().iloc[-1]
    long_term_avg = daily_load.rolling(window=28).mean().iloc[-1]
    if pd.isna(long_term_avg) or long_term_avg == 0: return "No Status", "Not enough long-term training data.",0,0,0,0
    
    load_ratio = short_term_avg / long_term_avg
    load_status = "Optimal"
    if load_ratio > 1.5: load_status = "High"
    elif load_ratio < 0.8: load_status = "Low"
    
    # --- Pillar 3: HRV Status ---
    hrv_status, _, _ = calculate_hrv_status(hrv_df)

    # --- The Decision Tree ---
    has_trained_last_week = daily_load.iloc[-7:].sum() > 0
    if not has_trained_last_week and vo2_max_trend == "Decreasing":
            status, explanation = "Detraining", "You have stopped training and your fitness is decreasing."

    elif load_status == "High":
        if vo2_max_trend == "Decreasing":
            status, explanation = "Strained", "Your training load is very high, causing your fitness to decrease. This is a strong sign of overreaching."
        else: # Covers "Stable" and "Increasing" VO2 Max
            status, explanation = "Strained", "Your training load is very high and likely unsustainable. Prioritize recovery to avoid burnout, even if fitness is currently stable or increasing."

    elif load_status == "Low":
        if vo2_max_trend in ["Increasing", "Stable"]:
                status, explanation = "Peaking", "Your load is reduced, allowing your body to recover for optimal performance. Ideal for a race."
        else: # Covers "Decreasing" VO2 Max
                status, explanation = "Recovery", "Your light training load is allowing your body to recover, but your fitness may be slightly declining."

    elif vo2_max_trend == "Increasing": # This now only runs if load_status is "Optimal"
        if hrv_status == "Strained":
            status, explanation = "Productive", "Your fitness is improving, but ensure you are getting enough recovery (HRV is strained)."
        status, explanation = "Productive", "Your fitness is improving! Your current training load is effective."

    elif vo2_max_trend == "Decreasing": # This now only runs if load_status is "Optimal"
        if hrv_status == "Strained":
            status, explanation = "Unproductive", "You are training, but your fitness is decreasing. Your body is struggling to recover (HRV is strained)."
        status, explanation = "Unproductive", "You are training, but your fitness is decreasing. Consider your recovery, sleep, and nutrition."

    else: # Default case: Optimal load and Stable VO2 Max
        status, explanation =  "Maintaining", "Your current training load is enough to maintain your fitness level."
    return status, explanation, short_term_sum, long_term_sum, vo2_max_trend, hrv_status

def create_training_load_gauge(short_term_load, long_term_load,status):
    """
    Creates a Plotly gauge chart to visualize the 7-day training load
    against the personalized optimal range.
    """
    if long_term_load == 0: return None

    chronic_daily_avg = long_term_load / 28
    lower_bound = chronic_daily_avg * 7 * 0.8
    upper_bound = chronic_daily_avg * 7 * 1.5
    gauge_max = upper_bound * 1.5 # Extend the axis for high loads

    # --- Dynamic Configuration based on Status ---
    title_text = f"Training Status: {status}"
    colors = ['lightskyblue', 'green', 'orange'] # Default: Low, Optimal, High

    if status == "Peaking":
        title_text = "Status: Peaking (Load is optimally low)"
        colors = ['green', 'lightgray', 'lightgray'] # Low zone is now the target
    elif status == "Strained":
        title_text = "Status: Strained (Load is unsustainably high)"
        colors = ['lightskyblue', 'green', 'red'] # High zone is now red for danger
    elif status == "Productive":
        title_text = "Status: Productive (Load is in the sweet spot)"
    elif status == "Recovery":
        title_text = "Status: Recovery (Load is low to allow adaptation)"
        colors = ['limegreen', 'lightgray', 'lightgray'] # Low zone is the target
    elif status == "Unproductive":
        title_text = "Status: Unproductive (Optimal load, but fitness is decreasing)"
        # Use muted colors to indicate a problem
        colors = ['#ADD8E6', '#90EE90', '#FFA500']


    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=short_term_load,
        title={'text': title_text, 'font': {'size': 20}},
        number={'suffix': " kcal (7-day total)"},
        gauge={
            'axis': {'range': [0, gauge_max]},
            'bar': {'color': "black", 'thickness': 0.2},
            'steps': [
                {'range': [0, lower_bound], 'color': colors[0], 'name': 'Low'},
                {'range': [lower_bound, upper_bound], 'color': colors[1], 'name': 'Optimal'},
                {'range': [upper_bound, gauge_max], 'color': colors[2], 'name': 'High'}
            ],
        }))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))

    return fig
# --- Main App UI ---
def main():
    st.set_page_config(layout="wide", page_title="Health Dashboard")
    st.title("Apple Health Advanced Dashboard")
    st.write("An app to analyze Apple Health data and provide Garmin-style metrics and insights.")

    st.sidebar.header("Instructions")
    st.sidebar.info(
        "1. **Export your data:** In the Apple Health app, tap your profile picture > Export All Health Data.\n"
        "2. **Unzip the file** to find `export.xml`.\n"
        "3. **Upload `export.xml` below.**"
    )

    uploaded_file = st.sidebar.file_uploader("Upload your export.xml file", type=["xml"])

    if uploaded_file is not None:
        with st.spinner('Parsing your health data... This may take a moment for large files.'):
            health_data = parse_apple_health_xml(uploaded_file)
        
        if health_data:
            st.success("Successfully parsed your health data!")
            st.header("Your Daily Snapshot")
            col1, col2, col3 = st.columns(3)
            with col1:
                score, text = calculate_body_battery(health_data)
                st.metric("Body Battery", f"{score}/100", help=text)
            with col2:
                score, text = calculate_sleep_coach(health_data)
                
                st.metric("Sleep Score", f"{score}/100", help=text)
            with col3:
                st.metric("Stress Level", "Coming Soon", help="Stress tracking requires consistent HRV readings throughout the day.")

            st.markdown("---")
            st.header("Training & Performance")
            tab1, tab2, tab3= st.tabs(["Training Status", "Race Predictor", "Recovery Time"])
            
            with tab1:
                st.subheader("Training Status")
                status, explanation, short_term_load, long_term_load,vo2_max_trend, hrv_status = calculate_training_status(health_data)

                if status != "No Status":
                    st.metric("Current Training Status", status)
                    st.info(explanation)
                    
                    st.markdown("---")
                    st.write("**Key Indicators Driving Your Status:**")
                    
                    cols = st.columns(3)
                    cols[0].metric("Acute Load (7-Day Total)", f"{short_term_load:.0f} kcal")
                    cols[1].metric("VO₂ Max Trend", vo2_max_trend)
                    cols[2].metric("HRV Status", hrv_status)
                else:
                    st.warning(f"**Could not determine Training Status:** {explanation}")


                
            with tab2:
                st.subheader("Race Predictor")
                prediction_df, info_text = calculate_race_predictor(health_data)
                
                if not prediction_df.empty:
                    st.table(prediction_df.set_index('Distance'))
                    st.caption(info_text)
                else:
                    st.warning(info_text)

            with tab3:
                st.subheader("Recovery Time")
                if 'workouts' in health_data and not health_data['workouts'].empty:
                    # ===== FIX: Sort by 'endDate' to get the TRUE last workout =====
                    last_workout = health_data['workouts'].sort_values('endDate', ascending=False).iloc[0]
                    
                    energy_burned = last_workout.get('totalEnergyBurned')
                    workout_duration_min = last_workout['duration']
                    
                    if pd.notna(energy_burned) and energy_burned > 0:
                        # Primary calculation based on energy burned (more accurate)
                        recovery_hours = np.ceil(energy_burned / 25) 
                    else: 
                        # Fallback if no energy data, based on duration
                        recovery_hours = np.ceil(workout_duration_min * 0.2) # e.g., 60 min workout -> 12 hours recovery
                    
                    st.metric("Recommended Recovery", f"{min(recovery_hours, 72):.0f} hours")
                    st.caption(f"Based on your last workout: a {workout_duration_min:.0f} min {last_workout['workoutActivityType'].replace('HKWorkoutActivityType', '')}.")
                else:
                    st.info("No workout data to calculate recovery time.")
            
            st.markdown("---")
            with st.expander("How are these metrics calculated? Click here for a detailed explanation."):
                st.markdown("""
                ### **Metric Calculation: Methodology & Rationale**

                This section details the logic used to approximate Garmin's proprietary metrics using Apple Health data. These are estimations based on sports science principles and should be used as a guide.

                ---
                #### **1. Body Battery**
                - **Concept:** Represents your energy level on a scale of 0-100. It recharges during rest and depletes with activity and stress.
                - **Apple Health Data Used:** Sleep duration (`HKCategoryTypeIdentifierSleepAnalysis`), Heart Rate Variability (`HKQuantityTypeIdentifierHeartRateVariabilitySDNN`), and Active Energy Burned (`HKQuantityTypeIdentifierActiveEnergyBurned`).
                - **Methodology & Rationale:**
                    - **Recharge:** Your "recharge" is calculated based on your last sleep session. It starts with a base value for sleep duration (up to 8 hours) and is then modified by your average Heart Rate Variability (HRV) during that sleep. A higher HRV than your baseline suggests better recovery and results in a bigger recharge.
                    - **Depletion:** This has two components:
                        1.  **Activity Depletion:** Calculated from the total active calories you've burned today. Every ~40 calories burned depletes the battery by about 1 point.
                        2.  **Passive Drain:** A constant, low-level drain is applied for every hour you are awake to simulate the general stress of being alive.
                - **Limitations:** This is a daily snapshot, not a continuous model like Garmin's. It roughly estimates your energy based on the previous night's recovery and today's activities.

                ---
                #### **2. Sleep Score**
                - **Concept:** A score from 0-100 that quantifies the quality of your previous night's sleep.
                - **Apple Health Data Used:** Sleep analysis records (`HKCategoryTypeIdentifierSleepAnalysis`), specifically time in bed vs. time asleep.
                - **Methodology & Rationale:**
                    - The score is a weighted average of two key factors:
                        1.  **Duration (60% weight):** How long you were asleep, scored against a target of 8 hours.
                        2.  **Efficiency (40% weight):** The percentage of time you were actually asleep while in bed. High efficiency means less restlessness.
                    - Both factors are crucial; long but restless sleep is not as restorative as shorter, more efficient sleep.

                ---
                #### **3. Race Predictor**
                - **Concept:** Estimates your race times for various distances based on your current fitness level.
                - **Apple Health Data Used:** VO2 Max (`HKQuantityTypeIdentifierVO2Max`).
                - **Methodology & Rationale:**
                    - **VO2 Max as a Fitness Benchmark:** VO2 Max is a strong indicator of your maximal aerobic capacity. Higher VO2 Max values correlate strongly with faster distance running potential.
                    - **Initial 5K Prediction:** We use a formula derived from established running tables to convert your latest VO2 Max value into a predicted 5k time.
                    - **Extrapolation:** The predicted 5k time is then extrapolated to other distances (10k, Half, Marathon) using **Riegel's endurance model**. This model uses an exponent (1.06) to account for the fact that you cannot hold the same pace for longer distances.
                    - **Comparison:** The calculation is performed for your latest VO2 Max and your latest reading from approximately one month prior to show your performance trend.

                ---
                #### **4. Training Status & Load Balance**
                - **Concept:** Assesses the effectiveness of your training and whether you are balancing fitness and fatigue appropriately.
                - **Apple Health Data Used:** Workout history (`Workout`), specifically duration and total energy burned.
                - **Methodology & Rationale:**
                    - **Training Load:** Each workout is assigned a "load" score, estimated from the total energy burned (in kcal).
                    - **Acute Load (Fatigue):** The **total** load of all workouts over the last 7 days. This represents your current-week fatigue.
                    - **Chronic Load (Fitness):** The **total** load of all workouts over the last 28 days. This represents your foundational fitness base.
                    - **Training Load Ratio (ACWR):** The ratio of your 7-day *daily average* load to your 28-day *daily average* load is the key metric. A ratio between **0.8 and 1.3** is considered the "sweet spot" for building fitness without excessive injury risk. Ratios above 1.5 suggest you are ramping up very quickly ("Strained"), while ratios below 0.8 suggest you are tapering or undertraining. Your **Training Status** ("Productive", "Maintaining", etc.) is derived from this ratio.

                ---
                #### **5. Recovery Time**
                - **Concept:** An estimate of the time your body needs to recover before your next high-effort workout.
                - **Apple Health Data Used:** Your most recent workout's total energy burned or duration.
                - **Methodology & Rationale:**
                    - This is a simple heuristic based on the training load (energy burned) of your last workout. The more energy you expended, the more micro-trauma occurred, and the longer your body needs to repair and adapt. The model recommends approximately 1 hour of recovery for every 25 kcal burned, capped at 72 hours.
                - **Limitations:** This is a very simplified model. True recovery is also affected by sleep quality, nutrition, and stress, which are not fully accounted for here.
                """)
    else:
        st.info("Upload your Apple Health XML file to get started.")

if __name__ == "__main__":
    main()