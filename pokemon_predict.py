import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load the pre-trained model
predictor = joblib.load('path_to_your_saved_model/pokemon_meta_model.joblib')

# Load the dataset for preprocessing and getting unique values for dropdowns
df = pd.read_csv("Pokemon.csv")
df['Type 1'] = df['Type 1'].str.lower()
df['Type 2'] = df['Type 2'].str.lower()

st.title("Pokémon Stats Predictor")

st.markdown("""
This tool helps you predict the stats of a hypothetical Pokémon based on its attributes.
""")

st.header("Pokémon Characteristics")
st.subheader("Type Selection")
types = list(df['Type 1'].unique())
type_1 = st.selectbox("Select the primary type:", options=types, index=0).lower()
type_2 = st.selectbox("Select the secondary type (if any):", options=["None"] + types, index=0).lower()
if type_2 == "none":
    type_2 = None

st.subheader("Generation and Legendary Status")
generation = st.slider("Select the generation (1-9):", min_value=1, max_value=9, step=1)
is_legendary = st.radio("Is the Pokémon legendary?", ["Yes", "No"]).lower() == 'yes'

st.header("Individual Values (IVs)")
st.markdown("**IVs range from 0 to 31 and contribute to the potential of your Pokémon's stats.**")
cols = st.columns(3)
iv_hp = cols[0].slider("Select IV for HP (0-31):", min_value=0, max_value=31, value=0)
iv_attack = cols[1].slider("Select IV for Attack (0-31):", min_value=0, max_value=31, value=0)
iv_defense = cols[2].slider("Select IV for Defense (0-31):", min_value=0, max_value=31, value=0)
iv_sp_atk = cols[0].slider("Select IV for Sp. Atk (0-31):", min_value=0, max_value=31, value=0)
iv_sp_def = cols[1].slider("Select IV for Sp. Def (0-31):", min_value=0, max_value=31, value=0)
iv_speed = cols[2].slider("Select IV for Speed (0-31):", min_value=0, max_value=31, value=0)

st.header("Effort Values (EVs)")
st.markdown("**Total EVs across all stats cannot exceed 510. Each stat can receive up to 252 EVs, and EVs can be set in increments of 4.**")
ev_hp = st.slider("Select EV for HP (0-252):", min_value=0, max_value=252, step=4, value=0)
ev_attack = st.slider("Select EV for Attack (0-252):", min_value=0, max_value=252, step=4, value=0)
ev_defense = st.slider("Select EV for Defense (0-252):", min_value=0, max_value=252, step=4, value=0)
ev_sp_atk = st.slider("Select EV for Sp. Atk (0-252):", min_value=0, max_value=252, step=4, value=0)
ev_sp_def = st.slider("Select EV for Sp. Def (0-252):", min_value=0, max_value=252, step=4, value=0)
ev_speed = st.slider("Select EV for Speed (0-252):", min_value=0, max_value=252, step=4, value=0)

total_ev_allocated = ev_hp + ev_attack + ev_defense + ev_sp_atk + ev_sp_def + ev_speed
if total_ev_allocated > 510:
    st.warning(f"Total EVs allocated exceed the limit of 510. Current total: {total_ev_allocated}")

st.header("Nature Selection")
nature = st.selectbox("Select Nature:", [
    'Hardy', 'Lonely', 'Brave', 'Adamant', 'Naughty',
    'Bold', 'Docile', 'Relaxed', 'Impish', 'Lax',
    'Timid', 'Hasty', 'Serious', 'Jolly', 'Naive',
    'Modest', 'Mild', 'Quiet', 'Bashful', 'Rash',
    'Calm', 'Gentle', 'Sassy', 'Careful', 'Quirky'
])

nature_effects = {
    'Hardy': (None, None), 'Lonely': ('Attack', 'Defense'), 'Brave': ('Attack', 'Speed'), 'Adamant': ('Attack', 'Sp. Atk'), 'Naughty': ('Attack', 'Sp. Def'),
    'Bold': ('Defense', 'Attack'), 'Docile': (None, None), 'Relaxed': ('Defense', 'Speed'), 'Impish': ('Defense', 'Sp. Atk'), 'Lax': ('Defense', 'Sp. Def'),
    'Timid': ('Speed', 'Attack'), 'Hasty': ('Speed', 'Defense'), 'Serious': (None, None), 'Jolly': ('Speed', 'Sp. Atk'), 'Naive': ('Speed', 'Sp. Def'),
    'Modest': ('Sp. Atk', 'Attack'), 'Mild': ('Sp. Atk', 'Defense'), 'Quiet': ('Sp. Atk', 'Speed'), 'Bashful': (None, None), 'Rash': ('Sp. Atk', 'Sp. Def'),
    'Calm': ('Sp. Def', 'Attack'), 'Gentle': ('Sp. Def', 'Defense'), 'Sassy': ('Sp. Def', 'Speed'), 'Careful': ('Sp. Def', 'Sp. Atk'), 'Quirky': (None, None)
}

# Prepare input data for prediction
if st.button("Predict Stats"):
    # Create a DataFrame with the input data
    input_data = {
        'HP': [0], 'Attack': [0], 'Defense': [0], 'Sp. Atk': [0], 'Sp. Def': [0], 'Speed': [0],  # Placeholder for actual stats
        f'Type 1_{type_1}': [1],
        f'Type 2_{type_2}': [1] if type_2 else [0],
        'Generation': [generation],
        'Legendary': [1 if is_legendary else 0]
    }
    input_df = pd.DataFrame(input_data)

    # Ensure the input DataFrame has the same columns as the training data
    input_df = input_df.reindex(columns=predictor.models['HP'].feature_names_in_, fill_value=0)

    # Predict stats
    predictions = predictor.predict(input_df)

    # Display the predicted stats
    st.write(f"Predicted Stats for {'Legendary' if is_legendary else 'Non-Legendary'} {type_1.capitalize()}/{type_2.capitalize() if type_2 else ''} Type Pokémon (Generation {generation}):")
    st.write(f"  HP: {predictions['HP'][0]}")
    st.write(f"  Attack: {predictions['Attack'][0]}")
    st.write(f"  Defense: {predictions['Defense'][0]}")
    st.write(f"  Sp. Atk: {predictions['Sp. Atk'][0]}")
    st.write(f"  Sp. Def: {predictions['Sp. Def'][0]}")
    st.write(f"  Speed: {predictions['Speed'][0]}")
