import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib

class PokemonStatsPredictor:
    def __init__(self):
        self.models = {
            'HP': DecisionTreeRegressor(random_state=42),
            'Attack': DecisionTreeRegressor(random_state=42),
            'Defense': DecisionTreeRegressor(random_state=42),
            'Sp. Atk': DecisionTreeRegressor(random_state=42),
            'Sp. Def': DecisionTreeRegressor(random_state=42),
            'Speed': DecisionTreeRegressor(random_state=42),
        }

    def train(self, X_train, y_train):
        for stat, model in self.models.items():
            model.fit(X_train, y_train[stat])

    def predict(self, X_test):
        predictions = {}
        for stat, model in self.models.items():
            predictions[stat] = model.predict(X_test)
        return predictions

# Load and preprocess the dataset
df = pd.read_csv("../data/Pokemon.csv")
df['Type 1'] = df['Type 1'].str.lower()
df['Type 2'] = df['Type 2'].str.lower()

st.title("Pokémon Stats Predictor")

st.markdown("""
This is the Pokémon Stats Predictor. This tool helps you predict the stats of a hypothetical Pokémon based on their attributes.
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
iv_defense = cols[2].slider("Select IV for Defense (0-31):", min_value=0, max=31, value=0)
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

# Main prediction and display logic
if st.button("Predict Stats"):
    # Filter the dataset based on user input
    if type_2:
        type_df = df[(df['Type 1'] == type_1) | (df['Type 2'] == type_1) | (df['Type 1'] == type_2) | (df['Type 2'] == type_2)]
    else:
        type_df = df[(df['Type 1'] == type_1) | (df['Type 2'] == type_1)]
        
    generation_df = df[df['Generation'] == generation]

    # Calculate the higher-end average stats for the specified types if the Pokémon is legendary
    if is_legendary:
        type_df = type_df.nlargest(int(len(type_df) * 0.25), 'Total')
        generation_df = generation_df.nlargest(int(len(generation_df) * 0.25), 'Total')

    average_stats_type = type_df.mean(numeric_only=True)
    average_stats_generation = generation_df.mean(numeric_only=True)
    legendary_df = df[df['Legendary'] == True]
    average_stats_legendary = legendary_df.mean(numeric_only=True)

    if is_legendary:
        avg_stats_combined = (average_stats_type + average_stats_generation + average_stats_legendary) / 3
        avg_stats_combined = (avg_stats_combined + average_stats_legendary) / 2 # use legendary twice to make the stats more like a legendary   
        combined_df = pd.concat([type_df, generation_df, legendary_df])
        combined_df_encoded = pd.get_dummies(combined_df, columns=['Type 1', 'Type 2'])
        X = combined_df_encoded.drop(['Total', 'Name', 'Legendary'], axis=1)
    else:
        avg_stats_combined = (average_stats_type + average_stats_generation) / 2
        combined_df = pd.concat([type_df, generation_df])
        combined_df_encoded = pd.get_dummies(combined_df, columns=['Type 1', 'Type 2'])
        X = combined_df_encoded.drop(['Total', 'Name'], axis=1)
    y = combined_df_encoded[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the meta-model
    predictor = PokemonStatsPredictor()
    predictor.train(X_train, y_train)

    # Save the trained meta-model
    joblib.dump(predictor, '../models/pokemon_meta_model.joblib')
