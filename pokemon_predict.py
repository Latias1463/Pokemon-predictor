import pandas as pd
import streamlit as st
import joblib

class PokemonStatsPredictor:
    def __init__(self):
        self.models = joblib.load('pokemon_meta_model.joblib')

    def predict(self, X_test):
        predictions = {}
        for stat, model in self.models.items():
            predictions[stat] = model.predict(X_test)
        return predictions

# Load and preprocess the dataset for reference (not for training)
df = pd.read_csv("Pokemon.csv")
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

# Function to apply nature adjustments
def apply_nature(stat_name, base_value, iv_value, ev_value):
    increase, decrease = nature_effects[nature]
    nature_multiplier = 1.1 if increase == stat_name else 0.9 if decrease == stat_name else 1.0
    
    if stat_name == 'HP':
        user_predicted_stat = (((2 * base_value + iv_value + math.floor(ev_value / 4)) * 100) / 100) + 100 + 10
    else:
        user_predicted_stat = (((2 * base_value + iv_value + math.floor(ev_value / 4)) * 100) / 100 + 5) * nature_multiplier
    
    return user_predicted_stat

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

    # Load the pre-trained model
    predictor = joblib.load('pokemon_meta_model.joblib')

    # Prepare the input for the hypothetical Pokémon
    input_data = {
        'HP': avg_stats_combined['HP'],
        'Attack': avg_stats_combined['Attack'],
        'Defense': avg_stats_combined['Defense'],
        'Sp. Atk': avg_stats_combined['Sp. Atk'],
        'Sp. Def': avg_stats_combined['Sp. Def'],
        'Speed': avg_stats_combined['Speed'],
        f'Type 1_{type_1}': 1,
    }
    if type_2:
        input_data[f'Type 2_{type_2}'] = 1

    # Convert input_data to a DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # One-hot encode the input_df to match the training data format
    input_df_encoded = pd.get_dummies(input_df)
    input_df_encoded = input_df_encoded.reindex(columns=X.columns, fill_value=0)

    # Predict the individual stats for the hypothetical Pokémon
    predicted_stats = predictor.predict(input_df_encoded)

    # Display the predicted stats
    st.write(f"Predicted Stats for {'Legendary' if is_legendary else 'Non-Legendary'} {type_1.capitalize()}/{type_2.capitalize() if type_2 else ''} Type Pokémon (Generation {generation}):")
    for stat, value in predicted_stats.items():
        st.write(f"  {stat}: {value[0]}")

    # Further adjustments based on nature, IVs, and EVs can be displayed here if needed
    
