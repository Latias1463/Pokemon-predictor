import pandas as pd
import streamlit as st
import joblib
import math
import matplotlib.pyplot as plt
import seaborn as sns

class PokemonStatsPredictor:
    def __init__(self):
        self.models = None
        self.expected_columns = None
        self.load_models()

    def load_models(self):
        try:
            model_data = joblib.load('pokemon_meta_model.joblib')
            self.models = model_data.models
            self.expected_columns = model_data.expected_columns
            
            # Debug: Print detailed information about the loaded models and columns
            print(f"Loaded models type: {type(self.models)}")
            print(f"Loaded models content: {self.models}")
            print(f"Loaded expected_columns type: {type(self.expected_columns)}")
            print(f"Loaded expected_columns content: {self.expected_columns}")
            
            # Check if models is a dictionary
            if not isinstance(self.models, dict):
                raise AttributeError("The loaded model does not contain a valid 'models' attribute.")
            if not isinstance(self.expected_columns, list):
                raise AttributeError("The loaded 'expected_columns' attribute is not a list or is empty.")
            
            print("Models and expected columns loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            self.models = {}
            self.expected_columns = []
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def predict(self, X_test):
        if not isinstance(self.models, dict):
            raise AttributeError("The 'models' attribute is not a dictionary.")
        
        predictions = {}
        for stat, model in self.models.items():
            if model:
                predictions[stat] = model.predict(X_test)
            else:
                predictions[stat] = [0]  # Default to 0 if model is missing
        return predictions

# Load and preprocess the dataset for reference (not for training)
# Load and preprocess the dataset for reference (not for training)
df = pd.read_csv("Pokemon.csv")
df['Type 1'] = df['Type 1'].str.lower()
df['Type 2'] = df['Type 2'].str.lower()
required_columns = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

st.title("Pokémon Stats Predictor")

st.markdown("""
Welcome to the Pokémon Stats Predictor! This tool helps you predict the stats of a hypothetical Pokémon based on various attributes.
""")

required_columns = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"The following required columns are missing from the dataset: {missing_columns}")
else:
    st.subheader("Distribution of Pokémon Stats")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[required_columns])
    plt.title('Distribution of Pokémon Stats')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Display the boxplot in Streamlit
    st.pyplot(plt.gcf())

st.header("Pokémon Characteristics")
st.subheader("Type Selection")


cols = st.columns(2)
type_1 = cols[0].selectbox("Select the primary type:", options=df['Type 1'].unique(), index=0).lower()
type_2 = cols[1].selectbox("Select the secondary type (if any):", options=["None"] + list(df['Type 1'].unique()), index=0).lower()
if type_2 == "none":
    type_2 = None

st.subheader("Generation and Legendary Status")
generation = st.slider("Select the generation (1-9):", min_value=1, max_value=9, step=1)
is_legendary = st.radio("Is the Pokémon legendary?", ["Yes", "No"]).lower() == 'yes'

st.header("Individual Values (IVs)")
st.markdown("IVs range from 0 to 31 and contribute to the potential of your Pokémon's stats.")
iv_cols = st.columns(6)
iv_hp = iv_cols[0].slider("HP", min_value=0, max_value=31, value=0)
iv_attack = iv_cols[1].slider("Attack", min_value=0, max_value=31, value=0)
iv_defense = iv_cols[2].slider("Defense", min_value=0, max_value=31, value=0)
iv_sp_atk = iv_cols[3].slider("Sp. Atk", min_value=0, max_value=31, value=0)
iv_sp_def = iv_cols[4].slider("Sp. Def", min_value=0, max_value=31, value=0)
iv_speed = iv_cols[5].slider("Speed", min_value=0, max_value=31, value=0)

iv_values = {'HP': iv_hp, 'Attack': iv_attack, 'Defense': iv_defense, 'Sp. Atk': iv_sp_atk, 'Sp. Def': iv_sp_def, 'Speed': iv_speed}

st.header("Effort Values (EVs)")
st.markdown("Total EVs across all stats cannot exceed 510. Each stat can receive up to 252 EVs, and EVs can be set in increments of 4.")
ev_cols = st.columns(6)
ev_hp = ev_cols[0].slider("HP", min_value=0, max_value=252, step=4, value=0)
ev_attack = ev_cols[1].slider("Attack", min_value=0, max_value=252, step=4, value=0)
ev_defense = ev_cols[2].slider("Defense", min_value=0, max_value=252, step=4, value=0)
ev_sp_atk = ev_cols[3].slider("Sp. Atk", min_value=0, max_value=252, step=4, value=0)
ev_sp_def = ev_cols[4].slider("Sp. Def", min_value=0, max_value=252, step=4, value=0)
ev_speed = ev_cols[5].slider("Speed", min_value=0, max_value=252, step=4, value=0)

total_ev_allocated = ev_hp + ev_attack + ev_defense + ev_sp_atk + ev_sp_def + ev_speed
if total_ev_allocated > 510:
    st.warning(f"Total EVs allocated exceed the limit of 510. Current total: {total_ev_allocated}")

ev_values = {'HP': ev_hp, 'Attack': ev_attack, 'Defense': ev_defense, 'Sp. Atk': ev_sp_atk, 'Sp. Def': ev_sp_def, 'Speed': ev_speed}

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
# Function to apply nature adjustments
def apply_nature(stat_name, base_value, iv_value, ev_value, is_worst=False, is_best=False):
    increase, decrease = nature_effects[nature]
    if is_worst:
        nature_multiplier = 0.9 
    elif is_best:
        nature_multiplier = 1.1
    else:
        nature_multiplier = 1.1 if increase == stat_name else 1.0
    
    if stat_name == 'HP':
        user_predicted_stat = math.floor((((2 * base_value + iv_value + math.floor(ev_value / 4)) * 100) / 100) + 100 + 10)
    else:
        user_predicted_stat = math.floor((((2 * base_value + iv_value + math.floor(ev_value / 4)) * 100) / 100 + 5) * nature_multiplier)
    
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
    predictor = PokemonStatsPredictor()

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
    
    # Ensure that all columns from the training data are present in the input data
    input_df_encoded = input_df_encoded.reindex(columns=predictor.expected_columns, fill_value=0)
    
    # Predict the individual stats for the hypothetical Pokémon
    predicted_stats = predictor.predict(input_df_encoded)
    
    # Display the predicted stats
    st.subheader("Predicted Stats")
    legendary_status = "Legendary" if is_legendary else "Non-Legendary"
    type_info = f"{type_1.capitalize()}{f' / {type_2.capitalize()}' if type_2 else ''}"
    st.write(f"This is a {legendary_status} {type_info} Pokémon from Generation {generation}.")
    st.write("---")
    st.write("### Stats Overview")

    total_base, total_best, total_worst, total_adjusted = 0, 0, 0, 0
    for stat in predicted_stats:
        base_stat = predicted_stats[stat][0]
        best_stat = apply_nature(stat, base_stat, 31, 252, is_best=True)  # IVs max at 31, EVs max at 252
        worst_stat = apply_nature(stat, base_stat, 0, 0, is_worst=True)    # IVs min at 0, EVs at 0
        actual_stat = apply_nature(stat, base_stat, iv_values[stat], ev_values[stat])
        st.markdown(f"""
        **{stat}:**
        - **Predicted base stat:** {base_stat}
        - **Best:** {best_stat}
        - **Worst:** {worst_stat}
        - **Adjusted:** {actual_stat} (based on selected IVs, EVs, and Nature)
        """)

        total_base += base_stat
        total_best += best_stat
        total_worst += worst_stat
        total_adjusted += actual_stat

    st.write(f"**Total Predicted Base Stats:** {total_base}")

