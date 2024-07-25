import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import math

# Load and preprocess the dataset
df = pd.read_csv(r"C:\Users\Latia\OneDrive\Documents\pokemon.csv")
df['Type 1'] = df['Type 1'].str.lower()
df['Type 2'] = df['Type 2'].str.lower()

st.title("Pokémon Stats Predictor")

# Dropdown for type selection
types = list(df['Type 1'].unique())
type_1 = st.selectbox("Select the primary type:", options=types, index=0).lower()
type_2 = st.selectbox("Select the secondary type (if any):", options=["None"] + types, index=0).lower()
if type_2 == "none":
    type_2 = None

# Scroller for generation selection
generation = st.slider("Select the generation (1-9):", min_value=1, max_value=9, step=1)
is_legendary = st.selectbox("Is the Pokémon legendary?", ["yes", "no"]).lower() == 'yes'

# Sliders for individual IVs
iv_hp = st.slider("Select IV for HP (0-31):", min_value=0, max_value=31, value=0)
iv_attack = st.slider("Select IV for Attack (0-31):", min_value=0, max_value=31, value=0)
iv_defense = st.slider("Select IV for Defense (0-31):", min_value=0, max_value=31, value=0)
iv_sp_atk = st.slider("Select IV for Sp. Atk (0-31):", min_value=0, max_value=31, value=0)
iv_sp_def = st.slider("Select IV for Sp. Def (0-31):", min_value=0, max_value=31, value=0)
iv_speed = st.slider("Select IV for Speed (0-31):", min_value=0, max_value=31, value=0)

# Dropdown for Nature selection
nature = st.selectbox("Select Nature:", [
    'Hardy', 'Lonely', 'Brave', 'Adamant', 'Naughty',
    'Bold', 'Docile', 'Relaxed', 'Impish', 'Lax',
    'Timid', 'Hasty', 'Serious', 'Jolly', 'Naive',
    'Modest', 'Mild', 'Quiet', 'Bashful', 'Rash',
    'Calm', 'Gentle', 'Sassy', 'Careful', 'Quirky'
])

# Mapping of nature effects on stats
nature_effects = {
    'Hardy': (None, None), 'Lonely': ('Attack', 'Defense'), 'Brave': ('Attack', 'Speed'), 'Adamant': ('Attack', 'Sp. Atk'), 'Naughty': ('Attack', 'Sp. Def'),
    'Bold': ('Defense', 'Attack'), 'Docile': (None, None), 'Relaxed': ('Defense', 'Speed'), 'Impish': ('Defense', 'Sp. Atk'), 'Lax': ('Defense', 'Sp. Def'),
    'Timid': ('Speed', 'Attack'), 'Hasty': ('Speed', 'Defense'), 'Serious': (None, None), 'Jolly': ('Speed', 'Sp. Atk'), 'Naive': ('Speed', 'Sp. Def'),
    'Modest': ('Sp. Atk', 'Attack'), 'Mild': ('Sp. Atk', 'Defense'), 'Quiet': ('Sp. Atk', 'Speed'), 'Bashful': (None, None), 'Rash': ('Sp. Atk', 'Sp. Def'),
    'Calm': ('Sp. Def', 'Attack'), 'Gentle': ('Sp. Def', 'Defense'), 'Sassy': ('Sp. Def', 'Speed'), 'Careful': ('Sp. Def', 'Sp. Atk'), 'Quirky': (None, None)
}

total_evs = 510
ev_hp = st.slider("Select EV for HP (0-252):", min_value=0, max_value=252, value=0)
ev_attack = st.slider("Select EV for Attack (0-252):", min_value=0, max_value=252, value=0)
ev_defense = st.slider("Select EV for Defense (0-252):", min_value=0, max_value=252, value=0)
ev_sp_atk = st.slider("Select EV for Sp. Atk (0-252):", min_value=0, max_value=252, value=0)
ev_sp_def = st.slider("Select EV for Sp. Def (0-252):", min_value=0, max_value=252, value=0)
ev_speed = st.slider("Select EV for Speed (0-252):", min_value=0, max_value=252, value=0)

# Check if total EVs exceed 510 and adjust accordingly
total_ev_allocated = ev_hp + ev_attack + ev_defense + ev_sp_atk + ev_sp_def + ev_speed
if total_ev_allocated > total_evs:
    st.warning(f"Total EV cannot exceed 510. Please adjust the EVs to not exceed the limit. Current total: {total_ev_allocated}")

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

    # Calculate the average stats for the specified generation
    average_stats_generation = generation_df.mean(numeric_only=True)

    # Calculate the average stats for all legendary Pokémon
    legendary_df = df[df['Legendary'] == True]
    average_stats_legendary = legendary_df.mean(numeric_only=True)

    # Combine average stats to create a hypothetical Pokémon
    avg_stats_combined = (average_stats_type + average_stats_generation + average_stats_legendary) / 3

    # Adjust the combined average stats if the Pokémon is legendary
    if is_legendary:
        avg_stats_combined = (avg_stats_combined + average_stats_legendary) / 2

    # Prepare the dataset for model training (using the combined type, generation, and legendary Pokémon dataset)
    combined_df = pd.concat([type_df, generation_df, legendary_df])
    combined_df_encoded = pd.get_dummies(combined_df, columns=['Type 1', 'Type 2'])
    X = combined_df_encoded.drop(['Total', 'Name', 'Legendary'], axis=1)
    y = combined_df_encoded[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Decision Tree Regressor for individual stats
    dt_hp = DecisionTreeRegressor(random_state=42)
    dt_attack = DecisionTreeRegressor(random_state=42)
    dt_defense = DecisionTreeRegressor(random_state=42)
    dt_sp_atk = DecisionTreeRegressor(random_state=42)
    dt_sp_def = DecisionTreeRegressor(random_state=42)
    dt_speed = DecisionTreeRegressor(random_state=42)

    dt_hp.fit(X_train, y_train['HP'])
    dt_attack.fit(X_train, y_train['Attack'])
    dt_defense.fit(X_train, y_train['Defense'])
    dt_sp_atk.fit(X_train, y_train['Sp. Atk'])
    dt_sp_def.fit(X_train, y_train['Sp. Def'])
    dt_speed.fit(X_train, y_train['Speed'])

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
    predicted_hp = dt_hp.predict(input_df_encoded)[0]
    predicted_attack = dt_attack.predict(input_df_encoded)[0]
    predicted_defense = dt_defense.predict(input_df_encoded)[0]
    predicted_sp_atk = dt_sp_atk.predict(input_df_encoded)[0]
    predicted_sp_def = dt_sp_def.predict(input_df_encoded)[0]
    predicted_speed = dt_speed.predict(input_df_encoded)[0]

    # Calculate the total stats as the sum of individual stats
    predicted_total_stats = predicted_hp + predicted_attack + predicted_defense + predicted_sp_atk + predicted_sp_def + predicted_speed

    # Function to calculate best and worst stats at level 100
    def calculate_stats(predicted_stat, stat_type):
        if stat_type == 'hp':
            worst_stat = ((2 * predicted_stat * 100) / 100 + 50 + 10)
            best_stat = (((2 * predicted_stat + 31 + 63) * 100) / 100 + 50 + 10)  
        else:
            worst_stat = ((2 * predicted_stat * 100) / 100 + 5) * 0.9
            best_stat = (((2 * predicted_stat + 31 + 252 / 4) * 100) / 100 + 5) * 1.1 
        return math.floor(worst_stat), math.floor(best_stat)

    worst_hp, best_hp = calculate_stats(predicted_hp, 'hp')
    worst_attack, best_attack = calculate_stats(predicted_attack, 'attack')
    worst_defense, best_defense = calculate_stats(predicted_defense, 'defense')
    worst_sp_atk, best_sp_atk = calculate_stats(predicted_sp_atk, 'sp_atk')
    worst_sp_def, best_sp_def = calculate_stats(predicted_sp_def, 'sp_def')
    worst_speed, best_speed = calculate_stats(predicted_speed, 'speed')

    # Adjust predicted stats based on nature, IVs, and EVs
    def apply_nature(stat_name, base_value, iv_value, ev_value):
        increase, decrease = nature_effects[nature]
        nature_multiplier = 1.1 if increase == stat_name else 0.9 if decrease == stat_name else 1.0
        
        if stat_name == 'HP':
            user_predicted_stat = math.floor((((2 * base_value + iv_value + math.floor(ev_value / 4)) * 100) / 100) + 50 + 10)
        else:
            user_predicted_stat = math.floor((((2 * base_value + iv_value + math.floor(ev_value / 4)) * 100) / 100 + 5) * nature_multiplier)
        
        return user_predicted_stat

    nature_adjusted_hp = apply_nature('HP', predicted_hp, iv_hp, ev_hp)
    nature_adjusted_attack = apply_nature('Attack', predicted_attack, iv_attack, ev_attack)
    nature_adjusted_defense = apply_nature('Defense', predicted_defense, iv_defense, ev_defense)
    nature_adjusted_sp_atk = apply_nature('Sp. Atk', predicted_sp_atk, iv_sp_atk, ev_sp_atk)
    nature_adjusted_sp_def = apply_nature('Sp. Def', predicted_sp_def, iv_sp_def, ev_sp_def)
    nature_adjusted_speed = apply_nature('Speed', predicted_speed, iv_speed, ev_speed)

    # Display the predicted stats and best/worst calculations
    if type_2 == None:
        st.write(f"Predicted Stats for {'Legendary' if is_legendary else 'Non-Legendary'} {type_1.capitalize()} Type Pokémon (Generation {generation}):")
    else:
       st.write(f"Predicted Stats for {'Legendary' if is_legendary else 'Non-Legendary'} {type_1.capitalize()}/{type_2.capitalize() } Type Pokémon (Generation {generation}):")
    st.write(f"  HP: {predicted_hp}")
    st.write(f"  Attack: {predicted_attack}")
    st.write(f"  Defense: {predicted_defense}")
    st.write(f"  Sp. Atk: {predicted_sp_atk}")
    st.write(f"  Sp. Def: {predicted_sp_def}")
    st.write(f"  Speed: {predicted_speed}")
    st.write(f"  Total Stats: {predicted_total_stats}")

    st.write("Best and Worst Stats at Level 100:")
    st.write(f"  HP: Best: {best_hp}, Worst: {worst_hp}")
    st.write(f"  Attack: Best: {best_attack}, Worst: {worst_attack}")
    st.write(f"  Defense: Best: {best_defense}, Worst: {worst_defense}")
    st.write(f"  Sp. Atk: Best: {best_sp_atk}, Worst: {worst_sp_atk}")
    st.write(f"  Sp. Def: Best: {best_sp_def}, Worst: {worst_sp_def}")
    st.write(f"  Speed: Best: {best_speed}, Worst: {worst_speed}")

    st.write(f"Predicted Stats with Nature, IV, and EV Adjustments:")
    st.write(f"  HP: {nature_adjusted_hp}")
    st.write(f"  Attack: {nature_adjusted_attack}")
    st.write(f"  Defense: {nature_adjusted_defense}")
    st.write(f"  Sp. Atk: {nature_adjusted_sp_atk}")
    st.write(f"  Sp. Def: {nature_adjusted_sp_def}")
    st.write(f"  Speed: {nature_adjusted_speed}")
