import pandas as pd
import os

# --- CONFIGURATION ---
current_script_folder = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(current_script_folder, "previous_data.xlsx")
OUTPUT_FILE = "experiment_data/previous_azimuth.csv"

def convert_grid_to_csv():
    # 1. Read the Excel file
    # assuming the first column (index 0) contains the Targets (-300, -275, etc.)
    try:
        df = pd.read_excel(INPUT_FILE, header=None) 
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # 2. Rename the first column to 'Target' for clarity
    # (We assume the first column holds the target positions)
    first_col_name = df.columns[0]
    df.rename(columns={first_col_name: 'Target'}, inplace=True)

    # 3. "Melt" the data
    # This turns the grid into a long list: [Target, Participant, Response]
    melted_df = df.melt(id_vars=['Target'], var_name='Participant_ID', value_name='Response')

    # 4. Filter out empty responses (if any)
    melted_df.dropna(subset=['Response'], inplace=True)

    # 5. Create the formatted CSV content manually
    # Your main script expects a specific 3-line header block
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # -- The Header Block --
        f.write("Experiment, Previous\n")  # Key for the main script to find
        f.write("Audio, Single Speaker\n")
        f.write("Participant, Pooled\n")
        f.write("Trial_ID, Time, Target, Response\n") # The Data Header

        # -- The Data Rows --
        # We generate a fake Trial_ID (i) and Time (0)
        for i, row in enumerate(melted_df.itertuples(), 1):
            # row.Target is the target position
            # row.Response is the user's guess
            f.write(f"{i}, 0, {row.Target}, {row.Response}\n")

    print(f"Successfully converted {len(melted_df)} data points.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    convert_grid_to_csv()