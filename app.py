import os
import pandas as pd

def convert_grouped_data_to_csv():
    excel_file = "Grouped_Data.xlsx"
    csv_file = "Grouped_Data.csv"
    
    # Check if the CSV already exists to avoid unnecessary conversion.
    if not os.path.exists(csv_file):
        df = pd.read_excel(excel_file)
        df.to_csv(csv_file, index=False)
        print(f"Converted {excel_file} to {csv_file}.")
    else:
        print(f"{csv_file} already exists.")

if __name__ == "__main__":
    convert_grouped_data_to_csv()
