import pandas as pd

# Load the Excel file
input_file = 'VAT DEC 2024.xlsx'

try:
    # Read the Excel file
    excel_file = pd.ExcelFile(input_file)

    # Iterate over each sheet
    for sheet_name in excel_file.sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        
        # Define the output file name based on the sheet name
        output_file = f'{sheet_name}.xlsx'
        
        # Save the DataFrame to a new Excel file
        df.to_excel(output_file, index=False)
        print(f'Saved: {output_file}')

except FileNotFoundError:
    print("The specified Excel file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")