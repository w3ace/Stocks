import os
import pandas as pd

def combine_info_to_csv(input_dir, output_file):
    # Initialize an empty list to store data from all .info files
    data = []

    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".info"):
            filepath = os.path.join(input_dir, filename)
            # Read data from each .info file
            with open(filepath, 'r') as file:
                lines = file.readlines()
                # Create a dictionary for storing data from each file
                info_dict = {}
                for line in lines:
                    # Check if the line can be split into a key-value pair
                    if ': ' in line:
                        key, value = line.strip().split(': ', 1)
                        info_dict[key] = value
                
                # Check if exchange is PNK (OTC Markets Group) or market cap is less than 1B
                if (info_dict.get('exchange') != 'PNK' and 
                    info_dict.get('marketCap') is not None and 
                    float(info_dict.get('marketCap', 0).replace(',', '')) >= 1e9):  # Ensure market cap is >= 1 billion
                    # Append the dictionary to the data list
                    data.append(info_dict)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Combined stock info saved to {output_file}")

def main():
    # Input directory containing .info files
    input_directory = 'Datasets'

    # Output CSV file
    output_csv = 'combined_stock_info_filtered.csv'

    # Call function to combine .info files to CSV
    combine_info_to_csv(input_directory, output_csv)

if __name__ == "__main__":
    main()
