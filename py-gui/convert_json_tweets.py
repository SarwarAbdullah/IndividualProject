import pandas as pd
import json

def remove_non_unicode_lines(input_file, output_file):
    # Read the input file and filter out non-Unicode lines
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        unicode_lines = [line for line in lines if all(ord(char) < 128 for char in line)]

    # Write the filtered lines to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(unicode_lines)

    print(f"Non-Unicode lines removed. Filtered content saved to {output_file}")

remove_non_unicode_lines('0.json', '0_filtered.json')

# Load the Json file
with open('0_filtered.json') as json_file:
    #each line is a json object
    # Load the JSON data using the json module, skipping over lines that are not valid JSON
    json_data = [json.loads(line) for line in json_file]


    #json_data = [json.loads(line) for line in json_file]

# Convert the nested JSON to a pandas DataFrame
df = pd.json_normalize(json_data)

# Save the DataFrame to a CSV file (replace 'output.csv' with your desired output file path)
df.to_csv('output.csv', index=False)

print(df.head())

print("Nested JSON data successfully converted to CSV and saved as 'output.csv'")

