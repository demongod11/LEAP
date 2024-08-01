import csv

input_file_path = "/workspace/ckarfa/Priority-Cuts/Priority-Cuts-Filter/data/" + "div" + "/all_cuts.csv"
output_file_path = "/workspace/ckarfa/Priority-Cuts/Priority-Cuts-Filter/data/" + "div" + "/filtered_cuts.csv"

# Count elements in each line of the output file
number = 0

with open(output_file_path, 'r') as output_file:
    csv_reader = csv.reader(output_file)
    for row in csv_reader:
        num_elements = len(row) - 1  # Exclude the first element (row[0])
        number = num_elements+number

print(number)