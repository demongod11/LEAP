# 1. Set the desired number of values to keep (n)
n = 5000  # Adjust this value as needed
total = 0
# 2. Open the input and output CSV files
with open("/workspace/ckarfa/Priority-Cuts/Priority-Cuts-Filter/data/adder/selective_cuts/cuts_5478.csv", "r") as infile, open("/workspace/ckarfa/Priority-Cuts/Priority-Cuts-Filter/data/adder/cuts_5478.csv", "w") as outfile:
    # 3. Iterate over each line (row) in the input file
    for line in infile:
        # 4. Split the line into values based on commas
        values = line.strip().split(",")

        # 5. Trim the values if the number of values is greater than n
        # if len(values) > n:
        #     values = values[:n]
        if total > n:
            values = values[:2]
        total += len(values)
        # 6. Join the trimmed values back into a comma-separated string
        trimmed_line = ",".join(values)

        # 7. Write the trimmed line to the output file
        outfile.write(trimmed_line + "\n")  # Add newline character
print(total)