import sys

# 1. Open the CSV file in text mode
max_val = 0
min_val = 9999
# for num in range(1,10000):
for num in {1}:
    with open("/workspace/ckarfa/Priority-Cuts/Priority-Cuts-Filter/data/adder/selective_cuts/cuts_"+ str(num) +".csv", "r") as file:
        # 2. Initialize a variable to store the total count
        total_values = 0

        # 3. Iterate over each line (row) in the file
        for line in file:
            # 4. Split the line into values based on commas
            values = line.strip().split(",")
            # 5. Count the number of values and add it to the total
            if len(values) > 10:
                total_values += 10
            else:
                total_values += len(values)-1


    # 6. Print the total count
    # if total_values > 5000:
    print("Total number of values in all rows:", total_values)
    #     print(num)
        # sys.exit()

    max_val = max(max_val, total_values)
    min_val = min(min_val, total_values)

print(max_val)
print(min_val)