import os
import sys
import csv
import re

# Get command line arguments
instance_name = sys.argv[1]
objective = sys.argv[2]
pool_copies = int(sys.argv[3])

# Define the file path
file_stub = f"../CR_parallel_manip_output/vary_pool_{instance_name}/{objective}_{pool_copies}/"

# Initialize variables
highest_deviation_benefit = float('-inf')
best_file_data = None

# Iterate through files
for filename in os.listdir(file_stub):
    if filename.startswith("manipvec") and filename.endswith("deviation_benefit.txt"):
        file_path = os.path.join(file_stub, filename)
        
        # Parse file data
        with open(file_path, 'r') as file:
            # lines of the file are of the form tuple, dictionary, float, float, float  
            line = file.readline()
            starting_vector, ending_vector, ending_marginal, starting_marginal, deviation_benefit = re.split(r',(?![^\(\{]*[\)\}])', line)
            deviation_benefit = float(deviation_benefit)
            # Update highest deviation benefit and best file data if necessary
            if deviation_benefit > highest_deviation_benefit:
                highest_deviation_benefit = deviation_benefit
                best_file_data = (starting_vector, ending_vector, starting_marginal, ending_marginal, deviation_benefit)

# Write the best file data to a new CSV file
output_file = f"../CR_parallel_manip_output/vary_pool_{instance_name}/{objective}_{pool_copies}.csv"
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Starting Vector", "Ending Vector", "Starting Marginal", "Ending Marginal", "Deviation Benefit"])
    writer.writerow(best_file_data)
