import sys
import os

input_file = sys.argv[1]

with open(input_file, 'r') as file:
    for line in file:
        line = line.strip()
        fields = line.split(r"'")
        url = fields[1]
        download_file = fields[3]
        
        if os.path.exists(f"../../data/cell/PDF/{download_file}"):
            continue
        else:
            print(line)