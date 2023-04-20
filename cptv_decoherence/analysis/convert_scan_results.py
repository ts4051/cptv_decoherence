import pisa
from pisa.utils.fileio import from_file
import numpy as np
import os, glob


#
# Parse inputs
#

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-id", "--input-dir", type=str, required=True, help="Input directory")
args = parser.parse_args()


#
# Convert results
#

# Find files
input_file_paths = sorted(glob.glob( os.path.join(args.input_dir, "scan_??????.json") ))

# Loop over files
for i_file, input_file_path in enumerate(input_file_paths) :

    # Load file
    try:
        result = from_file(input_file_path)

        if i_file == 0 :
            param_names = []
        param_values = []

        # Extract scan point
        for (param_name, param_val) in result["scan_point"] :
            if i_file == 0 :
                param_names.append(param_name)
            param_values.append(param_val.m)

        # Extract metric
        metric_name = result["best_fit"]["metric"][0]
        metric_val = result["best_fit"]["metric_val"]

        # Dump header
        if i_file == 0 :
            print( ", ".join(param_names+[metric_name]) )

        # Dump scan point
        print( ", ".join( "%0.6g"%x for x in (param_values+[metric_val]) ) )

    # File load issues
    except Exception as e :
        print("Failed to load %s : %s" % (input_file_path, str(e)))
        
