import pisa
from pisa.utils.fileio import from_file
import pandas as pd
import numpy as np

result = from_file("./results.json")

num_scan_points = len(result["results"])
print("theta23, deltma31, chi2")
for i in range(num_scan_points) :
    print( "%s, %s, %s" % (result["steps"]["theta23"][i].m, result["steps"]["deltam31"][i].m, result["results"][i]["metric_val"]) )
