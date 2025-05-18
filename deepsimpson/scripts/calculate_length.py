import numpy as np
import pandas as pd
import os

output = "deepsimpson/output/features/simpsons_ed_es.csv"
filelist = pd.read_csv("/home/eda/Desktop/dynamic/EchoNet-Dynamic/FileList.csv")

# Add "avi"
filelist["FileName"] = filelist["FileName"].astype(str) + ".avi"
if os.path.exists(output):
    print(f"File already exists at: {output}")
else:
    # Paths to input/output CSV files
    simpsons_train = f"deepsimpson/output/features/simpsons_train.csv"
    simpsons_test  = f"deepsimpson/output/features/simpsons_test.csv"
    simpsons_val   = f"deepsimpson/output/features/simpsons_val.csv"

    # Load CSVs
    df_train = pd.read_csv(simpsons_train)
    df_val = pd.read_csv(simpsons_val)
    df_test = pd.read_csv(simpsons_test)

    # Add split column
    df_train["Split"] = "train"
    df_val["Split"] = "val"
    df_test["Split"] = "test"

    # Combine them
    df_combined = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Save to one CSV
    df_combined.to_csv(output, index=False)

    print(f"Combined CSV saved to: {output}")

# Read combined file
df_output = pd.read_csv(output)

# Add length column
df_output["Length"] = np.sqrt((df_output["End_X"] - df_output["Start_X"]) ** 2 + (df_output["End_Y"] - df_output["Start_Y"]) ** 2)

# Add EF values
df_output["EF"] = df_output["Filename"].map(filelist.set_index("FileName")["EF"]) 

# Save as csv
df_output.to_csv(output, index=False)



