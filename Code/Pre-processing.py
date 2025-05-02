import pandas as pd

# Load the CSV file
file_path = '/Users/cettina/Documents/FI_PRJ/Energy/Energy/Data/Ausgrid Load Data.csv'  # Replace with the path to your CSV file
save_path = '/Users/cettina/Documents/FI_PRJ/Energy/Energy/Data/SET_1790.csv'  # Replace with the path to save the new CSV file

data = pd.read_csv(file_path, header=0, sep=';')
# Transpose the data
transposed_data = data.T
new_dat = transposed_data.iloc[:, 0:7160]
pd.DataFrame(new_dat.values.reshape(-1, 1790)).to_csv(save_path, index=False)

# for reading
#pd.read_csv(save_path, header=0, sep=',')