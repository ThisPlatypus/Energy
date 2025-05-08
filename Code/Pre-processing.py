import pandas as pd

# Load the CSV file
file_path = '/home/chiara/Energy/Data/Ausgrid Load Data.csv'  # Replace with the path to your CSV file
save_path = '/home/chiara/Energy/Data/Train_2m_72.csv'  # Replace with the path to save the new CSV file

data = pd.read_csv(file_path, header=0, sep=';')
# Transpose the data
transposed_data = data.T
result = []

for index, row in transposed_data.iterrows():
    row_id = index  # Use the index as a unique ID
    values = row.values  # Use all values in the row
    for i in range(0, len(values) - 144 + 1, 72):  # Sliding window of 72 elements for x and y
        x = values[i:i+72]
        y = values[i+72:i+144]
        if len(x) == 72 and len(y) == 72:
            result.append([row_id] + list(x) + list(y))

columns = ['ID'] + [f'x{i}' for i in range(72)] + [f'y{i}' for i in range(72)]
pd.DataFrame(result, columns=columns).to_csv(save_path, index=False)

