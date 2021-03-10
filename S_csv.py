import csv
import pandas as pd

data = [] #Buffer list
with open("kpi_0304.csv", "r") as input_file:
    reader = csv.reader(input_file)
    result = list(reader)
    for row in result:
        if row[-1] == 'ServiceTest1':
            print(row)
            data.append(row)

with open("0304-ST1.csv", "w+") as to_file:
    writer = csv.writer(to_file, delimiter=",")
    for new_row in data:
        writer.writerow(new_row)