import csv


images = []

with open('data/driving_log.csv') as f:
    print("open")
    reader = csv.reader(f)
    for line in reader:
        print(line[0])
