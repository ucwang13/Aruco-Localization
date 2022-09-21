from numpy import load

data = load('camCalibration.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])
