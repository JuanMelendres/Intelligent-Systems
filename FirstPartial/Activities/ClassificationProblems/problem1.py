import matplotlib.pyplot as plt
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
x = iris.data[:, :2]  # we only take the first two features.
y = iris.target

print(x)
print(y)