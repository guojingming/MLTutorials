import numpy as np

trainingSet = np.mat(
    [[100,90,2],
    [120,100,3],
    [110,100,7]]
)
trainingSet2 = np.mat(
    [[100,90,2],
    [120,100,3],
    [110,100,7]]
)

print(trainingSet * trainingSet2)