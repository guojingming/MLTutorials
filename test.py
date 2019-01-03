from Regression import SingleVarLR

trainingSet = (
    (40, 50),
    (44, 59),
    (50, 61),
    (60, 68),
    (60, 64),
    (70, 72),
    (80, 83),
    (90, 90),
    (110, 93),
    (106, 100)
)



SingleVarLR.Svlr sv(trainingSet);
sv.train(trainingSet)
print(sv.predict(180))
