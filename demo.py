import numpy as np
import Regression.SingleVarLR as SingleVarLR

#Demo1: Single variable linear regression using BGD
def demo1():
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
    sv = SingleVarLR.Svlr(trainingSet)
    sv.train()
    print(sv.predict(180))
    print(sv.lastestResult)

if __name__ == "__main__":
    demo1()