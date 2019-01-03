import sympy

#Single variable linear regression
#Using batch gradient descent(BGD)
class Svlr:
    def __init__(self, trainingSet, thresholdVariance = 20, maxIterationCount = 200):
        self.__J = 0
        self.__k1_s = sympy.symbols("k1")
        self.__k2_s = sympy.symbols("k2")
        self.__k1_v = 0
        self.__k2_v = 0
        self.__h = self.__hypothesis(self.__k1_v, self.__k2_v, sympy.symbols("x"))
        self.__trainingSet = trainingSet
        self.__thresholdVariance = thresholdVariance
        self.__maxIterationCount = maxIterationCount
        self.__alpha = 0.00005
        self.__k1_factor = 1
        self.__k2_factor = 500
        self.lastestResult = {}
        
    def __hypothesis(self, k1, k2, x):
        return k1 * x + k2

    def __BGD(self, currentIterationCount):
        variance = self.__J.subs(((self.__k1_s, self.__k1_v),(self.__k2_s, self.__k2_v)))
        if variance > self.__thresholdVariance and currentIterationCount < self.__maxIterationCount :
            new___k1_v = self.__k1_v - self.__alpha * self.__J_k1_diff.subs(((self.__k1_s, self.__k1_v),(self.__k2_s, self.__k2_v))) * self.__k1_factor
            new___k2_v = self.__k2_v - self.__alpha * self.__J_k2_diff.subs(((self.__k1_s, self.__k1_v),(self.__k2_s, self.__k2_v))) * self.__k2_factor
            self.__k1_v = new___k1_v
            self.__k2_v = new___k2_v
            return self.__BGD(currentIterationCount + 1)
        else:
            self.lastestResult = {"k1" : self.__k1_v, "k2" : self.__k2_v, "variance" : variance, "iterationCount" : currentIterationCount, "h(x)" : self.__hypothesis(self.__k1_v, self.__k2_v, sympy.symbols("x"))}
            return self.lastestResult["h(x)"]

    def train(self):
        for item in self.__trainingSet:
            self.__J = self.__J + (self.__hypothesis(self.__k1_s, self.__k2_s, item[0]) - item[1]) ** 2
        self.__J = self.__J / len(self.__trainingSet)
        self.__J_k1_diff = sympy.diff(self.__J, self.__k1_s)
        self.__J_k2_diff = sympy.diff(self.__J, self.__k2_s)
        self.__h = self.__BGD(0)

    def predict(self, x):
        return self.__h.subs(sympy.symbols("x"), x)