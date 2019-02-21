import sympy as sy

class ID3:
    def __init__(self, oriTrainingSet):
        self.preprocess(oriTrainingSet)
        entropRes = self.__entrop_allfeatures_calculate(oriTrainingSet, 1, 4, 5)
        
        print(entropRes)

    def preprocess(self, oriTrainingSet):
        for dataItem in oriTrainingSet:
            if dataItem[1] == "Sunny":
                dataItem[1] = 0
            elif dataItem[1] == "Overcast":
                dataItem[1] = 1
            elif dataItem[1] == "Rainy":
                dataItem[1] = 2
            else:
                dataItem[1] = 3

            if dataItem[2] == "Hot":
                dataItem[2] = 0
            elif dataItem[2] == "Cool":
                dataItem[2] = 1
            elif dataItem[2] == "Mild":
                dataItem[2] = 2
            else:
                dataItem[2] = 3

            if dataItem[3] == "Normal":
                dataItem[3] = 0
            elif dataItem[3] == "High":
                dataItem[3] = 1
            else:
                dataItem[3] = 2

            if dataItem[4] == "Weak":
                dataItem[4] = 0
            elif dataItem[4] == "Strong":
                dataItem[4] = 1
            else:
                dataItem[4] = 2

            if dataItem[5] == "No":
                dataItem[5] = 0
            elif dataItem[5] == "Yes":
                dataItem[5] = 1
            else:
                dataItem[5] = 2

    def __entrop_allfeatures_calculate(self, dataSet, featureBeginIndex, featureEndIndex, resIndex):
        entropRes = []
        for i in range(featureBeginIndex, featureEndIndex + 1):
            entrop = self.__entrop_feature_calculate(dataSet, i, resIndex)
            entropRes.append(entrop)
        return entropRes

    def __entrop_feature_calculate(self, dataSet, featureIndex, resIndex):
        dataSumCount = dataSet.__len__()
        featureCountMap = {}
        for dataItem in dataSet:
            if "f" + str(dataItem[featureIndex]) in featureCountMap:
                featureCountMap["f" + str(dataItem[featureIndex])]["dataCount"] += 1
                if "r" + str(dataItem[resIndex]) in featureCountMap["f" + str(dataItem[featureIndex])]:
                    featureCountMap["f" + str(dataItem[featureIndex])]["r" + str(dataItem[resIndex])] += 1
                else:
                    featureCountMap["f" + str(dataItem[featureIndex])]["r" + str(dataItem[resIndex])] = 1
            else:
                featureCountMap["f" + str(dataItem[featureIndex])] = {}
                featureCountMap["f" + str(dataItem[featureIndex])]["dataCount"] = 1
                featureCountMap["f" + str(dataItem[featureIndex])]["r" + str(dataItem[resIndex])] = 1
        #print(featureCountMap)
        
        entropRes = 0
        for features in featureCountMap:
            featureSumCount = featureCountMap[features]["dataCount"]
            p = featureSumCount / dataSumCount
            pResult = []
            del featureCountMap[features]["dataCount"]
            for resCount in featureCountMap[features]:
                pResult.append(featureCountMap[features][resCount])
            featurnP = 0
            for res in pResult:
                featurnP = featurnP - (res / featureSumCount * sy.log(res / featureSumCount))
            entropRes = entropRes + p * featurnP
        return entropRes



#       Day	    OutLook	    Temperature	Humidity	Wind	PlayTennis
oriTrainingSet = [
        [1,	    "Sunny",    "Hot",      "High",	    "Weak",     "No"],
        [2,	    "Sunny",    "Hot",      "High",	    "Strong",	"No"],
        [3,	    "Overcast",	"Hot",      "High",	    "Weak",     "Yes"],
        [4,	    "Rainy",    "Mild",	    "High",	    "Weak",	    "Yes"],
        [5,	    "Rainy",    "Cool",	    "Normal",	"Weak",	    "Yes"],
        [6,	    "Rainy",    "Cool",	    "Normal",	"Strong",	"No"],
        [7,	    "Overcast", "Cool",     "Normal",	"Strong",   "Yes"],
        [8,	    "Sunny",    "Mild",	    "High",	    "Weak",	    "No"],
        [9,	    "Sunny",    "Cool",	    "Normal",	"Weak",     "Yes"],
        [10,	"Rainy",    "Mild",	    "Normal",	"Weak",     "Yes"],
        [11,	"Sunny",    "Mild",	    "Normal",	"Strong",	"Yes"],
        [12,	"Overcast",	"Mild",     "High",	    "Strong",	"Yes"],
        [13,	"Overcast",	"Hot",      "Normal",	"Weak",	    "Yes"],
        [14,	"Rainy",    "Mild",	    "High",	    "Strong",	"No"]
    ]

print(1/2)
id3 = ID3(oriTrainingSet)