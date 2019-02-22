class MultiTreeRoot:
    def __init__(self):
        self.isRoot = True
        self.isLeaf = True
        self.children = {}
        self.childrenCount = 0
        self.value = {}


def appendChild(node, childName, childValue):
    node.children[childName] = {}
    node.isLeaf = False
    child = node.children[childName]
    child.isRoot = False
    child.isLeaf = True
    child.children = {}
    child.childrenCount = 0
    child.value = childValue

def getChild(node, childName):
    return node.children[childName]

def printTree(node):
    __printTree(node, 0, "")

def __printTree(node, currentDepth, printStr):
    return ""