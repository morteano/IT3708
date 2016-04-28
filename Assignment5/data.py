import openpyxl as px
import numpy as np


class Data:
    def __init__(self):
        self.cost, self.dist = self.readData()

    def readData(self):
        cost = self.readFile('Cost.xlsx')
        dist = self.readFile('Distance.xlsx')
        return cost, dist

    def readFile(self, filename):
        # read data
        W = px.load_workbook(filename, use_iterators = True)
        p = W.get_sheet_by_name(name = 'Sheet1')

        # store data in a list
        a=[]
        for row in p.iter_rows():
            for k in row:
                a.append(k.internal_value)

        # convert list a to a 48x48 matrix
        aa = np.resize(a, [49, 49])
        aa = np.delete(aa, 0, 0)
        aa = np.delete(aa, 0, 1)
        return aa