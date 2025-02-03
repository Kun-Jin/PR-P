#https://gist.github.com/dhadka/ba6d3c570400bdb411c3 有该代码详细说明  (版本较老)
#https://platypus.readthedocs.io/en/latest/experimenter.html#basic-use  （版本较新，本代码根据此进行的改动）
from platypus import *
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Times New Roman'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import pandas as pd
#设置字体为楷体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
from MLP1 import picriteria

from mpl_toolkits.mplot3d import Axes3D

class selfQusetion(Problem):

    def __init__(self):
        super().__init__(2, 2)
        self.types[:] = [Real(0, 10), Real(0, 10)]

    def evaluate(self, solution):
        w1 = solution.variables[0]  # 下界的权重
        w2 = solution.variables[1]  # 上界的权重

        PICP1, PINAW1 = picriteria(w1, w2)
        solution.objectives[:] = [-PICP1, PINAW1]

algorithm = NSGAII(selfQusetion())
algorithm.run(2000)

for solution in algorithm.result:
    print(solution.objectives)

plt.scatter([s.objectives[0] for s in algorithm.result],
            [s.objectives[1] for s in algorithm.result])
# plt.xlim([-1, 1.1])
# plt.ylim([0, 1.1])
plt.xlabel("-PICP")
plt.ylabel("PINAW")

plt.savefig('NSGA2.pdf')
plt.show()
