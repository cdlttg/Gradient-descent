import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from mpl_toolkits.mplot3d import Axes3D
import warnings

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

"""
二维原始图像
1、构建一个函数。
2、随机生成X1,X2点，根据X1,X2点生成Y点。
3、画出图像。
"""
np.set_printoptions(precision = 2)

def f2(x, y):
    return -math.log(1-x-y)-math.log(x)-math.log(y)
    #return x1 ** 2 + 2 * x2 ** 2 - 4 * x1 - 2 * x1 * x2

## 偏函数
def hx1(x, y):
    return 1/(1-x-y)-1/x
def hx2(x, y):
    return 1/(1-x-y)-1/y

X1 = np.arange(0.01,0.99,0.01)
X2 = np.arange(0.01,0.99,0.01)
x1 = []
x2 = []
for i in X1:
    for j in X2:
        if np.around(i,2)+np.around(j,2) <=0.99:
            x1.append(np.around(i,2))
            x2.append(np.around(j,2))
#X1, X2 = np.meshgrid(x1, x2) # 生成xv、yv，将X1、X2变成n*m的矩阵，方便后面绘图
Y = np.array(list(map(lambda t : f2(t[0],t[1]),zip(x1,x2))))
#Y.shape = x1.shape # 1600的Y图还原成原来的（40,40）

fig = plt.figure(facecolor='w', figsize=(20, 18))
ax = Axes3D(fig)
#ax.plot_surface(x1, x2, Y, rstride=1, cstride=1, cmap=plt.cm.jet)
surf = ax.plot_trisurf(x1, x2, Y, linewidth=0, antialiased=False, cmap=plt.cm.jet, alpha=0.3)
x1 = 0.2
x2 = 0.6
alpha = 0.005
# 保存梯度下降经过的点
GD_X1 = [x1]
GD_X2 = [x2]
GD_Y = [f2(x1, x2)]
# 定义y的变化量和迭代次数
y_change = f2(x1, x2)
iter_num = 0

while (y_change > 1e-10 and iter_num < 100):
    tmp_x1 = x1 - alpha * hx1(x1, x2)
    tmp_x2 = x2 - alpha * hx2(x1, x2)
    tmp_y = f2(tmp_x1, tmp_x2)

    f_change = np.absolute(tmp_y - f2(x1, x2))
    x1 = tmp_x1
    x2 = tmp_x2
    GD_X1.append(x1)
    GD_X2.append(x2)
    GD_Y.append(tmp_y)
    iter_num += 1
    print(x1,x2)
print(u"最终结果为:(%.5f, %.5f, %.5f)" % (x1, x2, f2(x1, x2)))
print(u"迭代过程中X的取值，迭代次数:%d" % iter_num)
print(GD_X1)


ax.plot(GD_X1, GD_X2, GD_Y, 'ko-', color='r',linewidth=5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_title(u'函数;\n学习率:%.3f; 最终解:(%.3f, %.3f, %.3f);迭代次数:%d' % (alpha, x1, x2, f2(x1, x2), iter_num))
plt.show()
