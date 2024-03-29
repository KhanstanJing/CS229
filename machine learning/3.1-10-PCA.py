'''对scikit-learn内置的鸢尾花数据使用PCA的示例代码
用于将4个特征变量变换为2个主成分

https://blog.csdn.net/u012162613/article/details/42192293
1、函数原型及参数说明
sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)
参数说明：
n_components:
意义：PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
类型：int 或者 string，缺省时默认为None，所有成分被保留。
赋值为int，比如n_components=1，将把原始数据降到一个维度。
赋值为string，比如n_components='mle'，将自动选取特征个数n，使得满足所要求的方差百分比。

copy:
类型：bool，True或者False，缺省时默认为True。
意义：表示是否在运行算法时，将原始训练数据复制一份。若为True，则运行PCA算法后，原始训练数据的值不会有任何改变，因为是在原始数据的副本上进行运算；若为False，则运行PCA算法后，原始训练数据的              值会改，因为是在原始数据上进行降维计算。

whiten:
类型：bool，缺省时默认为False
意义：白化，使得每个特征具有相同的方差。关于“白化”，可参考：Ufldl教程

2、PCA对象的属性
components_ ：返回具有最大方差的成分。
explained_variance_ratio_：返回 所保留的n个成分各自的方差百分比。
n_components_：返回所保留的成分个数n。
mean_：
noise_variance_：

3、PCA对象的方法
fit(X,y=None)
fit()可以说是scikit-learn中通用的方法，每个需要训练的算法都会有fit()方法，它其实就是算法中的“训练”这一步骤。因为PCA是无监督学习算法，此处y自然等于None。
fit(X)表示用数据X来训练PCA模型。
函数返回值：调用fit方法的对象本身。比如pca.fit(X)，表示用X对pca这个对象进行训练。

fit_transform(X)
用X来训练PCA模型，同时返回降维后的数据。
newX=pca.fit_transform(X)，newX就是降维后的数据。

inverse_transform()
将降维后的数据转换成原始数据，X=pca.inverse_transform(newX)

transform(X)
将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。

此外，还有get_covariance()、get_precision()、get_params(deep=True)、score(X, y=None)等方法

https://mp.weixin.qq.com/s/Dv51K8JETakIKe5dPBAPVg
SVD分解

https://zhuanlan.zhihu.com/p/37777074
鸢尾花数据集（Iris）最初由Edgar Anderson 测量得到，而后在著名的统计学家和生物学家R.A Fisher于1936年发表的文章「The use of multiple measurements in taxonomic problems」中被使用
用其作为线性判别分析（Linear Discriminant Analysis）的一个例子，证明分类的统计方法，从此而被众人所知，尤其是在机器学习这个领域。

数据中的两类鸢尾花记录结果是在加拿大加斯帕半岛上，于同一天的同一个时间段，使用相同的测量仪器，在相同的牧场上由同一个人测量出来的
这是一份有着70年历史的数据，虽然老，但是却很经典，详细数据集可以在UCI 数据库（http://archive.ics.uci.edu/ml/datasets/Iris） 中找到。

鸢尾花卉数据集是一类多重变量分析的数据集，它共有4个属性列和一个品种类别列
sepal length（萼片长度）、sepal width（萼片宽度）、petal length（花瓣长度）、petal width （花瓣宽度），单位都是厘米
3个品种类别是Setosa、Versicolour、Virginica，样本数量150个，每类50个。
'''

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
n_components = 2 # 将减少后的维度设置为2（尝试更改不同的n_components，观察输出结果）
model = PCA(n_components=n_components)
model = model.fit(data.data)
print(model.transform(data.data))
