
'''对scikit-learn内置的鸢尾花数据使用PCA的示例代码
用于将4个特征变量变换为2个主成分
'''

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
n_components = 2 # 将减少后的维度设置为2（尝试更改不同的n_components，观察输出结果）
model = PCA(n_components=n_components)
model = model.fit(data.data)
print(model.transform(data.data))
