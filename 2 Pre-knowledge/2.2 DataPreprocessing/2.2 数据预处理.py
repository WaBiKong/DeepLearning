# 2.2 数据预处理
import os
import pandas as pd
import torch

os.makedirs(os.path.join('data'), exist_ok=True)  # 在文件目录创建data文件夹
data_file = os.path.join('data', 'house_tiny.csv')  # 创建文件
with open(data_file, 'w') as f:  # 往文件中写数据
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 第1行的值
    f.write('2,NA,103000\n')  # 第2行的值
    f.write('4,NA,178100\n')  # 第3行的值
    f.write('NA,NA,140000\n')  # 第4行的值
data = pd.read_csv(data_file)  # 可以看到原始表格中的空值NA被识别成了NaN
print('原始数据:\n',data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]  # 切片获取原始数据
inputs = inputs.fillna(inputs.mean())  # 用均值填充NAN
print(inputs)
print(outputs)
# 利用pandas中的get_dummies函数来处理离散值或者类别值
# [对于inputs中的类别值或离散值，我们将"NaN"视为一个类别]由于"Alley"列只接受两种类型的类别值"Pave"和"NaN"
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 转换为张量
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x)
print(y)
