from test import prepare_data

samples = 100
# 根路径
rootpath = 'leapGestRecog/'
# 训练集图片路径
files, labels, classes = prepare_data(rootpath)


print(len(files),len(labels))
print(len(files[0]),len(labels[0]))
print(files[9][242])
print(labels[9][242])
print(classes)