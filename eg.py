# from test import prepare_data

# samples = 100
# # 根路径
# rootpath = 'leapGestRecog/'
# # 训练集图片路径
# files, labels, classes = prepare_data(rootpath)


# print(len(files),len(labels))
# print(len(files[0]),len(labels[0]))
# print(files[9][242])
# print(labels[9][242])
# print(classes)


import csv

List=[[1,2,3],[2,3,4],[4,3,5]]
with open('List.csv', 'w', newline='') as csvfile:
    writer  = csv.writer(csvfile)
    for row in List:
        writer.writerow(row)
