# import os
# from pathlib import Path
# import shutil
#
#
# def rename(count: int, path: str):
#     path = Path(path)
#     for file_name in os.listdir(path):
#         num = int(file_name.split('_')[0]) + count
#         new_file_name = str(num) + '_' + file_name.split('_')[1]
#         os.rename(path / file_name, path / new_file_name)
#         shutil.copy(path / new_file_name, "/Users/zw/Downloads/dd/" + new_file_name)
#
#
# count = 0
#
# for year in range(2009, 2019):
#     print(year)
#     src_path = "/Users/zw/Downloads/DIBCO/" + str(year)
#     h_path = src_path + "/handwritten"
#     p_path = src_path + "/printed"
#     if os.path.exists(h_path):
#         rename(count, h_path)
#         count += len(os.listdir(h_path)) / 2
#     if os.path.exists(p_path):
#         rename(count, p_path)
#         count += len(os.listdir(p_path)) / 2
#
#
#


import os


dd = "/Users/zw/Downloads/dataset"
for file_name in os.listdir(dd):
    new_name = ''.join(file_name.split('.0'))
    os.rename(dd + '/' + file_name, dd + '/' + new_name)