ds_path = "/Users/zw/Downloads/dataset/"

import os


images = []
labels = []
for num in range(int(len(os.listdir(ds_path)) / 2)):
    images.append(num)
    labels.append(num)

sorted(images)
sorted(labels)

images = list(map(lambda x: "dataset/" + str(x) + "_in.png", images))
labels = list(map(lambda x: "dataset/" + str(x) + "_gt.png", labels))

print(images)
print(labels)





