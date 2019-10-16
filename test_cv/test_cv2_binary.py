import shutil

import cv2


def load_image(img_path):
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256, 256))

    # image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    print(image)
    cv2.namedWindow("Hello", cv2.WINDOW_NORMAL)
    cv2.imshow("Hello", image)
    cv2.waitKey()


# src = "/Users/zw/Downloads/dataset/"
# dst = "/Users/zw/Documents/py36/Deep_binary/test_cv/test_images/"
#
# for num in range(123, 133):
#     file_name = str(num) + "_in.png"
#     print(file_name)
#     shutil.copy(src + file_name, dst + file_name)


# test opencv binary
load_image("test_images/123_in.png")

cv2.destroyAllWindows()

