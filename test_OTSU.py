import cv2


def processing(img_path):
    # read file
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow("HELLO", cv2.WINDOW_NORMAL)
    cv2.imshow("HELLO", img)
    cv2.namedWindow("HELLO_2", cv2.WINDOW_NORMAL)
    _, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow("HELLO_2", otsu_img)

    cv2.waitKey()
    cv2.destroyAllWindows()


processing("/Users/zw/Downloads/img_processing (20).png")
