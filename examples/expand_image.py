import cv2

dirpath = '../data/result/trigger/badnet_square_cifar10(3,3).png'
img = cv2.imread(dirpath)  # 读取数据
a = cv2.copyMakeBorder(img, 0, 29, 0, 29, cv2.BORDER_CONSTANT, value=[0, 0, 0])
cv2.imwrite('cifar-badnet', a)