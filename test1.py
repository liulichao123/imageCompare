# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

SIZE = (100, 100)
def aHash(img):
    # 均值哈希算法
    # 缩放为size*size的像素格子进行比较
    # 如果使用原大小进行比较，不讲两个图统一大小，会无法比较，或者两个图本身就宽高相等也可以
    SIZE = (int(img.shape[0]/10), int(img.shape[1]/10))
    print('缩放为',SIZE,'进行对比')
    # img = cv2.resize(img, (size, size))
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(SIZE[0]):
        for j in range(SIZE[1]):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / (SIZE[0]*SIZE[1])
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(SIZE[0]):
        for j in range(SIZE[1]):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    # plt.subplot(121)
    # plt.imshow(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    # plt.subplot(122)
    # plt.imshow(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    # plt.show()
    return hash_str

def getImageByUrl(url):
    # 根据图片url 获取图片对象
    html = requests.get(url, verify=False)
    image = Image.open(BytesIO(html.content))
    return image

def cmpHash(hash1, hash2):
    # Hash值对比
    # 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
    # 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
    # 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        print(hash1, '\n', hash2)
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def runAllImageSimilaryFun(para1, para2):
    # 均值、差值、感知哈希算法三种算法值越小，则越相似,相同图片值为0
    # 三直方图算法和单通道的直方图 0-1之间，值越大，越相似。 相同图片为1

    # t1,t2   14;19;10;  0.70;0.75
    # t1,t3   39 33 18   0.58 0.49
    # s1,s2  7 23 11     0.83 0.86  挺相似的图片
    # c1,c2  11 29 17    0.30 0.31

    if para1.startswith("http"):
        # 根据链接下载图片，并转换为opencv格式
        img1 = getImageByUrl(para1)
        img1 = cv2.cvtColor(np.asarray(img1), cv2.COLOR_RGB2BGR)

        img2 = getImageByUrl(para2)
        img2 = cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)
    else:
        # 通过imread方法直接读取物理路径
        img1 = cv2.imread(para1)
        img2 = cv2.imread(para2)

    hash1 = aHash(img1)
    hash2 = aHash(img2)
    n1 = cmpHash(hash1, hash2)
    print('均值哈希算法相似度aHash：', n1)

    img1 = cv2.resize(img1, SIZE)
    img2 = cv2.resize(img2, SIZE)

    plt.subplot(121)
    plt.imshow(Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)))
    plt.subplot(122)
    plt.imshow(Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)))
    plt.show()


if __name__ == "__main__":
    p1 = "A.jpg"
    p2 = "B.jpg"
    runAllImageSimilaryFun(p1, p2)

    # 均值哈希算法相似度aHash： 25  //越小越相似
    # 差值哈希算法相似度dHash： 19  //越小越相似
    # 感知哈希算法相似度pHash： 14  //越小越相似
    # 三直方图算法相似度： [0.70234877] //越大越相似
    # 单通道的直方图 [0.6933419]   //越大越相似
    # 25 19 14 0.70 0.69
    # 0.61 0.70 0.78 0.70 0.69

    # 两张相同的图结果如下：
    # 均值哈希算法相似度aHash： 0
    # 差值哈希算法相似度dHash： 0
    # 感知哈希算法相似度pHash： 0
    # 三直方图算法相似度： 1.0
    # 单通道的直方图 1.0
