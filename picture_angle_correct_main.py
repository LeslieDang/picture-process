# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/10/10 9:02

# opencv version:4.1.1.26
# pyhton version:3.7

"""
背景：
    易酒批交易平台上的图片是有业务人员上传的，可能存在图幅占比不统一，图片歪曲的问题。
    现做一个图片歪曲修正的算法——基于轮廓的最小区域矩形的倾斜角度的



伪代码：
    1、获取图像特征
        图像预处理
            读取图像
            进行灰度处理
            二值化为黑白图像
        获取图像大小、位置
            在黑白图像中获取物体的外接矩形（用于确定图像中物体的位置及大小）
        获取图像轮廓
            在黑白图像中获取物体的轮廓
        根据轮廓获取图像倾斜角度


    2、根据特征进行调整图像
        调整图像的倾斜角度
            筛选出最大的三个轮廓，获取其最小外切矩形的倾斜度
            采用均值计算整体的倾斜角度
            根据倾斜角度，进行仿射变换，修正图像倾斜度
        调整图像大小、位置
            调整倾斜度后的图像，获取其外接矩形
            根据外接矩形及图像的尺寸信息，调整图像大小、留白

    3、过程功能开发
        显示图片
        根据原图创建白底图片，用于绘制轮廓图
        将png格式图像修改为jpg格式（还原图像格式，由于opencv保存图像的alpha通道需要使用png格式，
            但是原图为jpg，因此输出之后暴力修改后缀名，并未影响图像位深度、显示效果）

主要技术细节问题：
    轮廓是如何计算得到的：https://www.cnblogs.com/mrfri/p/8550328.html
    倾斜角度问题：https://blog.csdn.net/qq_24237837/article/details/77850496
    黑白化的阈值设定问题：https://www.cnblogs.com/jyxbk/p/9638541.html
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1、获取图像特征
def thresh_process(picture_path):
    """
    对图像进行二值化处理（转为黑白图）
        读取图像
        进行灰度处理
        二值化为黑白图像
    :param picture_path: 原图路径
    :return: 返回黑白图像tresh
    """
    img = cv2.imread(picture_path)
    picture_path_file = "/".join(picture_path.split("/")[:-1])

    # 将图片转化为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("%s"%(picture_path_file+"/"+"img_gray.jpg"), img_gray)

    # 将图像二值化为黑白图片，1表示大于阈值的变为0，否则变为最大值255
    ret, thresh = cv2.threshold(img_gray, 127, 255, 1)  # (输入的灰度图像，阈值，最大值，划分时使用的算法)
    cv2.imwrite("%s"%(picture_path_file+"/"+"binary.jpg"), thresh)

    return thresh

def contours_acquire(picture_path):
    """
    寻找图像中物体的轮廓
    :param picture_path: 图片完整路径
    :return: list 返回图像的轮廓集合
    """
    thresh = thresh_process(picture_path)
    picture_path_file = "/".join(picture_path.split("/")[:-1])

    # image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # 旧版本返回三个参数，新版本返回2个
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours: 轮廓的点集  hierarchy: 各层轮廓的索引

    # 画出轮廓并保存图像
    img_arr_blank = img_copy_blank(picture_path)
    cv2.drawContours(img_arr_blank, contours, -1, (0, 0, 255), 2)
    cv2.imwrite("%s"%(picture_path_file+"/"+"contours.jpg"), img_arr_blank)
    show("%s"%(picture_path_file+"/"+"contours.jpg"))

    return contours

def contours_filter(contours):
    """
    根据轮廓计算面积，筛选面积最大的三个轮廓
    :param contours:list 轮廓集合
    :return:list 返回三个轮廓值（按轮廓面积从大到小）
    """
    area_list = {}
    index = 0
    for c in contours:
        area = cv2.contourArea(c)
        area_list[index] = area
        index += 1
    area_list_sorted = sorted(area_list.items(), key=lambda x: x[1], reverse=True)
    area_list_sorted = area_list_sorted[:3]

    contours_top3 = []  # 筛选得到的最大的三个轮廓的点集
    for i in area_list_sorted:
        i = i[0]
        contours_top3.append(contours[i])

    return contours_top3

def calculate_angle(contours_top3):
    """
    获取最大的三个轮廓的倾斜角度，并调整角度值
    :param contours_top3:list 获取到的最大的三个轮廓
    :return:list 调整后的角度值
    """
    # 1、计算倾斜角度
    angle_list = []
    for cnt in contours_top3:
        # 旋转角度
        theta = cv2.minAreaRect(cnt)[2]
        angle_list.append(theta)

    # 2、调整计算角度值
    def angle_process(angle_list):
        # 假设：我们认为，同一张图中的倾斜角度应该类似，如果出现-88与-2类似两个极端的情况时，我们认为二者时成“八”字形分布的。
        #       另外倾斜角度不会高于45度。

        # 调整规则如下：
        """
            （-45,0]   之间，则为左倾：x => x
            （-90，-45]之间，则为右倾：x => x+90

            最终得到的角度：
                为负的表示左倾
                为正的表示右倾
        """
        angle_processed = []
        for i in angle_list:
            if float(i) < -45.:
                angle_processed.append(float(i) + 90)
            else:
                angle_processed.append(float(i))

        return angle_processed

    angle_list = angle_process(angle_list)

    return angle_list

def minAreaRect(contours, picture_path):
    """
    根据轮廓点集，获取轮廓的最小外切矩形。用于展示使用。
    :param contours:list 轮廓点集
    :return:
    """
    # 1、画出轮廓图像
    img_arr_blank = img_copy_blank(picture_path)
    cv2.drawContours(img_arr_blank, contours, -1, (0, 0, 255), 2)

    # 2、在轮廓图上画出最小外切矩形
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)  # 最小外接矩形
        box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
        cv2.drawContours(img_arr_blank, [box], 0, (255, 0, 0), 2)  # 将最小外切矩形画在轮廓图上

    # 3、展示与保存
    cv2.imshow("minAreaRect",img_arr_blank)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    picture_path_file = "/".join(picture_path.split("/")[:-1])
    cv2.imwrite("%s"%(picture_path_file+"/"+"minAreaRect.jpg"), img_arr_blank)

def picture_location(thresh, picture_path):
    """
    根据黑白图，获取图像中物体的位置、大小,展示与保存
    :param thresh: 黑白图
    :param picture_path: 原始图像路径
    :return: None
    """
    x, y, w, h = cv2.boundingRect(thresh)  # 外接矩形  元组（x, y, w, h ) 矩形左上点坐标，w, h 是矩阵的宽、高

    img = cv2.imread(picture_path)
    img_copy = img.copy()
    img_boundingRect = cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("img_boundingRect", img_boundingRect)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    picture_path_file = "/".join(picture_path.split("/")[:-1])
    cv2.imwrite("%s" % (picture_path_file + "/" + "img_boundingRect.jpg"), img_boundingRect)

    return x, y, w, h



# 2、根据特征进行调整图像
def angle_correct(angle_list, picture_path):
    """
    根据倾斜角度，对图像做放射变换，调正图像。倾斜度大于0.1度的才调整
    :param angle_list: list 倾斜角度
    :param picture_path: 原始图像路径
    :return: None
    """
    angle = sum(angle_list) / len(angle_list)

    if abs(angle) > 0.1:
        img = cv2.imread(picture_path, cv2.IMREAD_UNCHANGED) # cv2.IMREAD_UNCHANGED： 用于读入图像的alpha通道，以保证位深度不变
        h, w = img.shape[:2]
        center  = (w//2, h//2)
        Mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        # center : 源图像旋转中心
        # angle  ：旋转角度，正值表示逆时针旋转
        # scale  ：缩放系数

        picture_rotated = cv2.warpAffine(img, Mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        """
        img       : 输入变换前图像
        Mat      : 变换矩阵，用另一个函数getAffineTransform()计算
        (w, h) : 设置输出图像大小
        flags    : 设置插值方式，默认方式为线性插值(另一种WARP_FILL_OUTLIERS)
        参数int borderMode=BORDER_CONSTANT    ：边界像素模式，默认值BORDER_CONSTANT
        参数const Scalar& borderValue=Scalar(): 在恒定边界情况下取的值，默认值为Scalar（），即0
        """
        # 保存旋转后的图片
        picture_path_file = "/".join(picture_path.split("/")[:-1])
        picture_name = picture_path.split(".")[-2].split("/")[-1]

        cv2.imwrite("%s" % (picture_path_file + "/" + picture_name + "_rotated.png"),
                    picture_rotated, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

        # 还原图像格式为jpg（由于图像需要在png格式下才能保存alpha通道信息，又为了保证格式被还原，故多了这一步
        import os
        filename = picture_path_file + "/" + picture_name + "_rotated.png"
        newname = picture_path_file + "/" + picture_name + "_rotated.jpg"
        os.rename(filename, newname)



# 3、过程功能开发
def img_copy_blank(picture_path):
    """根据jpg原图创建白底图片，用于绘制没有底图的轮廓图"""
    import numpy as np
    import cv2

    # 1、读取底图
    img = cv2.imread(picture_path)
    # 2、获取底图形状参数
    shape = img.shape
    # 3、根据底图形状参数，创建白底图numpy格式
    img_arr = np.ones((shape[0], shape[1], 3), dtype=int)
    img_arr_blank = img_arr * 255
    # 4、返回numpy图像数据
    return img_arr_blank

def show(picture_path):
    """显示图片"""
    img = cv2.imread(picture_path)
    # img = img[:,:,(2,1,0)]   # opencv中的通道顺序是BGR，与Python的RGB刚好相反。因此此处需要调整顺序。
    # img = img[:,:,::-1]        # opencv中的通道顺序是BGR，与Python的RGB刚好相反。因此此处需要调整顺序。
    # cv2.namedWindow('%s'%(picture_path), cv2.WINDOW_AUTOSIZE)
    cv2.imshow('%s'%(picture_path.split("/")[-1]),img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()



def main(picture_path):
    """
    主函数：对图像进行检测倾斜，并修正。
    :param picture_path:图像的完整路径
    :return:None
    """
    contours = contours_acquire(picture_path)
    contours_top3 = contours_filter(contours)
    angle_list = calculate_angle(contours_top3)
    if abs(sum(angle_list) / len(angle_list)) > 0.1:
        angle_correct(angle_list, picture_path)




if __name__ == '__main__':  # 防止该脚本在被import时以下语句被执行。只有本脚本自主运行时，以下代码才运行。

    picture_path = "./data/001.jpg"
    main(picture_path)

