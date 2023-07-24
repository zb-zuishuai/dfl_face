# import onnxruntime
#
# use_gpu = True
# providers = onnxruntime.get_available_providers()
# print(providers)
#
# import cv2
# import rawpy
#
# # 打开CR2文件
# raw = rawpy.imread('test_img/2.CR2')
#
# # 将原始图转换为RGB图像
# # rgb = raw.postprocess()
# rgb = raw.copy()
# rgb.set_white_balance(1, 1)  # 设置白平衡为默认值
# rgb.set_auto_wb(False)  # 关闭自动白平衡
#
# # 创建窗并显示图像
# cv2.namedWindow('CR2 Image', cv2.WINDOW_NORMAL)
# cv2.imshow('CR2 Image', rgb)
# cv2.imwrite('test_img/22.png',rgb[:,:,::-1])
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import rawpy

with rawpy.imread('test_img/1.CR2') as raw:
    # 转换为RGB图像
    rgb_image = raw.postprocess()
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('test_img/1test.png', bgr_image)