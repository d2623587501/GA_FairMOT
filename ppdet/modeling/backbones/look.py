import numpy as np
import PIL.Image as Image
import paddle

def fix_value(ipt):
    pix_max = np.max(ipt)    # 获取最大值和最小值 - 获取取值范围
    pix_min = np.min(ipt)
    base_value = np.abs(pix_min) + np.abs(pix_max) # 获取两者距离
    base_rate = 255 / base_value     # 求每个单位之间的平均距离
    pix_left = base_rate * pix_min   
    ipt = ipt * base_rate - pix_left  # 整体偏移使其最小值为0
    ipt[ipt < 0] = 0.	# 防止意外情况，增加程序健壮性
    ipt[ipt > 255] = 1.
    return ipt

def concat_feature(feature, im_name):
    batch_num = feature.shape[0]
    im_list = None
    for i in range(batch_num):
        one_feature = feature[i]
        result = np.sum(one_feature, axis=0)	# 将所有通道数据叠加
        im = fix_value(result) # 进行颜色拉伸
        if im_list is None:
            im_list = im
        else:
            im_list = np.append(im_list, im, axis=1)  # 将每一张处理好的图像数据
    im = Image.fromarray(np.array(im_list).astype('uint8')).convert("RGB")  # 转换为Pillow对象
    im.save(im_name + ".jpg")  # 保存图像数据       

# a = paddle.rand([4,4,25,25])
# a = np.array(a)
# print(a[0].shape)
# concat_feature(a,im_name='a')










