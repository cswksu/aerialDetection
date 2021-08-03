import os
from PIL import Image
import numpy as np

# data_path = 'C:/aerialimagelabeling/AerialImageDataset'

# gt_path = '/'.join([data_path, 'train', 'gt'])

# gt_list = os.listdir(gt_path)
# num_zeros = 0
# num_ones = 0

# for gt_im in gt_list:
#     x = Image.open('/'.join([gt_path, gt_im]))
#     x = np.asarray(x, dtype=np.ulonglong)
#     x = x / 255
#     num_pos = np.sum(x)
#     num_neg = 5000 * 50000 - num_pos
#     num_zeros += num_neg
#     num_ones += num_pos

# print('Overall Population stats')
# print('Num pos: ' + str(num_ones))
# print('Num neg: ' + str(num_zeros))
# posProportion = num_ones / float(num_ones + num_zeros)
# print('Percentage Pos: ' + str(100 * posProportion))
# print('Total Pixels: ' + str(5000*5000*180))
num_ones =  710215250
num_zeros = 44289784750
i=9
beta = 1 - np.power(float(10), -i)
print(beta)

effectivePos = (1-np.power(beta, num_ones))/(1-beta)

effectiveNeg = (1-np.power(beta, num_zeros))/(1-beta)
effectiveTot = (1-np.power(beta, num_ones + num_zeros))/(1-beta)
print('Num eff pos: ' + str(effectivePos))
print('Num eff neg: ' + str(effectiveNeg))
print('Num eff tot: ' + str(effectiveTot))
print()