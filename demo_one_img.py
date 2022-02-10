# from numpy.core.fromnumeric import shape
# import torch
# from torchvision import transforms
import numpy as  np
import cv2
# import time

# from skimage import io
# from skimage import transform

import onnxruntime as rt
# import onnx

# img = io.imread("/Users/cr/git/face/women/301/w_301_605_1.png")
bgr_img = cv2.imread("w_301_605_1.png")
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

tensor_data = np.transpose(rgb_img, [2, 0, 1])
# tensor_data = img.transpose((2,0,1))
tensor_data = np.reshape(tensor_data, (1, 3, np.shape(rgb_img)[0], np.shape(rgb_img)[1]))
tensor_data = tensor_data/255.0
tensor_data = tensor_data.astype(np.float32)

data = np.array(np.random.randn(2,3,np.shape(rgb_img)[0], np.shape(rgb_img)[1]), dtype=np.float32)
data[0, :, :, :] = tensor_data
data[1, :, :, :] = tensor_data

sess = rt.InferenceSession('rvm_mobilenetv3_fp32.onnx')
rec = [ np.zeros([1, 1, 1, 1], dtype=np.float32) ] * 4  # Must match dtype of the model.
downsample_ratio = np.array([1], dtype=np.float32)
fgr, alpha, *rec = pred_onx = sess.run([], {'src': data,'r1i': rec[0],'r2i': rec[1],'r3i': rec[2],'r4i': rec[3],'downsample_ratio': downsample_ratio})

# alpha = np.reshape(alpha, (np.shape(rgb_img)[0], np.shape(rgb_img)[1]))
# alpha = alpha*255
# alpha = alpha.astype(np.uint8)

print("end")
# cv2.imshow("out", bgr_img)
# cv2.imshow("alpha", alpha)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
