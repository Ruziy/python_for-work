from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
image = 'images_for_unet/img/0002.jpg'
mask = 'images_for_unet/masks/0002.png'

image = imread(image)
mask = imread(mask)
mask = resize(mask,(mask.shape[0],mask.shape[1]))
fig,axs = plt.subplots(nrows=1,ncols=2)
axs[0].set_title("Img")
axs[0].set_axis_off()
axs[0].imshow(image)
axs[1].set_title("Mask")
axs[1].set_axis_off()
axs[1].imshow(mask)

plt.show()
