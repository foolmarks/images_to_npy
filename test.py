
import numpy as np

from img_to_npy import *



def main():

  # input args to test function
  image_dir='image_dir'
  label_file='val.txt'
  classes=1000
  input_height=224
  input_width=224
  input_chans=3
  resize=True
  normalize=False
  one_hot=False
  compress=True
  output_file='dataset.npz'


  # call test function
  images_to_npy(image_dir,label_file,classes,input_height,input_width,input_chans,resize,normalize,one_hot,compress,output_file)

  # now load back in and unpack
  train_data = np.load(output_file)
  x_train = train_data['x']
  y_train = train_data['y']

  print(y_train[0])

  # recreate image as PNG
  if normalize:
    cv2.imwrite('test.png',x_train[0]*255)
  else:
    cv2.imwrite('test.png',x_train[0])




  return


if __name__ == '__main__':
  main()

