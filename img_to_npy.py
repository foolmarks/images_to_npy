
import os
import numpy as np
import cv2


image_dir = 'image_dir'
label_file = 'val.txt'

image_height=224
image_width=224
image_chans=3
classes=1000

one_hot=True
#one_hot=True

output_file = 'dataset.npz'

compress = True
#compress = False


def center_crop(image,out_height,out_width):
  image_height, image_width = image.shape[:2]
  offset_height = (image_height - out_height) // 2
  offset_width = (image_width - out_width) // 2
  image = image[offset_height:offset_height+out_height, offset_width:offset_width+out_width,:]
  return image


def resize_maintain_aspect(image,target_h,target_w):
  image_height, image_width = image.shape[:2]
  if image_height > image_width:
    new_width = target_w
    new_height = int(image_height*(target_w/image_width))
  else:
    new_height = target_h
    new_width = int(image_width*(target_h/image_height))

  image = cv2.resize(image,(new_width,new_height),interpolation=cv2.INTER_CUBIC)
  return image



def main():

  # open & read text file that lists all images and their labels
  f = open(label_file, 'r') 
  listImages = f.readlines()
  f.close()

  # make placeholder arrays
  x = np.ndarray(shape=(len(listImages),image_height,image_width,3), dtype=np.float32, order='C')

  if (one_hot):
    y = np.ndarray(shape=(len(listImages),classes), dtype=np.uint32, order='C')
  else:
    y = np.ndarray(shape=(len(listImages)), dtype=np.uint32, order='C')
  

  for i in range(len(listImages)):
    
    image_name,label = listImages[i].split()

    # open image to numpy array
    img = cv2.imread(os.path.join(image_dir,image_name))

    # resize
    img = resize_maintain_aspect(img,image_height,image_width)

    # center crop to target height & width
    img = center_crop(img,image_height,image_width)

    # switch to RGB from BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # normalize then write into placeholder
    x[i] = (img/255.0).astype(np.float32)


    if (one_hot):
      label_1hot = np.zeros(classes,dtype=np.uint32,order='C')
      np.put(label_1hot,int(label),1)
      y[i] = label_1hot
    else:
      y[i] = int(label)


  if (compress):
    np.savez_compressed(output_file, x=x, y=y)
  else:
    np.savez(output_file, x=x, y=y)
  
  print(' Saved to',output_file)


  # now load back in and unpack
  train_f = np.load(output_file)
  x_train = train_f['x']
  y_train = train_f['y']

  # this should print 2 identical integers
  _,label_0 = listImages[7].split()
  print(label_0, np.argmax(y_train[7]))


  return





if __name__ == '__main__':
  main()

