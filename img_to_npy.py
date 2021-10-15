
import os, sys
import numpy as np
import cv2
import argparse


DIVIDER = '-------------------------------------'


def center_crop(image,out_height,out_width):
  input_height, input_width = image.shape[:2]
  offset_height = (input_height - out_height) // 2
  offset_width = (input_width - out_width) // 2
  image = image[offset_height:offset_height+out_height, offset_width:offset_width+out_width,:]
  return image


def resize_maintain_aspect(image,target_h,target_w):
  input_height, input_width = image.shape[:2]
  if input_height > input_width:
    new_width = target_w
    new_height = int(input_height*(target_w/input_width))
  else:
    new_height = target_h
    new_width = int(input_width*(target_h/input_height))

  image = cv2.resize(image,(new_width,new_height),interpolation=cv2.INTER_CUBIC)
  return image



def images_to_npy(image_dir,label_file,classes,input_height,input_width,input_chans,resize,normalize,one_hot,compress,output_file):

  # open & read text file that lists all images and their labels
  f = open(label_file, 'r') 
  listImages = f.readlines()
  f.close()

  # make image placeholder array - float32 if resizing and/or normalizing
  if normalize:
    x = np.ndarray(shape=(len(listImages),input_height,input_width,input_chans), dtype=np.float32, order='C')
  else:
    x = np.ndarray(shape=(len(listImages),input_height,input_width,input_chans), dtype=np.uint8, order='C')

  # make labels placeholder array
  if (one_hot):
    y = np.ndarray(shape=(len(listImages),classes), dtype=np.uint8, order='C')
  elif(classes<=256):
    y = np.ndarray(shape=(len(listImages)), dtype=np.uint8, order='C')
  elif(classes<=65536):
    y = np.ndarray(shape=(len(listImages)), dtype=np.uint16, order='C')
  else:
    y = np.ndarray(shape=(len(listImages)), dtype=np.uint32, order='C')



  for i in range(len(listImages)):

    image_name,label = listImages[i].split()

    # open image to numpy array and switch to RGB from BGR
    img = cv2.imread(os.path.join(image_dir,image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # optionally resize & center crop
    if resize:
      img = resize_maintain_aspect(img,input_height,input_width)
      img = center_crop(img,input_height,input_width)
    
    # optionally normalize then write into placeholder array
    if normalize:
      x[i] = (img/255.0).astype(np.float32)
    else:
      x[i] = img

    # optionally 1-hot encode the label, then write into placeholder array
    if (one_hot):
      label_1hot = np.zeros(classes,dtype=np.uint32,order='C')
      np.put(label_1hot,int(label),1)
      y[i] = label_1hot
    else:
      y[i] = int(label)


  # report data types used
  print(' x shape:',x.shape)
  print(' x data type:',x[0].dtype)
  print(' y shape:',y.shape)
  print(' y data type:',y[0].dtype)


  # write output file
  if (compress):
    np.savez_compressed(output_file, x=x, y=y)
  else:
    np.savez(output_file, x=x, y=y)

  print(' Saved to',output_file)

  return  



def run_main():
    
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-id','--image_dir',    type=str,   default='image_dir', help='Full path of folder containing images.')
  ap.add_argument('-l', '--label_file',   type=str,   default='val.txt',   help='Full path of label file.')
  ap.add_argument('-c', '--classes',      type=int,   default=1000,        help='Number of classes.')
  ap.add_argument('-ih','--input_height', type=int,   default=224,         help='Input image height in pixels.')
  ap.add_argument('-iw','--input_width',  type=int,   default=224,         help='Input image width in pixels.')
  ap.add_argument('-ic','--input_chans',  type=int,   default=3,           help='Input image channels.')
  ap.add_argument('-r', '--resize',       action='store_true', help='Resize and center crop images if set. Default is no resize.')
  ap.add_argument('-n', '--normalize',    action='store_true', help='Normalize pixels to range 0,1 if set. Default is no normalization.')
  ap.add_argument('-oh','--one_hot',      action='store_true',  help='One-hot encode the labels if set. Default is no encoding.')
  ap.add_argument('-cp','--compress',     action='store_true', help='Compress the output file if set, otherwise no compression. Default is no compression.')
  ap.add_argument('-o', '--output_file',  type=str,   default='dataset.npz', help='Full path of output file.')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print('--image_dir     : ',args.image_dir    )
  print('--label_file    : ',args.label_file   )
  print('--classes       : ',args.classes      )
  print('--input_height  : ',args.input_height )
  print('--input_width   : ',args.input_width  )
  print('--input_chans   : ',args.input_chans  )
  print('--resize        : ',args.resize       )
  print('--normalize     : ',args.normalize    )
  print('--one_hot       : ',args.one_hot      )
  print('--compress      : ',args.compress     )
  print('--output_file   : ',args.output_file  )
  print(DIVIDER)
  

  images_to_npy(args.image_dir, args.label_file, args.classes, args.input_height, \
  args.input_width, args.input_chans, args.resize, args.normalize, args.one_hot, args.compress, args.output_file)


if __name__ == '__main__':
  run_main()

