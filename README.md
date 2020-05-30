# Converting images to numpy files

To save time during training, it can be useful to convert a dataset of images to numpy arrays, pre-process them (scaling, normalization, etc) and then save as one or more binary numpy files.


Each image should be read with the OpenCV imread function which converts an image file (JPEG, PNG, etc) into a numpy array.

```python
# open image as numpy array
img = cv2.imread('image_file.jpg')
```


Then it can be pre-processed as required (resizing, normalization, means subtraction, etc) before being saved into a numpy (.npy) file:

```python
np.save('image.npy', img)
```


During training, the numpy file is then loaded like this:

```python
# train_image is a numpy array
train_image = np.load('image.npy')
```


Obviously, converting each single image into a numpy file is not a viable solution for a large dataset, especially as an image saved into numpy file format is decidedly larger that its compressed JPEG equivalent. We need to ‘push’ as many images as possible into one single numpy file.

We need to first create an array which will hold all of our images:

```python
x = np.ndarray(shape=(num_of_images,height,width,channels), dtype=np.float32)
```

Note how the data type is set for floating-point – the images will normally be unsigned 8bit integers when first opened and it would be ideal to remain in that format as it creates much smaller numpy files, but we will almost certainly need to use sfloating-point due to the image pre-processing…but if you can remain in uint8 format then do so.
Next, we create a loop that will run through all of the images in a folder, pre-process them then insert them into the ‘placeholder’ array. The placeholder array is then saved to a numpy file:

```python
import numpy as np
image_dir = 'image_dir'
# make a list of all the images
imageList = os.listdir(image_dir)

# create a placeholder array
x = np.ndarray(shape=(len(imageList),height,width,channels), dtype=np.float32)

# loop through all images
for i in range(len(imageList)):
      # open image to numpy array
      img = cv2.imread(imageList[i])

      # do all the pre-processing…
      img = pre_process(img)

      # insert into placeholder array
      x[i] = img

# write placeholder array into a binary npy file
np.save('dataset.npy', x)
```

When we need to use this in training, just load it and then index into the resulting numpy array:

```python
# np.load returns a numpy array
x_train = np.load(‘dataset.npy’)

# fetch batches from training dataset
for i in range(num_of_batches):
        x_batch = x_train[i*batchsize:(i+1)*batchsize]
```


With np.save, we can only write one array into a numpy file, but the np.savez function allows us to pack multiple arrays into a single file and this can be very useful for placing both the training data and labels into one single file:

```python
# placeholder arrays for data and labels
# data is float32, labels are integers
x = np.ndarray(shape=(len(imageList),height,width,channels), dtype=np.float32)
y = np.ndarray(shape=(len(imageList)), dtype=np.int32)

# loop through all images
for i in range(len(imageList)):
      # open image to numpy array
      img = cv2.imread(imageList[i])

      # do all the pre-processing…
      img = pre_process(img)

      # insert into placeholder array
      x[i] = img
      y[i] = label

# write placeholder arrays into a binary npz file
np.savez('dataset.npz', x=x, y=y)
```

For .npz files, the np.load function does not directly return numpy arrays, we need to unpack them like this:

```python
train_f = np.load('dataset.npz')
x_train = train_f['x']
y_train = train_f['y']
```

..and then use the resulting numpy arrays (x_train and y_train) as indicated before. The numpy files can also be saved in compressed format using the np.savez_compressed function.

```python
np.savez_compressed('dataset.npz', x=x, y=y)
```

They need to be unpacked in the same way as the non-compressed .npz files.

The example in this repository shows how to read a list of files and labels (..actually a shortened version of the ImageNet validation dataset index file..), pre-process the images by resizing, center-cropping and then normalizing to range 0 to 1.0.  The labels can be one-hot encoded if required. The output file contains both data and labels.
Converting integers to their one-hot encoded equivalent is easy with numpy. If we have for example 10 classes, we need to generate a numpy array that has 10 elements where all of them are zero, except for the bit at index ‘n’, where ‘n’ is the class value. 

```python
total_classes = 1000
label = 65
# placeholder array
y = np.ndarray(shape=(num_of_samples, total_classes), dtype=np.uint32)

# make an array of all zeros
label_1hot = np.zeros(total_classes, dtype=np.uint32)
# set bit #65 to 1
np.put(label_1hot, int(label), 1)
# push into placeholder array
y[i] = label_1hot
```

