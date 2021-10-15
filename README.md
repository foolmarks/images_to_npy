# Converting images to numpy files

To save time during training, it can be useful to convert a dataset of images to numpy arrays, pre-process them (scaling, normalization, etc) and then save as one or more binary numpy files.


Each image should be read with the OpenCV imread function which converts an image file (JPEG, PNG, etc) into a numpy array. We will also reorder the color planes to RGB from the default BGR used by OpenCV:

```python
# open image to numpy array and switch to RGB from BGR
img = cv2.imread(os.path.join(image_dir,image_name))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

We need to first create a placeholder array which will hold all of our images. Next, we create a loop that will run through all of the images in a folder, pre-process them then insert them into the ‘placeholder’ array. The placeholder array is then saved to a numpy file.

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

The img_to_npy.py script in this repository shows how to read a list of files and labels (..actually a shortened version of the ImageNet validation dataset index file..), pre-process the images by resizing, center-cropping and then normalizing to range 0 to 1.0.  The labels can be one-hot encoded if required. The output file contains both data and labels.


The complete list of command line arguments of img_to_npy.py are as follows:

|Argument|Default|Description|
|:-------|:-----:|:----------|
|--image_dir|image_dir|Path to folder containing images|
|--label_file|val.txt|Path to text file that matches labels to images|
|--classes|1000|Total number of classes|
|--resize|False|Resize & center-crop all images to input_height x input_width|
|--normalize|False|Normalize pixels to range 0.0 to 1.0|
|--one_hot|False|One-hot encode the labels if set|
|--compress|False|Compress the output file if set|
|--output_file|dataset.npz|Path to output file|
|--input_height|224|See note 2 below|
|--input_width|224|See note 2 below|
|--input_chans|3|Number of channels in input image|


Notes:

1. If `--normalize` is specified, the x array will be of float32 type, otherwise uint8 is used.
2. If `--resize` is specified, the images will be resized and center cropped to input_height x input_width. If `--resize` is not specified, the images must all be of dimensions input_height x input_width.




