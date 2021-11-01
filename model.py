#GAN network for MAR algorithm, written and compiled using GoogleColab
#Sofia Cannizzaro

#importo librerie:
import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import matplotlib as plt
import seaborn as sns
import cv2
# Utilities for working with remote data
import requests
from io import BytesIO
import zipfile
# Image processing shortcuts
import imageio
import cv2
# SciKit-Image tools for working with the image data
from skimage import exposure
import skimage.morphology as morp
from skimage.filters import rank
from __future__ import division
from os import mkdir
from os.path import join, isdir
from imageio import get_writer
import tensorflow as tf
from PIL import Image
from google.colab import files
from google.colab.patches import cv2_imshow
from keras.preprocessing.image import save_img


#define parameter
BUFFER_SIZE = 400
BATCH_SIZE = 16
IMG_WIDTH = 384
IMG_HEIGHT = 384

#LOADING OF IMAGES

#loading and splitting one image into the three part that forms it: input is an image composed of input image-real image-mask.
def load(image_file):
  w = tf.shape(image_file)[1]
  w = w // 3
  real_image = image_file[:, :w, :]
  input_image = image_file[:,w:2*w , :]
  mask=image_file[:,2*w:,:]
  return input_image, real_image,mask
#resize of the 3 images
def resize(input_image, real_image,mask, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  mask=tf.image.resize(mask, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image, real_image,mask

# normalizing the images to [0, 1]
def normalize(input_image, real_image,mask):

  input_image=input_image/255
  real_image = real_image /255
  mask = mask / 255
  return input_image, real_image,mask
#loading of the images
def load_image_train(image_file):
  input_image, real_image,mask = load(image_file)

  input_image, real_image,mask = normalize(input_image, real_image,mask)

  return input_image, real_image,mask

#loading of images for training and testing 
ctr1 = requests.get(
    'http://144.91.118.156/datoreteproj_1.zip')
ctzip1 = BytesIO(ctr1.content)
ct_fdata1 = zipfile.ZipFile(ctzip1)
proj1 = np.array(
    [imageio.imread(ct_fdata1.open(fname)) for fname in ct_fdata1.namelist()])
ctr2 = requests.get(
    'http://144.91.118.156/datoreteproj_2.zip')
ctzip2 = BytesIO(ctr2.content)
ct_fdata2 = zipfile.ZipFile(ctzip2)
proj2 = np.array(
    [imageio.imread(ct_fdata2.open(fname)) for fname in ct_fdata2.namelist()])

ctr3 = requests.get(
    'http://144.91.118.156/datoreteproj_3.zip')
ctzip3 = BytesIO(ctr3.content)
ct_fdata3 = zipfile.ZipFile(ctzip3)
proj3 = np.array(
    [imageio.imread(ct_fdata3.open(fname)) for fname in ct_fdata3.namelist()])

ctr4 = requests.get(
    'http://144.91.118.156/datoreteproj_4.zip')
ctzip4 = BytesIO(ctr4.content)
ct_fdata4 = zipfile.ZipFile(ctzip4)
proj4 = np.array(
    [imageio.imread(ct_fdata4.open(fname)) for fname in ct_fdata4.namelist()])

ctr5 = requests.get(
    'http://144.91.118.156/datoreteproj_5.zip')
ctzip5 = BytesIO(ctr5.content)
ct_fdata5 = zipfile.ZipFile(ctzip5)
proj5 = np.array(
    [imageio.imread(ct_fdata5.open(fname)) for fname in ct_fdata5.namelist()])

ctr6 = requests.get(
    'http://144.91.118.156/datoreteproj_6.zip')
ctzip6 = BytesIO(ctr6.content)
ct_fdata6 = zipfile.ZipFile(ctzip6)
proj6 = np.array(
    [imageio.imread(ct_fdata6.open(fname)) for fname in ct_fdata6.namelist()])
ctr7 = requests.get(
    'http://144.91.118.156/datoreteproj_7.zip')
ctzip7 = BytesIO(ctr7.content)
ct_fdata7 = zipfile.ZipFile(ctzip7)
proj7 = np.array(
    [imageio.imread(ct_fdata7.open(fname)) for fname in ct_fdata7.namelist()])

ctr8 = requests.get(
    'http://144.91.118.156/datoreteproj_8.zip')
ctzip8 = BytesIO(ctr8.content)
ct_fdata8 = zipfile.ZipFile(ctzip8)
proj8 = np.array(
    [imageio.imread(ct_fdata8.open(fname)) for fname in ct_fdata8.namelist()])

ctr10 = requests.get(
    'http://144.91.118.156/datoreteproj_10.zip')
ctzip10 = BytesIO(ctr10.content)
ct_fdata10 = zipfile.ZipFile(ctzip10)
proj10 = np.array(
    [imageio.imread(ct_fdata10.open(fname)) for fname in ct_fdata10.namelist()])

ctr11 = requests.get(
    'http://144.91.118.156/datoreteproj_11.zip')
ctzip11 = BytesIO(ctr11.content)
ct_fdata11 = zipfile.ZipFile(ctzip11)
proj11 = np.array(
    [imageio.imread(ct_fdata11.open(fname)) for fname in ct_fdata11.namelist()])

ctr12 = requests.get(
    'http://144.91.118.156/datoreteproj_12.zip')
ctzip12 = BytesIO(ctr12.content)
ct_fdata12 = zipfile.ZipFile(ctzip12)
proj12 = np.array(
    [imageio.imread(ct_fdata12.open(fname)) for fname in ct_fdata12.namelist()])
ctr13 = requests.get(
    'http://144.91.118.156/datoreteproj_13.zip')
ctzip13 = BytesIO(ctr13.content)
ct_fdata13 = zipfile.ZipFile(ctzip13)
proj13 = np.array(
    [imageio.imread(ct_fdata13.open(fname)) for fname in ct_fdata13.namelist()])

ctr14 = requests.get(
    'http://144.91.118.156/datoreteproj_14.zip')
ctzip14 = BytesIO(ctr14.content)
ct_fdata14 = zipfile.ZipFile(ctzip14)
proj14 = np.array(
    [imageio.imread(ct_fdata14.open(fname)) for fname in ct_fdata14.namelist()])

ctr15 = requests.get(
    'http://144.91.118.156/datoreteproj_15.zip')
ctzip15 = BytesIO(ctr15.content)
ct_fdata15 = zipfile.ZipFile(ctzip15)
proj15 = np.array(
    [imageio.imread(ct_fdata15.open(fname)) for fname in ct_fdata15.namelist()])

ctr16 = requests.get(
    'http://144.91.118.156/datoreteproj_16.zip')
ctzip16 = BytesIO(ctr16.content)
ct_fdata16 = zipfile.ZipFile(ctzip16)
proj16 = np.array(
    [imageio.imread(ct_fdata16.open(fname)) for fname in ct_fdata16.namelist()])

ctr17 = requests.get(
    'http://144.91.118.156/datoreteproj_17.zip')
ctzip17 = BytesIO(ctr17.content)
ct_fdata17 = zipfile.ZipFile(ctzip17)
proj17 = np.array(
    [imageio.imread(ct_fdata17.open(fname)) for fname in ct_fdata17.namelist()])

ctr9 = requests.get(
    'http://144.91.118.156/datoreteproj_prova.zip')
ctzip9 = BytesIO(ctr9.content)
ct_fdata9 = zipfile.ZipFile(ctzip9)
proj9 = np.array(
    [imageio.imread(ct_fdata9.open(fname)) for fname in ct_fdata9.namelist()])

#creation of training and testing dataset
train_data=[]
proj1 =tf.expand_dims(proj1, axis=3)
proj3 =tf.expand_dims(proj3, axis=3)
proj4 =tf.expand_dims(proj4, axis=3)
proj5 =tf.expand_dims(proj5, axis=3)
proj2 =tf.expand_dims(proj2, axis=3)
proj6 =tf.expand_dims(proj6, axis=3)
proj7 =tf.expand_dims(proj7, axis=3)
proj8 =tf.expand_dims(proj8, axis=3)
proj17 =tf.expand_dims(proj17, axis=3)
proj10 =tf.expand_dims(proj10, axis=3)
proj11 =tf.expand_dims(proj11, axis=3)
proj13 =tf.expand_dims(proj13, axis=3)
proj12 =tf.expand_dims(proj12, axis=3)
proj16 =tf.expand_dims(proj16, axis=3)
proj14 =tf.expand_dims(proj14, axis=3)
proj15 =tf.expand_dims(proj15, axis=3)
for j in range(90):
  train_data.append(proj1[j,:,:,:])
  train_data.append(proj3[j,:,:,:])
  train_data.append(proj4[j,:,:,:])
  train_data.append(proj5[j,:,:,:])
for j in range(90):
  k=90
  z=j+90
  train_data.append(proj2[z,:,:,:])
  train_data.append(proj6[z,:,:,:])
  train_data.append(proj7[z,:,:,:])
  train_data.append(proj8[z,:,:,:])
for j in range(90):
  k=180
  z=j+180
  train_data.append(proj10[z,:,:,:])
  train_data.append(proj12[z,:,:,:])
  train_data.append(proj14[z,:,:,:])
  train_data.append(proj16[z,:,:,:])
for j in range(90):
  k=270
  z=j+270
  train_data.append(proj11[z,:,:,:])
  train_data.append(proj13[z,:,:,:])
  train_data.append(proj15[z,:,:,:])
  train_data.append(proj17[z,:,:,:])
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)

train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_data=[]
proj9 =tf.expand_dims(proj9, axis=3)
for j in range(360):
  test_data.append(proj9[j,:,:,:])
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
test_dataset = test_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE)




#defining downsaple and upsample level, setting droput parameter at 0.3
OUTPUT_CHANNELS = 1
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample1(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(tf.keras.layers.UpSampling2D(
    size=(2, 2), data_format=None, interpolation="nearest"))
  result.add(
    tf.keras.layers.Conv2D(filters, size, strides=1,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
  result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.3))

  result.add(tf.keras.layers.ReLU())

  return result

#GENERATOR model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
def Generator():
  dim=384
  inputs = tf.keras.layers.Input(shape=[dim,dim, 1])
  mask=tf.keras.layers.Input(shape=[dim,dim, 1])
  s=4
  down_stack = [
    downsample(128, s,apply_batchnorm=False),  # (bs, 128, 128, 64)
    downsample(128, s,apply_batchnorm=True),  # (bs, 64, 64, 128)
    downsample(256, s,apply_batchnorm=True),  # (bs, 64, 64, 128)
    downsample(512, s,apply_batchnorm=True),  # (bs, 64, 64, 128)
    downsample(512, s,apply_batchnorm=True)
   ]

  up_stack = [
    upsample1(512, s,apply_dropout=False),  # (bs, 16, 16, 1024)
    upsample1(256,s,apply_dropout=True),  # (bs, 32, 32, 512)
    upsample1(128, s,apply_dropout=True),  # (bs, 64, 64, 256)
    upsample1(64, s,apply_dropout=True),  # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)

  last_1=tf.keras.layers.UpSampling2D(
    size=(2, 2), data_format=None, interpolation="nearest")
  last=tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 3,
                                         strides=1,
                                        padding='same',
                                        kernel_initializer=initializer,
                                         activation='tanh')  # (bs, 256, 256, 3)
  #last3=tf.keras.layers.Dropout(0.5)

  #MASK PYRAMID NETWORK
  avg_pool_1=tf.keras.layers.AveragePooling2D(pool_size=(s,s), strides=2, padding="same")
  avg_pool_2=tf.keras.layers.AveragePooling2D(pool_size=(s,s), strides=2, padding="same")
  avg_pool_3=tf.keras.layers.AveragePooling2D(pool_size=(s,s), strides=2, padding="same")
  avg_pool_4=tf.keras.layers.AveragePooling2D(pool_size=(s,s), strides=2, padding="same")
  avg_pool_5=tf.keras.layers.AveragePooling2D(pool_size=(s,s), strides=2, padding="same")
  x = inputs
  MA=mask
  s_1=avg_pool_1(MA)
  s_2=avg_pool_2(s_1)
  s_3=avg_pool_3(s_2)
  s_4=avg_pool_4(s_3)
  s_5=avg_pool_5(s_4)
  s_recap=[MA,s_1,s_2,s_3,s_4,s_5]
  
  #s_recap=[]
 # Downsampling through the model
  skips = []
  for down in down_stack:
   ind=down_stack.index(down)+1
   x = down(x)
   s_new=s_recap[ind]
   x=concatenate([x,s_new])
   skips.append(x)

  skips = reversed(skips[:-1])
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concatenate([x, skip])


  

  #METODO1
  x=last_1(x)
  x=last(x)
  #x=last3(x)
  #METODO2
  #x=concatenate([x,s_1])
  #x = last(x)


  #NOMMP
 # return tf.keras.Model(inputs=inputs, outputs=[x])
  #MMP
  return tf.keras.Model(inputs=[inputs,mask], outputs=[x])

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

generator.summary()

#gener loss con mask fusion loss

LAMBDA = 100
def generator_loss(G_input1,ycap,target):#aggiungere mask come input?
 # N(s)*(1-D(ycap)=mask_disg_gen_out)
 #mask_gen_out=ycap
  #gan_loss=tf.reduce_mean(tf.square(G_input1))
  image=ycap

  gradient=tf.image.sobel_edges(image)
  #print(gradient)
  grad_mag_components = gradient**2

  grad_mag_square = tf.math.reduce_sum(grad_mag_components,axis=-1) # sum all magnitude components

  grad_mag_img = tf.sqrt(grad_mag_square)
  normgrad=tf.norm(grad_mag_img,ord=2)
  gan_loss=tf.norm(G_input1,ord=2)
 # l1_loss = tf.reduce_mean(tf.abs(target - ycap))
  #l1_loss=tf.norm(target - ycap,ord=1)
  l2_loss=tf.norm(target-ycap,ord=2)
  #print(l2_loss)
  total_gen_loss = gan_loss + (LAMBDA * l2_loss )+normgrad

  return total_gen_loss, gan_loss, l2_loss

#DISCRIMINATORI 
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)
  dim=384
  mask=tf.keras.layers.Input(shape=[dim, dim, 1])
  inp = tf.keras.layers.Input(shape=[dim, dim, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[dim, dim, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

  #zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
  #conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                               # kernel_initializer=initializer,
                               # use_bias=False)(zero_pad1)  # (bs, 31, 31, 512 loro 4-1 noi tolto

  #batchnorm1 = tf.keras.layers.BatchNormalization()(down3)

 # leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  #zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

  conv = tf.keras.layers.Conv2D(1, 3, strides=1,
                                kernel_initializer=initializer)(down3)  # (bs, 30, 30, 1). loro 4-1 noi 3-1
  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  last = tf.keras.layers.LeakyReLU()(batchnorm1)
  avg_pool_1=tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=2, padding="same")
  avg_pool_2=tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=2, padding="same")
  avg_pool_3=tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=2, padding="same")
  avg_pool_4=tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=1)
  #avg_pool_5=tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding="same")
  s=mask
  s_1=avg_pool_1(s)
  s_2=avg_pool_2(s_1)
  s_3=avg_pool_3(s_2)
  s_4=avg_pool_4(s_3)

  
  #s_5=avg_pool_5(s_4)
  #s_recap=[s,s_1,s_2,s_3,s_4,s_5]
  

  return tf.keras.Model(inputs=[inp, tar,mask], outputs= [last,s_4])

discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
#discriminator.summary()

#discri con mask fusion loss
def discriminator_loss(D_input1,D_input2):
   # N(s)*(1-D(y)=D_input2
   #N(s)(D(ycap))=D_input2
  #real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  real_loss = tf.norm(D_input1,ord=2)

  generated_loss = tf.norm(D_input2,ord=2)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.0001,
    decay_steps=1800,
    decay_rate=0.5,
    staircase=True)
#lrate = LearningRateScheduler(lr_schedule)
#step sono numero dati=360*16/dim batch(16), quindi ogni epoch 360 step, se dimezzi ogni 3 epoch viene 1080

#lrate=0.0005
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule , beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule , beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def se(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
  A=np.array(imageA)
  B=np.array(imageB)
  err = np.sum((A - B) ** 2)
	#err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
  return err

def generate_images(model, test_input, tar,mask):
  prediction = model([test_input,mask], training=True)
  pred = np.array(prediction)
  A=tf.identity(mask)
  A=np.array(A)
  #cv2.normalize(A,A, 0, 1, cv2.NORM_MINMAX)
  #cv2.normalize(pred,pred, 0, 1, cv2.NORM_MINMAX)
  inp=np.array(test_input)
  C=np.ones_like(A)- A
  
  s=np.multiply(A,pred)
  im=np.multiply(A,pred)+np.multiply(C,inp)
  plt.pyplot.figure(figsize=(15, 15))
  display_list = [test_input[0], tar[0], mask[0],prediction[0],im[0]]
  title = ['Input Image', 'Ground Truth','mask', 'Generated image','ycap']

  for i in range(5):
    plt.pyplot.subplot(1, 5, i+1)
    plt.pyplot.title(title[i])
    plt.pyplot.imshow(display_list[i][:,:,0] * 0.5 + 0.5,cmap=plt.cm.gray)
    plt.pyplot.axis('off')
  plt.pyplot.show()
  #ssim=tf.image.ssim(display_list[1]
    #, display_list[4], max_val=1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
 # print(f'ssim={ssim}')
  err=se(display_list[1], display_list[4])
  #ssim=tf.image.ssim(display_list[1]
    #, display_list[4], max_val=1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
  print(f'error={err}')
  return err

#for example_input, example_target,example_mask in train_dataset.take(1):
 # generate_images(generator, example_input, example_target,example_mask)

EPOCHS = 100
import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

#training con mask fusion loss
import cv2
import copy
@tf.function
def train_step(input_image, target,mask, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    #G(x)
    G_x = generator([input_image,mask], training=True)
    #G=tf.identity(G_x)
    #G=tf.add(tf.multiply(0.5,G),0.5)
    #y=target
    #x=input_image
    #y^
    ycap=tf.math.multiply(mask,G_x)+tf.math.multiply((tf.ones_like(mask)- mask),input_image)
    #D(y^)
    D_ycap,N_mask=discriminator([input_image,ycap,mask],training=True)
    #D(y)
    D_y,N_mask= discriminator([input_image, target,mask], training=True)
    #N(s)x(1-D(y^))
    G_input1=tf.math.multiply(N_mask,tf.ones_like(D_ycap)-D_ycap)
    #N(s)x(1-D(y))
    D_input1=tf.math.multiply(N_mask,tf.ones_like(D_y)-D_y)
    #N(s)x(D(y^))
    D_input2=tf.math.multiply(N_mask,D_ycap)
    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(G_input1, ycap,target)
    disc_loss = discriminator_loss(D_input1,D_input2)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

error=[]
def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    for example_input, example_target,example_mask in test_ds.take(1):
      err=generate_images(generator, example_input, example_target,example_mask)
      error.append(err)
    print("Epoch: ", epoch)
    differr=abs(np.diff(error[-6:]))
    if ((epoch>5) and (all(i<0.01 for i in differr))):
         #print(np.diff(err[-6:]))
         break
    # Train
    for n, (input_image, target,mask) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, mask,epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix=checkpoint_prefix)

#!pip install -U tensorboard

##docs_infra: no_execute
#%load_ext tensorboard
#%tensorboard --logdir {log_dir}

fit(train_dataset, EPOCHS, test_dataset)

#no grad normal1 con 0.0001 1800 e 0.5 dropout
print(error)

#con 0.0001 1800 e 0.5 dropout
print(error)

#0.0005 con 1800 ultimi tre con 0.5 dropout
 print(error)

print((error))
#con gran da 0.001 con 1800

#tre dtop con 0.5 grad:1 da 0.0001 con 1800
print((error))

#download prova dataset per download
ctrp = requests.get(
    'http://144.91.118.156/datoreteproj_1.zip')
ctzip = BytesIO(ctrp.content)
ct_fdata = zipfile.ZipFile(ctzip)
proj = np.array(
    [imageio.imread(ct_fdata.open(fname)) for fname in ct_fdata.namelist()])
print(proj.shape)
prova_data=[]
proj =tf.expand_dims(proj, axis=3)
for j in range(360):
  prova_data.append(proj[j,:,:,:])
prova_dataset = tf.data.Dataset.from_tensor_slices(prova_data)
prova_dataset = prova_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)

prova_dataset = prova_dataset.batch(1)

j=0
error2=np.zeros(360)

import PIL
for inp, tar,mask in prova_dataset:
  G_x = generator([inp,mask], training=False)
  ycap=tf.math.multiply(mask,G_x)+tf.math.multiply((tf.ones_like(mask)- mask),inp)
  S=se(tar
  ,ycap)
  
  error2[j]=S
  j=j+1

print(np.amax(error2))
print(np.amin(error2))
print(np.mean(error2))

print(np.amax(error2))
print(np.amin(error2))
print(np.mean(error2))

print(error)
plt.pyplot.plot(error[10:])
plt.pyplot.ylabel('error')
plt.pyplot.show()

tf.keras.models.save_model(
    generator, 'modelproj30.h5', overwrite=True, include_optimizer=True, save_format=None,
    signatures=None, options=None, save_traces=True
)
files.download('modelproj30.h5')

!tensorboard dev upload --logdir  {log_dir}   --name "modelproj23"

kill $(ps -e | grep 'tensorboard' | awk '{print $1}')
