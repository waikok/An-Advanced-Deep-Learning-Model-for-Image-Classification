
#%%
#1. Import packages
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, metrics, callbacks, applications
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

#%%
#2. Set parameters
BATCH_SIZE = 32
data_dir = 'dataset'
EPOCHS = 5
IMG_SIZE = (128,128)
SEED = 123

# %%
#3. Data loading and Preparation
#(A) Load the data into tensorflow dataset using the specific method
train_dataset = keras.utils.image_dataset_from_directory(data_dir,validation_split=0.3,subset='training',seed=SEED,shuffle=True,image_size=IMG_SIZE,batch_size=BATCH_SIZE)
val_dataset = keras.utils.image_dataset_from_directory(data_dir,validation_split=0.3,subset='validation',seed=SEED,shuffle=True,image_size=IMG_SIZE,batch_size=BATCH_SIZE)

# %%
#4. Display some images as example
class_names = train_dataset.class_names

plt.figure(figsize=(10,10))
for images,labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')

# %%
#5. Convert the BatchDataset into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = val_dataset.prefetch(buffer_size=AUTOTUNE)

# %%
#6. Create a small pipeline for data augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

# %%
#7. Apply the data augmentation to test it out
for images,labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')


#%%
#8. Apply transfer learning
#(A) Import MobileNetV3Large
"""
MobileNetV3Large comes with preprocess_input and global_avg layer packaged in it, so we don't need to create those layers by ourselves.
"""
IMG_SHAPE = IMG_SIZE + (3,)
feature_extractor = applications.MobileNetV3Large(input_shape=IMG_SHAPE,include_top=False,weights='imagenet',pooling='avg')

#Disable the training for the feature extractor (freeze the layers)
feature_extractor.trainable = False
feature_extractor.summary()
keras.utils.plot_model(feature_extractor,show_shapes=True)

# %%
#9. Create the classification layers
l2 = keras.regularizers.L2()
output_layer = layers.Dense(len(class_names),activation='softmax',kernel_regularizer=l2)

#%%
#10. Use funtional API to create the entire model pipeline
#feature_extractor.trainable = False
inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = feature_extractor(x)
x = layers.Dropout(0.3)(x)

outputs = output_layer(x)

# %%
#11. Model development
model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

# %%
#12. Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['acc'])

# %%
# 13. Create the tensorboard callbacks
es = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)
log_path = os.path.join('log_dir',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_dir=log_path)

#%%
#14. Train the model
history = model.fit(pf_train,validation_data=pf_val,epochs=EPOCHS,callbacks=[es,tb])

# %%
#15. Evaluate the model
history.history.keys()
test_loss,test_acc = model.evaluate(pf_val)

print("Loss = ",test_loss)
print("Accuracy = ",test_acc)

#16. Plot the chart
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'])
plt.show()

#%%
#17. Deploy the model using the test data
image_batch, label_batch = pf_val.as_numpy_iterator().next()
predictions = np.argmax(model.predict(image_batch),axis=1)

plt.figure(figsize=(20,20))

for i in range(len(image_batch)):
    plt.subplot(8,4,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title(f"Label: {class_names[label_batch[i]]}, Prediction: {class_names[predictions[i]]}")
    plt.axis('off')
plt.show()

#%%
#18. Model Saving
# To create folder if not exists
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

saved_path = os.path.join('saved_models','model.h5')
model.save(saved_path)
