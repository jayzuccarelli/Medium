import gensim.models as gm
import glob as gb
import keras.applications as ka
import keras.layers as kl
import keras.models as km
import keras.optimizers as ko
import keras_preprocessing.image as ki
import keras_preprocessing.sequence as ks
import keras_preprocessing.text as kt
import numpy as np
import pandas as pd
import pickle as pk
import tensorflow as tf
import utils as ut


# Data
Data_dir = np.array(gb.glob('../Data/MELD.Raw/train_splits/*'))
Validation_dir = np.array(gb.glob('../Data/MELD.Raw/dev_splits_complete/*'))
Test_dir = np.array(gb.glob('../Data/MELD.Raw/output_repeated_splits_test/*'))

# Parameters
BATCH = 16
EMBEDDING_LENGTH = 32


# Convert Audio to Spectrograms
for file in Data_dir:
    filename, name = file, file.split('/')[-1].split('.')[0]
    ut.create_spectrogram(filename, name)

for file in Validation_dir:
    filename, name = file, file.split('/')[-1].split('.')[0]
    ut.create_spectrogram_validation(filename, name)

for file in Test_dir:
    filename, name = file, file.split('/')[-1].split('.')[0]
    ut.create_spectrogram_test(filename, name)


# Data Loading
train = pd.read_csv('../Data/MELD.Raw/train_sent_emo.csv', dtype=str)
validation = pd.read_csv('../Data/MELD.Raw/dev_sent_emo.csv', dtype=str)
test = pd.read_csv('../Data/MELD.Raw/test_sent_emo.csv', dtype=str)

# Create mapping to identify audio files
train["ID"] = 'dia'+train["Dialogue_ID"]+'_utt'+train["Utterance_ID"]+'.jpg'
validation["ID"] = 'dia'+validation["Dialogue_ID"]+'_utt'+validation["Utterance_ID"]+'.jpg'
test["ID"] = 'dia'+test["Dialogue_ID"]+'_utt'+test["Utterance_ID"]+'.jpg'


# Text Features
tokenizer = kt.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train['Utterance'])

vocab_size = len(tokenizer.word_index) + 1

train_tokens = tokenizer.texts_to_sequences(train['Utterance'])
text_features = pd.DataFrame(ks.pad_sequences(train_tokens, maxlen=200))

validation_tokens = tokenizer.texts_to_sequences(validation['Utterance'])
validation_features = pd.DataFrame(ks.pad_sequences(validation_tokens, maxlen=200))


# Data Pipeline
def train_generator(features, batch):
    train_generator = ki.ImageDataGenerator(
        rescale=1. / 255.)
    train_generator = train_generator.flow_from_dataframe(
        dataframe=train,
        directory="../Images/Train/",
        x_col="ID",
        y_col="Sentiment",
        batch_size=batch,
        seed=0,
        shuffle=False,
        class_mode="categorical",
        target_size=(64, 64))
    train_iterator = features.iterrows()
    j = 0
    i = 0
    while True:
        genX2 = pd.DataFrame(columns=features.columns)
        while i < batch:
            k,r = train_iterator.__next__()
            r = pd.DataFrame([r], columns=genX2.columns)
            genX2 = genX2.append(r)
            j += 1
            i += 1
            if j == train.shape[0]:
                X1i = train_generator.next()
                train_generator = ki.ImageDataGenerator(
                    rescale=1. / 255.)
                train_generator = train_generator.flow_from_dataframe(
                    dataframe=train,
                    directory="../Images/Train/",
                    x_col="ID",
                    y_col="Sentiment",
                    batch_size=batch,
                    seed=0,
                    shuffle=False,
                    class_mode="categorical",
                    target_size=(64, 64))
                train_iterator = features.iterrows()
                i = 0
                j=0
                X2i = genX2
                genX2 = pd.DataFrame(columns=features.columns)
                yield [X1i[0], tf.convert_to_tensor(X2i.values, dtype=tf.float32)], X1i[1]
        X1i = train_generator.next()
        X2i = genX2
        i = 0
        yield [X1i[0], tf.convert_to_tensor(X2i.values, dtype=tf.float32)], X1i[1]


def validation_generator(features, batch):
    validation_generator = ki.ImageDataGenerator(
        rescale=1. / 255.)
    validation_generator = validation_generator.flow_from_dataframe(
        dataframe=validation,
        directory="../Images/Validation/",
        x_col="ID",
        y_col="Sentiment",
        batch_size=batch,
        seed=0,
        shuffle=False,
        class_mode="categorical",
        target_size=(64, 64))
    validation_iterator = features.iterrows()
    j = 0
    i = 0
    while True:
        genX2 = pd.DataFrame(columns=features.columns)
        while i < batch:
            k,r = validation_iterator.__next__()
            r = pd.DataFrame([r], columns=genX2.columns)
            genX2 = genX2.append(r)
            j += 1
            i += 1
            if j == validation.shape[0]:
                X1i = validation_generator.next()
                validation_generator = ki.ImageDataGenerator(
                    rescale=1. / 255.)
                validation_generator = validation_generator.flow_from_dataframe(
                    dataframe=validation,
                    directory="../Images/Validation/",
                    x_col="ID",
                    y_col="Sentiment",
                    batch_size=batch,
                    seed=0,
                    shuffle=False,
                    class_mode="categorical",
                    target_size=(64, 64))
                validation_iterator = features.iterrows()
                i = 0
                j = 0
                X2i = genX2
                genX2 = pd.DataFrame(columns=features.columns)
                yield [X1i[0], tf.convert_to_tensor(X2i.values, dtype=tf.float32)], X1i[1]
        X1i = validation_generator.next()
        X2i = genX2
        i = 0
        yield [X1i[0], tf.convert_to_tensor(X2i.values, dtype=tf.float32)], X1i[1]


# Model
# Inputs
images = km.Input(shape=(64, 64, 3))
features = km.Input(shape=(200, ))

# Transfer Learning Bases
vgg19 = ka.VGG19(weights='imagenet', include_top=False)
vgg19.trainable = False

# Image Classification Branch
x = vgg19(images)
x = kl.GlobalAveragePooling2D()(x)
x = kl.Dense(32, activation='relu')(x)
x = kl.Dropout(rate=0.25)(x)
x = km.Model(inputs=images, outputs=x)

# Text Classification Branch
y = kl.Embedding(vocab_size, EMBEDDING_LENGTH, input_length=200)(features)
y = kl.SpatialDropout1D(0.25)(y)
y = kl.LSTM(25, dropout=0.25, recurrent_dropout=0.25)(y)
y = kl.Dropout(0.25)(y)
y = km.Model(inputs=features, outputs=y)

combined = kl.concatenate([x.output, y.output])

z = kl.Dense(32, activation="relu")(combined)
z = kl.Dropout(rate=0.25)(z)
z = kl.Dense(32, activation="relu")(z)
z = kl.Dropout(rate=0.25)(z)
z = kl.Dense(3, activation="softmax")(z)

model = km.Model(inputs=[x.input, y.input], outputs=z)

model.compile(optimizer=ko.Adam(lr=0.0001), loss='categorical_crossentropy', metrics='accuracy')

model.summary()


# Hyperparameters
EPOCHS = 13
TRAIN_STEPS = np.floor(train.shape[0]/BATCH)
VALIDATION_STEPS = np.floor(validation.shape[0]/BATCH)

# Model Training
model.fit_generator(generator=train_generator(text_features, BATCH),
                    steps_per_epoch=TRAIN_STEPS,
                    validation_data=validation_generator(validation_features, BATCH),
                    validation_steps=VALIDATION_STEPS,
                    epochs=EPOCHS)

# Performance Evaluation
# Validation
model.evaluate_generator(generator=validation_generator(validation_features, BATCH))

# Save the Model and Labels
model.save('Model.h5')


# Test
test_generator.reset()
predictions = model.predict_generator(test_generator, verbose=True)
predicted_class_indices = np.argmax(predictions, axis=1)

# Fetch labels from train gen for testing
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print(predictions[0:6])