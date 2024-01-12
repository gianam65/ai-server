from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from pathlib import Path
import cv2
import numpy as np

class CNN_Model(object):
    def __init__(self, weight_path=None):
        self.weight_path = weight_path
        self.model = None

    def build_model(self, rt=False):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))

        if self.weight_path is not None:
            self.model.load_weights(self.weight_path)
        # self.model.summary()
        if rt:
            return self.model

    @staticmethod
    def load_data():
        dataset_dir = './datasets/'
        images = []
        labels = []

        for img_path in Path(dataset_dir + 'unchoice/').glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            img = img.reshape((28, 28, 1))
            label = to_categorical(0, num_classes=2)
            images.append(img / 255.0)
            labels.append(label)

        for img_path in Path(dataset_dir + 'choice/').glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            img = img.reshape((28, 28, 1))
            label = to_categorical(1, num_classes=2)
            images.append(img / 255.0)
            labels.append(label)

        datasets = list(zip(images, labels))
        np.random.shuffle(datasets)
        images, labels = zip(*datasets)
        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    def plot_roc_auc(self, test_images, test_labels):
        predictions = self.model.predict(test_images)
        
        # Tính FPR, TPR, và AUC
        fpr, tpr, _ = roc_curve(test_labels[:, 1], predictions[:, 1])
        roc_auc = auc(fpr, tpr)

        # Hiển thị đồ thị ROC-AUC
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curve')
        plt.legend()
        plt.show()

    def train(self):
        images, labels = self.load_data()

        self.build_model(rt=False)

        self.model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(1e-3), metrics=['acc'])
        # self.model.summary()

        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, )

        cpt_save = ModelCheckpoint('./weight_test.h5', save_best_only=True, monitor='val_acc', mode='max')

        self.model.fit(images, labels, callbacks=[cpt_save, reduce_lr], verbose=1, epochs=20, validation_split=0.15, batch_size=32,
                       shuffle=True)
        
        # plt.style.use("ggplot")
        # plt.figure()
        # plt.plot(np.arange(1, 21), H.history["loss"], label="train_loss")
        # plt.plot(np.arange(1, 21), H.history["val_loss"], label="val_loss")
        # plt.plot(np.arange(1, 21), H.history["acc"], label="train_acc")
        # plt.plot(np.arange(1, 21), H.history["val_acc"], label="val_acc")
        # plt.title("Training Loss and Acc on CIFAR-10")
        # plt.xlabel("Epoch #")
        # plt.ylabel("Loss/Accuracy")
        # plt.legend()
        # plt.show()

        # plt.plot(history.history['acc'], label='Training Accuracy')
        # plt.plot(history.history['val_acc'], label='Validation Accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.show()
        # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1)
        # cpt_save = ModelCheckpoint('./weight_test.h5', save_best_only=True, monitor='val_acc', mode='max')

        # self.plot_roc_auc(images, labels)


# if __name__ == "__main__":
#     cnn_model = CNN_Model()  
#     cnn_model.train()
