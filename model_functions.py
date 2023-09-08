import tensorflow
from tensorflow.keras.models import Model, load_model
from keras.layers import Flatten, Dense, Input, Dropout, Activation, BatchNormalization
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# image manipulation
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

# face alignment
from mtcnn.mtcnn import MTCNN
from keras.regularizers import l2

# model metrics
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# common packages
import os
import numpy as np
import pandas as pd

def mae(x,y):
    return mean_absolute_error(x,y)

def cor(x,y):
    return pearsonr(x,y)[0]

def auc(label, pred):
    return roc_auc_score(label, pred)
    
def process_arr(arr, version):
    img = cv2.resize(arr, (224, 224)).astype('float32')
    img = np.expand_dims(img, 0)
    img = img / 255.0
    img = utils.preprocess_input(img, version = version)
    return img

def img2arr(img_path, version):
    if isinstance(img_path, str):
        img = load_img(img_path)
    else:
        img = Image.fromarray(np.uint8(img_path))
    img = img_to_array(img)
    img = process_arr(img, version)
    return img

def imgs2arr(img_names, img_dir, version = 1):
    imgs = []
    for img in img_names:
        imgs += [img2arr(os.path.join(img_dir, str(img)), version)]
    return np.concatenate(imgs)

def crop_img(im,x,y,w,h):
    return im[y:(y+h),x:(x+w),:]

def input_generator(data, bs, img_dir, is_train=True, version=1):
    sex_map = {'Male': 1, 'Female': 0}
    loop = True
    epoch_count = 0
    
    # Define the data augmentation transformations
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    while loop:
        if is_train:
            sampled = data.sample(bs)
            x = imgs2arr(sampled['name'], img_dir, version)
            y = [sampled['bmi'].values, sampled['gender'].map(lambda i: sex_map.get(i, 0)).values]
        else:
            if len(data) >= bs:
                sampled = data.iloc[:bs, :]
                data = data.iloc[bs:, :]
                x = imgs2arr(sampled['name'], img_dir, version)
                y = [sampled['bmi'].values, sampled['gender'].map(lambda i: sex_map.get(i, 0)).values]
            else:
                loop = False
                if len(data) > 0:
                    x = imgs2arr(data['name'], img_dir, version)
                    y = [data['bmi'].values, data['gender'].map(lambda i: sex_map.get(i, 0)).values]        
        if is_train:
            # Apply data augmentation transformations
            augmented_images = []
            for img in x:
                augmented_images.append(img)
                for _ in range(bs - 1):
                    augmented_img = datagen.random_transform(img)
                    augmented_images.append(augmented_img)
            
            x = np.array(augmented_images)
            y = [np.repeat(y[0], bs), np.repeat(y[1], bs)]
            
        yield x, y

class FacePrediction(object):
    
    def __init__(self, img_dir, model_type = 'vgg16', sex_thresh = 0.05):
        self.model_type = model_type
        self.img_dir = img_dir
        self.detector = MTCNN()
        self.sex_thresh = sex_thresh
        if model_type in ['vgg16','vgg16_fc6']:
            self.version = 1
        else:
            self.version = 2
    
    
    def define_model(self, hidden_dim = 256, drop_rate=0.5, freeze_backbone = True):
        
        if self.model_type == 'vgg16_fc6':
            vgg_model = VGGFace(model = 'vgg16', include_top=True, input_shape=(224, 224, 3))
            last_layer = vgg_model.get_layer('fc6').output
            flatten = Activation('relu')(last_layer)
        else:
            vgg_model = VGGFace(model = self.model_type, include_top=False, input_shape=(224, 224, 3))
            last_layer = vgg_model.output
            flatten = Flatten()(last_layer)
        
        if freeze_backbone:
            for layer in vgg_model.layers:
                layer.trainable = False
                
        def block(flatten, name):
            x = Dense(hidden_dim, name=name + '_fc1', kernel_regularizer=l2(0.001))(flatten)
            x = BatchNormalization(name = name + '_bn1')(x)
            x = Activation('relu', name = name+'_act1')(x)
            x = Dropout(drop_rate)(x)
            return x
        
        x = block(flatten, name = 'bmi')
        out_bmi = Dense(1, activation='linear', name='bmi')(x)
                
        x = block(flatten, name = 'gender')
        out_sex = Dense(1, activation = 'sigmoid', name = 'gender')(x)
        
        # fine tuning
        x2 = Dense(hidden_dim)(flatten)
        x2 = Add()([x2, x])
        x2 = block(x2, name = 'bmi')
        out_bmi = Dense(1, activation='linear', name='bmi')(x2)

        custom_vgg_model = Model(vgg_model.input, [out_bmi, out_sex])
        custom_vgg_model.compile('adam', 
                                 {'bmi':'mae','gender':'binary_crossentropy'},
                                 {'gender': 'accuracy'}, 
                                 loss_weights={'bmi': 0.9, 'gender':0.1})

        self.model = custom_vgg_model
        

    def train(self, train_data, valid_data, bs, epochs, callbacks):
        train_gen = input_generator(train_data, bs, self.img_dir, True, self.version)
        valid_gen = input_generator(valid_data, bs, self.img_dir, False, self.version)
        model_history = self.model.fit(x, y, len(train_data) // bs, epochs, 
                                                 validation_data = valid_gen, 
                                                 validation_steps = len(valid_data) //  bs, 
                                                 callbacks=callbacks)
        return model_history
    
        
    def evaluate(self, valid_data):
        imgs = valid_data['name'].values
        arr = imgs2arr(imgs, self.img_dir, self.version)
        bmi, sex = self.model.predict(arr)        
        gender_values  = valid_data.gender.map({'Male': 1, 'Female': 0}).values.astype(float)
        gender_values = np.array(gender_values)  # Convert to NumPy array        
        metrics = {'bmi_mae':mae(bmi[:,0], valid_data.bmi.values), 
                   'bmi_cor':cor(bmi[:,0], valid_data.bmi.values),
                   'gender_auc':auc(gender_values, sex[:,0])}
        return metrics

    def save_model(self, model_dir):
        self.model.save(model_dir)
        
    def load_model(self, model_dir):
        self.model = load_model(model_dir)
      
    def detect_faces(self, face_path, confidence):
        img = load_img(face_path)
        img = img_to_array(img)
        box = self.detector.detect_faces(img)
        box = [i for i in box if i['confidence'] > confidence]
        res = [crop_img(img, *i['box']) for i in box]
        res = [process_arr(i, self.version) for i in res]
        return box, res
    
    def rt_detect_faces(self, frame, confidence):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box = self.detector.detect_faces(img)
        box = [i for i in box if i['confidence'] > confidence]
        res = [crop_img(img, *i['box']) for i in box]
        return box, res
    
    def predict(self, img_dir, show_img = True):
        if os.path.isdir(img_dir):
            imgs = os.listdir(img_dir)
            arr = imgs2arr(imgs, img_dir, self.version)
        else:
            arr = img2arr(img_dir, self.version)
        preds = self.model.predict(arr)
        
        if show_img and os.path.isdir(img_dir):
            bmi, sex = preds
            num_plots = len(imgs)
            ncols = 5
            nrows = int((num_plots - 0.1) // ncols + 1)
            fig, axs = plt.subplots(nrows, ncols)
            fig.set_size_inches(3 * ncols, 3 * nrows)
            for i, img in enumerate(imgs):
                col = i % ncols
                row = i // ncols
                axs[row, col].imshow(plt.imread(os.path.join(img_dir,img)))
                axs[row, col].axis('off')
                axs[row, col].set_title('BMI: {:3.1f} GENDER: {:2.1f}'.format(bmi[i,0], sex[i,0]), fontsize = 10)        
        return preds
    
    def predict_df(self, img_dir):
        assert os.path.isdir(img_dir), 'input must be directory'
        fnames = os.listdir(img_dir)
        bmi, sex = self.predict(img_dir)
        res = pd.DataFrame({'img':fnames, 'bmi':bmi[:,0], 'gender':sex[:,0]})
        res['sex_prob'] = res['gender']
        res['gender'] = res['gender'].map(lambda i: 'Male' if i > self.sex_thresh else 'Female')        
        return res
    
    def predict_faces(self, img_path, show_img=True, color="white", fontsize=12, confidence=0.95, fig_size=(16, 12)):
        assert os.path.isfile(img_path), 'only single image is supported'
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        boxes, faces = self.detect_faces(img_path, confidence)
        preds = [self.model.predict(face) for face in faces]

        if show_img:
            num_box = len(boxes)
            fig, ax = plt.subplots()
            fig.set_size_inches(fig_size)
            ax.imshow(img)
            ax.axis('off')
            for idx, box in enumerate(boxes):
                bmi, sex = preds[idx]
                box_x, box_y, box_w, box_h = box['box']
                rect = patches.Rectangle((box_x, box_y), box_w, box_h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(box_x, box_y,
                        'BMI:{:3.1f}\nSEX:{:s}'.format(bmi[0, 0], 'M' if sex[0, 0] > self.sex_thresh else 'F'),
                        color=color, fontsize=fontsize)
            plt.show()            
        return preds       
        
    def rt_predict_faces(self, show_img=True, color="white", fontsize=12, confidence=0.95, fig_size=(16, 12)):
        cap = cv2.VideoCapture(0)
                
        while True: 
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640,480))
            boxes, faces = self.rt_detect_faces(frame, confidence)
            
            if show_img:
                for box, face in zip(boxes, faces):
                    x, y, w, h = box['box']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face = crop_img(frame, x, y, w, h)
                    face = process_arr(face, self.version)
                    bmi, sex = self.model.predict(face)
                    cv2.putText(frame, 'BMI:{:3.1f} SEX:{:s}'.format(bmi[0, 0], 'M' if sex[0, 0] > self.sex_thresh else 'F'),
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                cv2.imshow('Real-Time Face Detection and BMI Prediction', frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()