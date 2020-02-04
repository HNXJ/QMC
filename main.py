# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 21:56:52 2020

@author: hamed
"""


from __future__ import absolute_import, division, print_function, unicode_literals
from matplotlib import pyplot as plt
from PIL import Image
from models import *
import tensorflow as tf
import numpy as np
import argparse
import zipfile
import pickle
import time
import wget
import json
import re
import os


class FileHandler1:

  def __init__(self, root=''):
    
#    self.download(folder_path='jsons')
    print("\n File handling ... \n -> Opening JSON files ...")
    
    with open(root + 'captions.json') as f:
        self.captions_json = json.load(f)
    with open(root + 'labels.json') as f:
        self.labels_json = json.load(f)

    with open(root + 'categories_info.json') as f:
        self.categories_json = json.load(f)
    with open(root + 'images_info.json') as f:
        self.images_json = json.load(f)

    print("Done :) \n -> Creating dictionaries ...")

    self.labels = dict()
    self.captions = dict()
    self.categories = dict()

    self.images = dict()
    self.images_info = dict()
    
    for item in self.categories_json:
      self.categories[item['id']] = item['supercategory'] + ' : ' + item['name']

    for item in self.labels_json:
      self.labels[item['image_id']] = self.categories[item['category_id']]

    for item in self.captions_json:
      self.captions[item['image_id']] = item['caption']

    for item in self.images_json:
      self.images[item['file_name']] = self.captions[item['id']]
      try:
        self.images_info[item['file_name']] = self.labels[item['id']]
      except:
        self.images_info[item['file_name']] = ' '
    
    print("Done :) \n")
    return

  def create_data(self, data_size=4000, images_path='/content/Dataset/val2017/'):
      
      print('Creating data ...')
      captions = []
      images_name = []
      cnt = 0
      for item in list(self.images.keys()):
          caption = '<start> ' + self.images[item] + ' <end>'
          image_path = images_path + item
          images_name.append(image_path)
          captions.append(caption)
          cnt += 1
          if cnt >= data_size:
              break
          
      print('Data transforming ...')
      train_captions, img_name_vector = shuffle(captions,
                                                images_name,
                                                random_state=1)
      print('Done :)')
      return train_captions, img_name_vector

  @staticmethod
  def download(folder_path=''):

    wget.download('https://github.com/HNXJ/QMC/raw/master/captions.json', out=folder_path)
    wget.download('https://github.com/HNXJ/QMC/raw/master/categories_info.json', out=folder_path)
    wget.download('https://github.com/HNXJ/QMC/raw/master/images_info.json', out=folder_path)
    wget.download('https://github.com/HNXJ/QMC/raw/master/labels.json', out=folder_path)

    print("All files are downloaded to " + folder_path)
    return


class FileOutput:

  def __init__(self, root='', root_lab='labeling_images/', root_cap='captioning_images/'):
    try:
        with open(root + 'image_captioning_info.json') as f:
            self.captions_json = json.load(f)
    except:
        print('No image captioning json file')
    
    try:
        with open(root + 'image_labeling_info.json') as f:
            self.labels_json = json.load(f)
    except:
        print('No image labeling json file')

    print("Done :) \n -> Creating dictionaries ...")

    self.images_l = dict()
    self.images_c = dict()
    
    try:
      for item in self.captions_json:
        self.images_l.append(root_cap + item['file_name'])
    except:
      pass
    
    try:
      for item in self.labels_json:
        self.images_l.append(root_lab + item['file_name'])
    except:
      pass
    
    print("Done :) \n")
    return


def net_test(img_url='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Pagani_Huayra_at_Goodwood_2014_001.jpg/1280px-Pagani_Huayra_at_Goodwood_2014_001.jpg'):
    
    image_extension = img_url[:-4]
    image_path = tf.keras.utils.get_file('image5' + image_extension,
                                         origin=img_url)
    
    result, attention_plot = evaluate(image_path)
    caption_ = ' '.join(result)
    print ('Prediction Caption:', caption_)
    plot_attention(image_path, result, attention_plot)
    
    ii = Image.open(image_path)
    ii = np.array(ii)[:, :, :3]
    
    plt.figure(figsize=(14, 8))
    plt.imshow(ii)
    plt.title(caption_)
    plt.show()
    return


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def undzip(path):
    folder_path = '/Trained'
    z = zipfile.ZipFile(path, 'r')
    z.extractall(folder_path)
    print('File ' + path + 'Extracted to ' + folder_path)
    return
     

def save_model_w(fc_encoder_, decoder_):
    save_model_weights(fc_encoder_, model_name='FCEncoder')
    save_model_weights(decoder_, model_name='RNNDecoder')


def load_image(image_path_):
    img_ = tf.io.read_file(image_path_)
    img_ = tf.image.decode_jpeg(img_, channels=3)
    img_ = tf.image.resize(img_, (299, 299))
    img_ = tf.keras.applications.inception_v3.preprocess_input(img_)
    return img_, image_path_


class Data:
    def __init__(self):
        pass

    def back_to_zero(self, input):
        return tf.slice(input)


    @tf.function
    def load_data(self, inputs):
        new_x = self.back_to_zero(inputs)

        return new_x

# @tf.function
def evaluate(image, image_encoder, fc_encoder, decoder, tokenizer):
    attention_features_shape = 64
    # attention_plot = np.zeros((49, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_encoder(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = fc_encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    d = Data()
    # print('3')
    for i in range(49):
        # print(i, dec_input.shape, features.shape, hidden.shape)
        predictions, hidden, _ = decoder(dec_input, features, hidden)
#        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0]
        # print(predicted_id, predicted_id, type(predicted_id))
        predicted_id = predicted_id.numpy()
        result.append(tokenizer.index_word[predicted_id])
        if tokenizer.index_word[predicted_id] == '<end>':
            return result, _
        dec_input = tf.expand_dims([predicted_id], 0)
#    attention_plot = attention_plot[:len(result), :]
    return result


def random_test_data(image_encoder, fc_encoder, decoder):
    rid = np.random.randint(0, len(img_name_val))
    image = img_name_val[rid]
    target_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
    result, attention_plot = evaluate(image, image_encoder, fc_encoder, decoder)
    
    caption_ = ' '.join(result)
    print ('Main:', target_caption)
    print ('Prediction:', ' '.join(result))
    plot_attention(image, result, attention_plot)
    
    ii = Image.open(image)
    ii = np.array(ii)[:, :, :3]
    
    plt.figure(figsize=(14, 8))
    plt.imshow(ii)
    plt.title(caption_)
    plt.show()
    return


def load_model_caption(embedding_dim=256, units=512, vocab_size_vec=5001, path='Model_cap.zip'):
    
    print('Loading models ...')
    image_encoder_ = CNN_Encoder('inceptionv3')
    print('CNN loaded')
    fc_encoder_ = EFC_Encoder(embedding_dim)
    print('FCNN loaded')
    decoder_ = RNN_Decoder(embedding_dim, units, vocab_size_vec)
    print('RNN loaded')
#    try:
#        undzip(path)
#    except:
#        print('load_from_folder' + path)
    
    
    fc_encoder_.load_weights('TrainedC/FCEncoder/m0.ckpt')
    decoder_.load_weights('TrainedC/RNNDecoder/m0.ckpt')
    print('Models are loaded.')
    return image_encoder_, fc_encoder_, decoder_


def load_model_label(embedding_dim=256, units=512, vocab_size_vec=101, path='Model_cap.zip'):
    print('Loading models ...')
    image_encoder_ = CNN_Encoder('inceptionv3')
    fc_encoder_ = EFC_Encoder(embedding_dim)
    decoder_ = RNN_Decoder(embedding_dim, units, vocab_size_vec)
    
#    try:
#        undzip(path)
#    except:
#        print('load_from_folder' + path)
    
    
    fc_encoder_.load_weights('TrainedL/FCEncoder/m0.ckpt')
    decoder_.load_weights('TrainedL/RNNDecoder/m0.ckpt')
    print('Models are loaded.')
    return image_encoder_, fc_encoder_, decoder_


def TxtCapOutput(Image_encoder, Fc_encoder, Decoder, tokenizer):
    fo = FileOutput()
    file1 = open("captions.txt", "w")
    for image_path in fo.images_c:
        result, _ = evaluate(image_path, Image_encoder, Fc_encoder, Decoder)
        caption_ = ' '.join(result)
        for i in range(10):
            file1.write(caption_ + '\n')

    file1.close()
    print('TXT captions are in the file captions.txt now')
    return


def TxtLabOutput(Image_encoder, Fc_encoder, Decoder, tokenizer):
    fo = FileOutput()
    file2 = open("labels.txt", "w")
    for image_path in fo.images_c:
        result, _ = evaluate(image_path, Image_encoder, Fc_encoder, Decoder, tokenizer)
        label_ = ' '.join(result)
        for i in range(10):
            file2.write(label_ + '\n')

    file2.close()
    print('TXT labels are in the file labels.txt now')
    return


def load_tokenizer_cap():
    with open('tokenizerc.pickle', 'rb') as handle:
        print('tokenizer for captions loaded.')
        return pickle.load(handle)    


def load_tokenizer_lab():
    with open('tokenizerl.pickle', 'rb') as handle:
        print('tokenizer for label loaded.')
        return pickle.load(handle)    


def run():
    
    Image_encoder1, Fc_encoder1, Decoder1 = load_model_caption()
    TxtCapOutput(Image_encoder1, Fc_encoder1, Decoder1, tokenizerC)
    
    Image_encoder2, Fc_encoder2, Decoder2 = load_model_label()
    TxtLabOutput(Image_encoder2, Fc_encoder2, Decoder2, tokenizerL)


def run_img(image_path1 = 'img/VN1.jpg'):
    
    a = evaluate(image_path1, Image_encoder1, Fc_encoder1, Decoder1, tokenizerC)
    b = evaluate(image_path1, Image_encoder2, Fc_encoder2, Decoder2, tokenizerL)
    ii = Image.open(image_path1)
    
    cap_ = ' '.join(a[0])
    lab_ = ' '.join(b[0])
    ii = np.array(ii)[:, :, :3]
    plt.figure(figsize=(20, 10))
    
    plt.imshow(ii)
    plt.title('Title = ' + cap_)
    plt.xlabel('Label = ' + lab_)
    plt.show()


tokenizerC = load_tokenizer_cap()
tokenizerL = load_tokenizer_lab()
Image_encoder1, Fc_encoder1, Decoder1 = load_model_caption()
Image_encoder2, Fc_encoder2, Decoder2 = load_model_label()
    
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--autorun", required=False,
	help="Autorun, on or off")
ap.add_argument("-i", "--img", required=False,
    help="Image path")
args = vars(ap.parse_args())

if args["autorun"]:
    run()        
else:
    try:
        run_img(args["img"])
    except:
        run_img()
