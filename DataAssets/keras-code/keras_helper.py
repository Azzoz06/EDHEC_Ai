''' 
This helper file contains essential utility functions which will be required
by the main driver file in the generated code.
'''


import os
import sys
import PIL

import numpy as np
import pickle as pkl

from PIL import Image
from io import BytesIO
from zipfile import ZipFile
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords as sw
from nltk.stem import PorterStemmer

#### Custom imports - Start ####
from keras import utils
from keras.layers import Layer
import keras.backend as K
#### Custom imports - End ####

############### Custom Layers ###############


def abs_val(x):
    ''' Helper function to build the `AbsVal` activation layer
    '''
    return K.abs(x)

def argmax(x):
    ''' Helper function to build the `AbsVal` activation layer
    '''
    return K.argmax(x, axis=-1)

class LRN2D(Layer):
    ''' This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    '''

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError(
                'LRN2D only works with odd n. n provided: ' + str(n))
        super(LRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = K.shape(X)
        half_n = self.n // 2
        input_sqr = K.square(X)
        extra_channels = K.zeros((b, ch + 2 * half_n, r, c))
        input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],
                                   input_sqr,
                                   extra_channels[:, half_n + ch:, :, :]],
                                  axis=1)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i + ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Power(Layer):
    def __init__(self, shift, scale, power, **kwargs):
        super(Power, self).__init__(**kwargs)
        self.shift = shift
        self.scale = scale
        self.power = power

    def call(self, inputs):
        return K.pow((self.shift + self.scale * inputs), self.power)

    def get_config(self):
        config = {
            'shift': float(self.shift),
            'scale': float(self.scale),
            'power': float(self.power)
        }
        base_config = super(Power, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

############### Utility functions ###############


def calc_new_shape(old_shape, new_shape):
    ''' Calculates the new shape of an incoming tensor which goes as input
    to the Reshape layer. Assumptions:  0  -> copy input dimension and
                                        -1 -> infer new dimension
    There can be only one '-1' in `params`

    Returns a well-formed tuple of `new_shape`
    '''
    old_shape = old_shape[1:]
    old_shape = [int(i) for i in old_shape]
    new_shape = [i for i in new_shape if i is not None]
    new_shape = [i for i in new_shape if i != -2]

    minus_one_count = new_shape.count(-1)
    if minus_one_count > 1:
        raise ValueError(
            'There can be only one -1 in the `new_shape` of a reshape layer')

    for idx, val in enumerate(new_shape):
        if val == 0:
            if idx > len(old_shape) - 1:
                raise ValueError('Invalid shape given to a reshape layer')
            new_shape[idx] = old_shape[idx]

    if minus_one_count == 1:
        total_prod = np.prod(old_shape)
        new_prod = np.prod(new_shape) * -1
        if total_prod % new_prod != 0:
            raise ValueError(str(old_shape) +
                             ' cannot be reshaped into' + str(new_shape))
        new_shape[new_shape.index(-1)] = int(total_prod / new_prod)

    return tuple(new_shape)

############### Image pre-processing ###############


def if_grayscale(img):
    img = img.convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            if r != g != b:
                return False
    return True


def image_unpkl(path, dim_ordering, resize, num_classes):
    with open(path, 'rb') as f:
        (data, label) = pkl.load(f)
        data = np.array(data)
        label = np.array(label)
        if (len(data.shape) == 3):
            if dim_ordering == 'channels_last':
                data = data.reshape(
                    data.shape[0], data.shape[1], data.shape[2], 1).astype('float32')
            elif dim_ordering == 'channels_first':
                data = data.reshape(
                    data.shape[0], 1, data.shape[1], data.shape[2]).astype('float32')
        label = utils.to_categorical(label, num_classes)

        return data, label, None


def image_unzip(path, dim_ordering, resize, num_classes):
    ignore_list = ('__', '.')
    format_check = False
    is_grayscale = False
    data, label, label_names = [], [], []

    with ZipFile(path) as archive:
        classCounter = 0
        all_folders = [f for f in archive.infolist() if f.filename.endswith('/')]
        for i, cfolder in enumerate(all_folders):
            if cfolder.filename.startswith(ignore_list):
                continue
            all_files = [f.filename for f in archive.infolist()
                         if f.filename.startswith(cfolder.filename) and not f.filename.endswith('/')]
            
            for f in all_files:
                splits = f.split('/')
                if splits[1].startswith(ignore_list):
                    all_files.remove(f)

            if not format_check:
                img = archive.read(all_files[0])
                img = BytesIO(img)
                img = Image.open(img)
                is_grayscale = if_grayscale(img)
                format_check = True

            for j, cimage in enumerate(all_files):
                img = archive.read(cimage)
                img = BytesIO(img)
                if is_grayscale:
                    img = Image.open(img).convert('L')
                else:
                    img = Image.open(img).convert('RGB')
                img = img.resize((resize[0], resize[1]), PIL.Image.ANTIALIAS)
                img.load()
                cdata = np.asarray(img, dtype="int32")
                data.append(cdata)
                label.append(classCounter)
                label_names.append(cfolder.filename.split('/')[0])
            classCounter += 1

        data = np.asarray(data)
        label = np.asarray(label)
        label_names = np.asarray(label_names)

        if (len(data.shape) == 3):
            if dim_ordering == 'channels_last':
                data = data.reshape(
                    data.shape[0], data.shape[1], data.shape[2], 1).astype('float32')
            elif dim_ordering == 'channels_first':
                data = data.reshape(
                    data.shape[0], 1, data.shape[1], data.shape[2]).astype('float32')
        label = utils.to_categorical(label, num_classes)

        return data, label, label_names


def image_data_handler(params):
    train_path = params.get('train_dataset', None)
    val_path = params.get('val_dataset', None)
    test_path = params.get('test_dataset', None)
    num_classes = params.get('num_classes', -1)
    rows = params.get('rows', -1)
    cols = params.get('cols', -1)
    format = params.get('dbformat', None)
    dim_ordering = params.get('dim_ordering', 'channels_last')

    # defining these here in case they are not defined below
    val_x, val_y = [], []
    test_x, test_y = [], []

    if not train_path:
        raise ValueError(
            'Input Data missing required parameter: `train_dataset`')
    if num_classes == -1:
        raise ValueError('Input Data missing required parameter: `classes`')
    if rows == -1 or cols == -1:
        raise ValueError('Input Data missing required parameter: `rows/cols`')
    if not format:
        raise ValueError('Input Data missing required parameter: `dbformat`')

    split = False
    if not val_path and not test_path:
        val_split = params.get('validation_split', 0.1)
        test_split = params.get('test_split', 0.1)
        if val_split > 0 or test_split > 0:
            split = True

    if format == 'Zip':
        train_x, train_y, labels = image_unzip(
            train_path, dim_ordering, (rows, cols), num_classes)
        if val_path:
            val_x, val_y, _ = image_unzip(val_path, dim_ordering, (rows, cols), num_classes)
        if test_path:
            test_x, test_y, _ = image_unzip(
                test_path, dim_ordering, (rows, cols), num_classes)
    elif format == 'Python Pickle':
        train_x, train_y, labels = image_unpkl(
            train_path, dim_ordering, (rows, cols), num_classes)
        if val_path:
            val_x, val_y, _ = image_unpkl(
                val_path, dim_ordering, (rows, cols), num_classes)
        if test_path:
            test_x, test_y, _ = image_unpkl(
                test_path, dim_ordering, (rows, cols), num_classes)
    else:
        print('This format is not yet supported. Please use .pkl or .zip')
        sys.exit()

    if split:
        train_x, rest_x, train_y, rest_y = train_test_split(
            train_x, train_y, test_size=val_split + test_split)
        val_x, test_x, val_y, test_y = train_test_split(
            rest_x, rest_y, test_size=test_split / (val_split + test_split))

    print('Training data shape: ' + str(train_x.shape))
    if len(val_x) > 0: print('Validation data shape: ' + str(val_x.shape))
    if len(test_x) > 0: print('Testing data shape: ' + str(test_x.shape))

    return {
        'train_x': train_x,
        'train_y': train_y,
        'val_x': val_x,
        'val_y': val_y,
        'test_x': test_x,
        'test_y': test_y,
        'labels': labels
    }

############### Text pre-processing ###############


def lower(data):
    def to_lower(x): return x.strip().lower()
    return [to_lower(d) for d in data]


def remove_stopwords(data, stop_words, separator):
    sentences_processed = []
    for sentence in data:
        curr_sentence = [i for i in sentence.split(
            separator) if i not in stop_words]
        sentences_processed.append(' '.join(curr_sentence))
    return sentences_processed


def stem(data, separator):
    ps = PorterStemmer()
    sentences_processed = []
    for sentence in data:
        curr_sentence = [ps.stem(i) for i in sentence.split(separator)]
        sentences_processed.append(' '.join(curr_sentence))
    return sentences_processed


def build_vocab(data, separator):
    all_words = []
    for sent in data:
        all_words.extend(sent.split(separator))
    vocab_list = sorted(set(all_words))

    word2idx = {}
    idx2word = {}
    for i, word in enumerate(vocab_list):
        word2idx[word] = i + 1
        idx2word[i + 1] = word

    # add an entry for padding string => ''
    vocab_list.append('')
    word2idx[''] = 0
    idx2word[0] = ''

    return vocab_list, word2idx, idx2word


def prepare_lm(data, word2idx, ctx_len):
    input_idx = []
    output_idx = []

    for sent in data:
        splits = sent.split()
        word_list = [word2idx[word] for word in splits]
        if len(word_list) < ctx_len + 1:
            word_list = np.pad(
                word_list, (ctx_len + 1 - len(word_list), 0), 'constant')
        for x in range(len(word_list) - ctx_len):
            input_idx.append(word_list[x:x + ctx_len])
            output_idx.append(word_list[x + ctx_len])

    input_idx = np.array(input_idx)
    output_idx = np.array(output_idx)
    output_one_hot = np.zeros((len(output_idx), len(word2idx) + 1))
    output_one_hot[np.arange(len(output_idx)), output_idx] = 1

    return input_idx, output_idx, output_one_hot


def get_max_len(data):
    max_len = 5 			# This should ideally be passed by the user
    for sent in data:
        l = len(sent.split())
        if max_len < l:
            max_len = l
    return max_len


def prepare_classification(data, labels, label2class, word2idx, separator, max_len=5):
    input = []
    output = []

    for sent in data:
        splits = sent.split(separator)
        word_list = [word2idx[word] for word in splits]
        if len(word_list) < max_len + 1:
            input.append(
                np.pad(word_list, (max_len + 1 - len(word_list), 0), 'constant'))
        elif len(word_list) > max_len + 1:
            input.append(np.array(word_list[:-max_len + 1]))
        else:
            input.append(np.array(word_list))

    input = np.array(input)
    for l in labels:
        output.append(label2class[l])
    output = np.array(output)

    output_one_hot = np.zeros((len(output), len(label2class)))
    output_one_hot[np.arange(len(output)), output] = 1

    return input, output, output_one_hot

def split_x_y(path, task, sep):
    x, y = [], []
    if task == 'Lang_Model':
        with open(path, 'r') as f:
            x = f.read().split('\n')
    elif task == 'Classification':
        with open(path, 'r') as f:
            data = f.read().split('\n')
            for line in data:
                io = line.split(sep)
                x.append(io[0])
                y.append(io[1])

    return x, y

def text_data_handler(params):
    nltk.download('stopwords')
    stop_words = set(sw.words('english'))
    train_path = params.get('train_dataset', None)
    val_path = params.get('val_dataset', None)
    test_path = params.get('test_dataset', None)
    lowercase = params.get('lowercase', False)
    ipopseparator = params.get('ipopseparator', ',')
    separator = params.get('separator', ' ')
    stopwords = params.get('stopwords', False)
    stemming = params.get('stemming', False)
    num_classes = params.get('num_classes', -1)
    vocab_path = params.get('vocab_path', None)             # Skipping. There is no good way to handle this yet
    task = params.get('task', None)
    ctx_len = params.get('ctx_len', 3)

    # defining these here in case they are not defined below
    val_x, val_y = [], []
    test_x, test_y = [], []

    if not train_path:
        raise ValueError(
            'Input Data missing required parameter: `train_dataset`')
    if not task:
        raise ValueError('Input Data missing required parameter: `task`')
    if num_classes == -1 and task == 'Classification':
        raise ValueError('Input Data missing required parameter: `classes`')
    
    split = False
    if not val_path and not test_path:
        val_split = params.get('validation_split', 0.1)
        test_split = params.get('test_split', 0.1)
        if val_split > 0 or test_split > 0:
            split = True

    train_x, train_y = split_x_y(train_path, task, ipopseparator)
    if val_path: val_x, val_y = split_x_y(val_path, task, ipopseparator)
    if test_path: test_x, test_y = split_x_y(test_path, task, ipopseparator)

    if(lowercase):
        train_x = lower(train_x)
        if val_path: val_x = lower(val_x)
        if test_path: test_x = lower(test_x)
    if(stopwords):
        train_x = remove_stopwords(train_x, stop_words, separator)
        if val_path: val_x = remove_stopwords(val_x, stop_words, separator)
        if test_path: test_x = remove_stopwords(test_x, stop_words, separator)
    if(stemming):
        train_x = stem(train_x, separator)
        if val_path: val_x = stem(val_x, separator)
        if test_path: test_x = stem(test_x, separator)

    vocab_list, word2idx, idx2word = build_vocab(train_x + val_x + test_x, separator)
    if task == 'Lang_Model': train_x, _, train_y = prepare_lm(train_x, word2idx, ctx_len)
    elif task == 'Classification':
        labels = set(train_y + val_y + test_y)
        label2class = {}
        for i, l in enumerate(labels):
            label2class[l] = i
        
        max_len = get_max_len(train_x)
        train_x, _, train_y = prepare_classification(train_x, train_y, label2class, word2idx, separator,max_len)
        if len(val_x) > 0: val_x, _, val_y = prepare_classification(val_x, val_y, label2class, word2idx, separator, max_len)
        if len(test_x) > 0: test_x, _, test_y = prepare_classification(test_x, test_y, label2class, word2idx, separator, max_len)

    if split:
        train_x, rest_x, train_y, rest_y = train_test_split(
            train_x, train_y, test_size=val_split + test_split)
        val_x, test_x, val_y, test_y = train_test_split(
            rest_x, rest_y, test_size=test_split / (val_split + test_split))
        
    print('Training data shape: ' + str(train_x.shape))
    if len(val_x) > 0: print('Validation data shape: ' + str(val_x.shape))
    if len(test_x) > 0: print('Testing data shape: ' + str(test_x.shape))

    return {
        'train_x': train_x,
        'train_y': train_y,
        'val_x': val_x,
        'val_y': val_y,
        'test_x': test_x,
        'test_y': test_y,
        'vocab_length': len(vocab_list)
    }
