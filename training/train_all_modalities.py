import os
import random
import gc, numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import compute_class_weight
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Dense, Dropout,Flatten, BatchNormalization, Conv2D, MultiHeadAttention, concatenate
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve


# OVO Attention
from tensorflow.keras import layers

class OvOAttention(layers.Layer):
    """
    TensorFlow layer that implements One-vs-Others attention mechanism.
    """

    def __init__(self):
        super(OvOAttention, self).__init__()

    def call(self, others, main, W):
        """
        Compute context vector and attention weights using One-vs-Others attention.

        Args:
            others: List of tensors of shape (batch_size, num_heads, seq_len, embed_dim) representing
                    the other modality inputs.
            main: A tensor of shape (batch_size, num_heads, seq_len, embed_dim) representing the main modality input.
            W: A learnable parameter tensor of shape (d_head, d_head) representing the weight matrix.

        Returns:
            A tensor of shape (batch_size, embed_dim) representing the context vector.
            A tensor of shape (batch_size, num_heads, seq_len) representing the attention weights.
        """
        mean = tf.reduce_mean(tf.stack(others, axis=0), axis=0)
        score = tf.matmul(tf.squeeze(mean, 2), W) @ tf.transpose(tf.squeeze(main, 2), perm=[0, 2, 1])
        attn = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(attn, tf.squeeze(main, 2))
        return context, attn

class CustomMultiHeadAttention(layers.Layer):
    """
    TensorFlow layer that implements Multi-Head attention mechanism.
    """

    def __init__(self, d_model=512, num_heads=8):
        super(CustomMultiHeadAttention, self).__init__()
        self.d_head = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.ovo_attn = OvOAttention()
        self.query_proj = layers.Dense(self.d_head * num_heads)
        self.key_proj = layers.Dense(self.d_head * num_heads)
        self.value_proj = layers.Dense(self.d_head * num_heads)
        self.W = self.add_weight(shape=(self.d_head, self.d_head), initializer="random_normal")

    def call(self, others, main):
        """
        Compute context vector using Multi-Head attention.

        Args:
            others: List of tensors of shape (batch_size, num_heads, seq_len, embed_dim) representing
                    the other modality inputs.
            main: A tensor of shape (batch_size, num_heads, seq_len, embed_dim) representing the main modality input.

        Returns:
            A tensor of shape (batch_size, seq_len, embed_dim) representing the context vector.
        """
        batch_size = tf.shape(main)[0]
        main = tf.expand_dims(main, 1)
        bsz, tgt_len, embed_dim = tf.shape(main)
        src_len, _, _ = tf.shape(main)
        
        main = tf.reshape(main, (tgt_len, bsz * self.num_heads, self.d_head))
        main = tf.transpose(main, perm=[1, 0, 2])
        main = tf.reshape(main, (bsz, self.num_heads, tgt_len, self.d_head))
        processed_others = []
        for mod in others:
            mod = tf.expand_dims(mod, 1)
            mod = tf.reshape(mod, (tgt_len, bsz * self.num_heads, self.d_head))
            mod = tf.transpose(mod, perm=[1, 0, 2])
            mod = tf.reshape(mod, (bsz, self.num_heads, tgt_len, self.d_head))
            processed_others.append(mod)

        context, attn = self.ovo_attn(processed_others, main, self.W)
        context = tf.reshape(context, (bsz * tgt_len, embed_dim))
        context = tf.reshape(context, (bsz, tgt_len, tf.shape(context)[1]))

        return context



config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

def make_img(t_img):
    img = pd.read_pickle(t_img)
    img_l = []
    for i in range(len(img)):
        img_l.append(img.values[i][0])
    
    return np.array(img_l)


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
   
               
def create_model_snp():
    
    model = Sequential()
    model.add(Dense(200,  activation = "relu")) 
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(50, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    return model

def create_model_clinical():
    
    model = Sequential()
    model.add(Dense(200,  activation = "relu")) 
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(50, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))    
    return model

def create_model_img():
    
    
    
    model = Sequential()
    model.add(Conv2D(72, (3, 3), activation='relu')) 
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))   
    return model


def plot_classification_report(y_tru, y_prd, mode, learning_rate, batch_size,epochs, figsize=(7, 7), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = ["Control", "Moderate", "Alzheimer's" ] 
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,
                annot=True, 
                cbar=False, 
                xticklabels=xticks, 
                yticklabels=yticks,
                ax=ax, cmap = "Blues")
    
    plt.savefig('report_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'_' + str(epochs)+'.png')
    


def calc_confusion_matrix(result, test_label,mode, learning_rate, batch_size, epochs):
    test_label = to_categorical(test_label,3)

    true_label= np.argmax(test_label, axis =1)

    predicted_label= np.argmax(result, axis =1)
    
    n_classes = 3
    precision = dict()
    recall = dict()
    thres = dict()
    for i in range(n_classes):
        precision[i], recall[i], thres[i] = precision_recall_curve(test_label[:, i],
                                                            result[:, i])


    print ("Classification Report :") 
    print (classification_report(true_label, predicted_label))
    cr = classification_report(true_label, predicted_label, output_dict=True)
    return cr, precision, recall, thres



def cross_modal_attention(x, y):
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)
    a1 = MultiHeadAttention(num_heads = 4,key_dim=50)(x, y)
    a2 = MultiHeadAttention(num_heads = 4,key_dim=50)(y, x)
    a1 = a1[:,0,:]
    a2 = a2[:,0,:]
    return concatenate([a1, a2])

def ovo_modal_attention(x, other_modalities):
    """
    Function to compute attention using One-vs-Others attention mechanism.

    Args:
        x: A tensor representing the main modality input.
        other_modalities: A list of tensors representing other modality inputs.

    Returns:
        Tensor: The result of applying One-vs-Others attention mechanism.
    """
    # Expand the dimensions of the main modality input
    x = tf.expand_dims(x, axis=1)

    # Prepare other modalities
    processed_others = [tf.expand_dims(modality, axis=1) for modality in other_modalities]

    # Initialize the CustomMultiHeadAttention layer
    mha = CustomMultiHeadAttention(num_heads=4, key_dim=50)

    # Apply CustomMultiHeadAttention using OvOAttention
    a1 = mha(processed_others, x)
    a1 = a1[:, 0, :]

    # For symmetry, you might also want to compute the attention with x being treated as the other modality
    # This would be similar to how cross-modal attention was computed in both directions
    # If not needed, you can comment out the following lines
    a2 = mha([x] * len(other_modalities), other_modalities[0])  # Assuming all other modalities are of the same shape
    a2 = a2[:, 0, :]

    # Concatenate the results
    return tf.concat([a1, a2], axis=1)

def self_attention(x):
    x = tf.expand_dims(x, axis=1)
    attention = MultiHeadAttention(num_heads = 4, key_dim=50)(x, x)
    attention = attention[:,0,:]
    return attention
    

def multi_modal_model(mode, train_clinical, train_snp, train_img):
    
    in_clinical = Input(shape=(train_clinical.shape[1]))
    
    in_snp = Input(shape=(train_snp.shape[1]))
    
    in_img = Input(shape=(train_img.shape[1], train_img.shape[2], train_img.shape[3]))
    
    dense_clinical = create_model_clinical()(in_clinical)
    dense_snp = create_model_snp()(in_snp) 
    dense_img = create_model_img()(in_img) 
    
 
        
    ########### Attention Layer ############

    ## One-Versus-Others Attention ##

    if mode == 'MM_OVO':
        main_1 = dense_img
        other_modalities_1 = [dense_clinical, dense_snp]
        main_2 = dense_clinical
        other_modalities_2 = [dense_img, dense_snp]
        main_3 = dense_snp
        other_modalities_3 = [dense_img, dense_clinical]

        out_1 = ovo_modal_attention(main_1, other_modalities_1)
        out_2 = ovo_modal_attention(main_2, other_modalities_2)
        out_3 = ovo_modal_attention(main_3, other_modalities_3)

        merged = concatenate([out_1, out_2, out_3])
        
    ## Cross Modal Bi-directional Attention ##

    if mode == 'MM_BA':
            
        vt_att = cross_modal_attention(dense_img, dense_clinical)
        av_att = cross_modal_attention(dense_snp, dense_img)
        ta_att = cross_modal_attention(dense_clinical, dense_snp)
                
        merged = concatenate([vt_att, av_att, ta_att, dense_img, dense_snp, dense_clinical])
                 
   
        
        
    ## Self Attention ##
    elif mode == 'MM_SA':
            
        vv_att = self_attention(dense_img)
        tt_att = self_attention(dense_clinical)
        aa_att = self_attention(dense_snp)
            
        merged = concatenate([aa_att, vv_att, tt_att, dense_img, dense_snp, dense_clinical])
        
    ## Self Attention and Cross Modal Bi-directional Attention##
    elif mode == 'MM_SA_BA':
            
        vv_att = self_attention(dense_img)
        tt_att = self_attention(dense_clinical)
        aa_att = self_attention(dense_snp)
        
        vt_att = cross_modal_attention(vv_att, tt_att)
        av_att = cross_modal_attention(aa_att, vv_att)
        ta_att = cross_modal_attention(tt_att, aa_att)
            
        merged = concatenate([vt_att, av_att, ta_att, dense_img, dense_snp, dense_clinical])
            
        
    ## No Attention ##    
    elif mode == 'None':
            
        merged = concatenate([dense_img, dense_snp, dense_clinical])
                
    else:
        print ("Mode must be one of 'MM_SA', 'MM_BA', 'MU_SA_BA' or 'None'.")
        return
                
        
    ########### Output Layer ############
        
    output = Dense(3, activation='softmax')(merged)
    model = Model([in_clinical, in_snp, in_img], output)        
        
    return model



def train(mode, batch_size, epochs, learning_rate, seed, save_path):
    
 
    train_clinical = pd.read_csv("X_train_clinical.csv").drop("Unnamed: 0", axis=1).values
    test_clinical= pd.read_csv("X_test_clinical.csv").drop("Unnamed: 0", axis=1).values

    
    train_snp = pd.read_csv("X_train_snp.csv").drop("Unnamed: 0", axis=1).values
    test_snp = pd.read_csv("X_test_snp.csv").drop("Unnamed: 0", axis=1).values

    
    train_img= make_img("X_train_img.pkl")
    test_img= make_img("X_test_img.pkl")

    
    train_label= pd.read_csv("y_train.csv").drop("Unnamed: 0", axis=1).values.astype("int").flatten()
    test_label= pd.read_csv("y_test.csv").drop("Unnamed: 0", axis=1).values.astype("int").flatten()

    reset_random_seeds(seed)
    class_weights = compute_class_weight(class_weight = 'balanced',classes = np.unique(train_label),y = train_label)
    d_class_weights = dict(enumerate(class_weights))
    
    # compile model #
    model = multi_modal_model(mode, train_clinical, train_snp, train_img)
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    

    # summarize results
    history = model.fit([train_clinical,
                         train_snp,
                         train_img],
                        train_label,
                        epochs=epochs,
                        batch_size=batch_size,
                        class_weight=d_class_weights,
                        validation_split=0.1,
                        verbose=1)
                        
                

    score = model.evaluate([test_clinical, test_snp, test_img], test_label)
    
    acc = score[1] 
    test_predictions = model.predict([test_clinical, test_snp, test_img])
    cr, precision_d, recall_d, thres = calc_confusion_matrix(test_predictions, test_label, mode, learning_rate, batch_size, epochs)
    
    # Save model
    model.save_weights(save_path)

    
    """
    plt.clf()
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('accuracy_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('loss_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.clf()
    """
    
 
    
    # release gpu memory #
    K.clear_session()
    del model, history
    gc.collect()
        
        
    print ('Mode: ', mode)
    print ('Batch size:  ', batch_size)
    print ('Learning rate: ', learning_rate)
    print ('Epochs:  ', epochs)
    print ('Test Accuracy:', '{0:.4f}'.format(acc))
    print ('-'*55)
    
    return acc, batch_size, learning_rate, epochs, seed
    
    
if __name__=="__main__":

    # Model saving
    current_script_path = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.abspath(os.path.join(current_script_path, os.pardir))
    checkpoints_directory = os.path.join(parent_directory, 'checkpoints')
    os.makedirs(checkpoints_directory, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(checkpoints_directory, 'model.h5')

    
    m_a = {}
    seeds = random.sample(range(1, 200), 5)
    for s in seeds:
        acc, bs_, lr_, e_ , seed= train('MM_OVO', 32, 50, 0.001, s, MODEL_SAVE_PATH)
        m_a[acc] = ('MM_OVO', acc, bs_, lr_, e_, seed)
    print(m_a)
    print ('-'*55)
    max_acc = max(m_a, key=float)
    print("Highest accuracy of: " + str(max_acc) + " with parameters: " + str(m_a[max_acc]))
    
