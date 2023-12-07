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
from tensorflow.keras import layers
import pickle
import pdb
import re



########### CSCI 2952G: Added OVO Attention ############    

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

    def __init__(self, d_model=50, num_heads=5):
        super(CustomMultiHeadAttention, self).__init__()
        self.d_head = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.ovo_attn = OvOAttention()
        self.query_proj = layers.Dense(self.d_head * num_heads)
        self.key_proj = layers.Dense(self.d_head * num_heads)
        self.value_proj = layers.Dense(self.d_head * num_heads)
        self.W = self.add_weight(shape=(self.d_head, self.d_head), initializer="random_normal")

    def call(self, others, main, return_attention_weights=False):
        """
        Compute context vector using Multi-Head attention.

        Args:
            others: List of tensors of shape (batch_size, num_heads, seq_len, embed_dim) representing
                    the other modality inputs.
            main: A tensor of shape (batch_size, num_heads, seq_len, embed_dim) representing the main modality input.

        Returns:
            A tensor of shape (batch_size, seq_len, embed_dim) representing the context vector.
        """
        main = tf.expand_dims(main, 1)
        sh = tf.shape(main)
        bsz = sh[0]
        tgt_len = sh[1]
        embed_dim = sh[2]

        main = tf.reshape(main, (tgt_len, bsz * self.num_heads, self.d_head))
        main = tf.transpose(main, perm=[1, 0, 2])
        main = tf.reshape(main, (bsz, self.num_heads, tgt_len, self.d_head))

        processed_others = []
        mod = others[0]
        mod = tf.expand_dims(mod, 1)
        mod = tf.reshape(mod, (tgt_len, bsz * self.num_heads, self.d_head))
        mod = tf.transpose(mod, perm=[1, 0, 2])
        mod = tf.reshape(mod, (bsz, self.num_heads, tgt_len, self.d_head))
        processed_others.append(mod)

        mod = others[1]
        mod = tf.expand_dims(mod, 1)
        mod = tf.reshape(mod, (tgt_len, bsz * self.num_heads, self.d_head))
        mod = tf.transpose(mod, perm=[1, 0, 2])
        mod = tf.reshape(mod, (bsz, self.num_heads, tgt_len, self.d_head))
        processed_others.append(mod)

        #pdb.set_trace()
        # print("BEFORE OVO")
        # print(tf.shape(main))
        context, attn = self.ovo_attn(processed_others, main, self.W)
        # print("AFTER OVO")
        # print(tf.shape(context))
        # context = tf.reshape(context, (bsz * tgt_len, embed_dim))
        # context = tf.reshape(context, (bsz, tgt_len, tf.shape(context)[1]))

        if return_attention_weights:
            return context, attn
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
    
    plt.savefig('../reports/report_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'_' + str(epochs)+'.png')
    


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


########### CSCI 2952G: Utilize OVO Attention Class ############    
def ovo_modal_attention(x, other_modalities, return_attention_weights=False):
    """
    Function to compute attention using One-vs-Others attention mechanism.

    Args:
        x: A tensor representing the main modality input.
        other_modalities: A list of tensors representing other modality inputs.
        return_attention_weights (bool): If True, returns attention weights along with the outputs.

    Returns:
        Tensor: The result of applying One-vs-Others attention mechanism.
        Tensor (optional): The attention weights if return_attention_weights is True.
    """
    # Expand the dimensions of the main modality input
    x = tf.expand_dims(x, axis=1)

    # Prepare other modalities
    processed_others = [tf.expand_dims(modality, axis=1) for modality in other_modalities]

    # Initialize the CustomMultiHeadAttention layer
    mha = CustomMultiHeadAttention(num_heads=5)

    # Apply CustomMultiHeadAttention using OvOAttention
    a1, weights1 = mha(processed_others, x, return_attention_weights=True)
    a1 = a1[:, 0, :]

    # For symmetry, compute the attention with x being treated as the other modality
    a2, weights2 = mha([x] * len(other_modalities), other_modalities[0], return_attention_weights=True)
    a2 = a2[:, 0, :]

    # Concatenate the results
    output = tf.concat([a1, a2], axis=1)

    if return_attention_weights:
        return output, (weights1, weights2)
    else:
        return output


def self_attention(x):
    x = tf.expand_dims(x, axis=1)
    attention = MultiHeadAttention(num_heads = 4, key_dim=50)(x, x)
    attention = attention[:,0,:]
    return attention
    

def multi_modal_model(mode, train_clinical, train_snp, train_img, return_attention_weights=False):
    
    in_clinical = Input(shape=(train_clinical.shape[1]))
    
    in_snp = Input(shape=(train_snp.shape[1]))
    
    in_img = Input(shape=(train_img.shape[1], train_img.shape[2], train_img.shape[3]))
    
    dense_clinical = create_model_clinical()(in_clinical)
    dense_snp = create_model_snp()(in_snp) 
    dense_img = create_model_img()(in_img) 
    
 
        
    ########### Attention Layer ############
    
    ########### CSCI 2952G: Add OVO Attention Mode ############    
    ## One-Versus-Others Attention ##

    if mode == 'MM_OVO':
        main_1 = dense_img
        other_modalities_1 = [dense_clinical, dense_snp]
        main_2 = dense_clinical
        other_modalities_2 = [dense_img, dense_snp]
        main_3 = dense_snp
        other_modalities_3 = [dense_img, dense_clinical]
        
        ########### CSCI 2952G: Add Attention Weight Extraction ############
        if return_attention_weights:
            out_1, attn_weights_1 = ovo_modal_attention(main_1, other_modalities_1, return_attention_weights=True)
            out_2, attn_weights_2 = ovo_modal_attention(main_2, other_modalities_2, return_attention_weights=True)
            out_3, attn_weights_3 = ovo_modal_attention(main_3, other_modalities_3, return_attention_weights=True)
        else:
            out_1 = ovo_modal_attention(main_1, other_modalities_1)
            out_2 = ovo_modal_attention(main_2, other_modalities_2)
            out_3 = ovo_modal_attention(main_3, other_modalities_3)

        merged = concatenate([out_1, out_2, out_3])



        
    
    
    ## Cross Modal Bi-directional Attention ##

    elif mode == 'MM_BA':
            
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

    if return_attention_weights:
        if mode == 'MM_OVO':
            return model, [attn_weights_1, attn_weights_2, attn_weights_3]
        elif mode == 'MM_BA':
            return model, [vt_att, av_att, ta_att]
        elif mode == 'MM_SA':
            return model, [vv_att, tt_att, aa_att]
        elif mode == 'MM_SA_BA':
            return model, [vt_att, av_att, ta_att]


    return model



def train(mode, batch_size, epochs, learning_rate, seed, save_path):
    
    
    train_clinical = pickle.load(open("../ADNI/X_train_clinical.pkl", 'rb')).values
    test_clinical= pickle.load(open("../ADNI/X_test_clinical.pkl", 'rb')).values

    
    train_snp = pickle.load(open("../ADNI/X_train_snp.pkl", 'rb')).values
    test_snp = pickle.load(open("../ADNI/X_test_snp.pkl", 'rb')).values

    train_img= make_img("../ADNI/X_train_img.pkl")
    test_img= make_img("../ADNI/X_test_img.pkl")

    train_label= pickle.load(open("../ADNI/img_y_train.pkl", 'rb')).values.astype("int").flatten()
    test_label= pickle.load(open("../ADNI/img_y_test.pkl", 'rb')).values.astype("int").flatten()
    

    print("Training data shapes: ")
    print('Clinical: ', train_clinical.shape)
    print('SNP: ', train_snp.shape)
    print('Image: ', train_img.shape)
    print('Labels: ', train_label.shape)
    print()

    print("Testing data shapes: ")
    print('Clinical: ', test_clinical.shape)
    print('SNP: ', test_snp.shape)
    print('Image: ', test_img.shape)
    print('Labels: ', test_label.shape)
    print()

    # train_label = np.zeros(215)
    # test_label = np.zeros(24)

    reset_random_seeds(seed)
    class_weights = compute_class_weight(class_weight = 'balanced',classes = np.unique(train_label),y = train_label)
    d_class_weights = dict(enumerate(class_weights))
    
    # compile model #
    #model, attention_weights = multi_modal_model(mode, train_clinical, train_snp, train_img, return_attention_weights=True)
    model = multi_modal_model(mode, train_clinical, train_snp, train_img, return_attention_weights=False)

    model.compile(optimizer=Adam(learning_rate = learning_rate), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    
    # Visualize attention weights
    #visualize_attention_weights(attention_weights, 'Attention Weights Visualization')

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
    #model.load_weights(save_path)

    score = model.evaluate([test_clinical, test_snp, test_img], test_label)
    
    # SALIENCY MAPS
    saliency_maps = compute_multi_modal_saliency_maps(model, [test_clinical, test_snp, test_img])
    visualize_some_saliency(test_img, saliency_maps, save_path)
    
    pdb.set_trace()

    acc = score[1] 
    test_predictions = model.predict([test_clinical, test_snp, test_img])
    cr, precision_d, recall_d, thres = calc_confusion_matrix(test_predictions, test_label, mode, learning_rate, batch_size, epochs)
    
    # Classification report
    #plot_classification_report(test_label, test_predictions, mode, learning_rate, batch_size, epochs)

    # Save model
    model.save_weights(save_path)
    
    
    plt.clf()
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('../reports/accuracy_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('../reports/loss_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.clf()

    
 
    
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
    

########### CSCI 2952G: Explainability Core ############    


def visualize_attention_weights(attention_weights, title):
    """
    Assumes attention_weights is a list of numpy arrays
    """

    # Loop through modalities
    print(type(attention_weights))
    for i, attn in enumerate(attention_weights):
        print(attn)
        plt.figure(figsize=(10, 4))
        # Make heatmap of attention weights
        sns.heatmap(attn.numpy(), cmap='viridis')
        plt.title(f'{title} - Modality {i+1}')
        plt.xlabel('Sequence Length')
        plt.ylabel('Heads')
        plt.show()

def visualize_some_saliency(test_img, saliency_maps, save_path):
    for mri_index_now in range(24):
        extract_mode = lambda s: re.search(r'model_(.*?)\.h5', s).group(1)
        MODE = extract_mode(save_path)
        mri_path = f'../reports/saliency_mri_{MODE}_{mri_index_now}.png'
        plot_mri_with_heatmaps(test_img[mri_index_now], saliency_maps[2][mri_index_now], mri_path)
    

def plot_mri_with_heatmaps(mri_images, heatmaps, save_path):
    """
    Plots MRI images and the same images with normalized heatmaps overlay.

    Args:
    mri_images: (72, 72, 3) numpy array representing the MRI images (3 slices).
    heatmaps: (72, 72, 3) numpy array representing the importance scores (3 heatmaps).
    save_path: Path where the combined figure will be saved.
    """
    if mri_images.shape != (72, 72, 3) or heatmaps.shape != (72, 72, 3):
        raise ValueError("Both mri_images and heatmaps must be of shape (72, 72, 3)")

    plt.figure(figsize=(12, 18))
    slice_names = ["Sagittal", "Axial", "Coronal"]

    for i in range(3):
        # Normalize the heatmap for slice i
        mean = np.mean(heatmaps[:, :, i])
        std = np.std(heatmaps[:, :, i])
        heatmap_normalized = (heatmaps[:, :, i] - mean) / std
        heatmap_normalized = (heatmap_normalized - np.min(heatmap_normalized)) / (np.max(heatmap_normalized) - np.min(heatmap_normalized))

        # Flip Sagittal and Coronal slices and their heatmaps
        if slice_names[i] in ["Sagittal", "Coronal"]:
            mri_slice = np.flipud(mri_images[:, :, i])
            heatmap_slice = np.flipud(heatmap_normalized)
        else:
            mri_slice = mri_images[:, :, i]
            heatmap_slice = heatmap_normalized

        # Plot the original MRI image
        plt.subplot(3, 2, 2*i + 1)
        plt.imshow(mri_slice, cmap='gray')
        plt.title(f"Original {slice_names[i]} Image")
        plt.axis('off')

        # Plot MRI with normalized heatmap overlay
        plt.subplot(3, 2, 2*i + 2)
        plt.imshow(mri_slice, cmap='gray')
        plt.imshow(heatmap_slice, cmap='viridis', alpha=0.5)  # Overlay the normalized heatmap
        plt.title(f"{slice_names[i]} with Heatmap Overlay")
        plt.axis('off')

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def compute_multi_modal_saliency_maps(model, inputs):
    """
    Compute saliency maps for all inputs in a multi-modal model without specifying a class index.

    Args:
    model: The trained multi-modal model.
    inputs: List of inputs corresponding to each modality (clinical, snp, mri).
            These can be numpy arrays or tf.Tensors.

    Returns:
    List of numpy arrays: The computed saliency maps for each modality.
    """
    # Convert inputs to tf.float32 tensors if they are not already
    tensor_inputs = [tf.cast(tf.convert_to_tensor(input_data), tf.float32) for input_data in inputs]

    with tf.GradientTape() as tape:
        # Watch the tensor inputs
        tape.watch(tensor_inputs)
        predictions = model(tensor_inputs)
        # Use the model's output (e.g., the logits or probabilities) for gradient computation
        loss = tf.reduce_sum(predictions)  # Sums up the outputs for gradient calculation

    gradients = tape.gradient(loss, tensor_inputs)

    # Check if gradients are computed successfully
    if any(grad is None for grad in gradients):
        raise ValueError("Gradient computation failed. Check if the model and inputs are compatible.")

    saliency_maps = [tf.abs(grad).numpy() for grad in gradients]
    return saliency_maps


def maximize_multi_modal_activation(model, target_layer, input_shapes, iterations=30, step=1.0):
    """
    Generate inputs for each modality that maximize the activation of a specified layer in a multi-modal model.

    Args:
    model: The trained multi-modal model.
    target_layer: The name of the layer to be maximized.
    input_shapes: List of shapes for each modality input.
    iterations: Number of gradient ascent steps.
    step: Step size for gradient ascent.

    Returns:
    List of numpy arrays: The generated inputs for each modality.
    """
    # Initialize random inputs for each modality
    input_data = [tf.random.uniform((1, *shape), 0, 1) for shape in input_shapes]

    # Retrieve the symbolic output of the target layer
    layer_output = model.get_layer(target_layer).output

    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(input_data)
            model_output = model(input_data)
            # Target the activation of the specific layer
            activation = tf.reduce_mean(model_output[layer_output])

        # Compute gradients for each modality
        grads = tape.gradient(activation, input_data)

        # Update each modality input with normalized gradients
        for i in range(len(input_data)):
            normalized_grads = grads[i] / (tf.sqrt(tf.reduce_mean(tf.square(grads[i]))) + 1e-5)
            input_data[i] += step * normalized_grads

    # Decode the resulting input data for each modality
    maximized_activation_inputs = [data.numpy()[0] for data in input_data]
    return maximized_activation_inputs



if __name__=="__main__":
    tf.config.experimental_run_functions_eagerly(True)

    # Model saving
    current_script_path = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.abspath(os.path.join(current_script_path, os.pardir))
    checkpoints_directory = os.path.join(parent_directory, 'checkpoints')
    os.makedirs(checkpoints_directory, exist_ok=True)
    
    m_a = {}
    seeds = random.sample(range(1, 200), 5)
    for s in seeds:
        MODEL_SAVE_PATH = os.path.join(checkpoints_directory, f'model_SA_{s}.h5')
        acc, bs_, lr_, e_ , seed= train('MM_SA', 32, 50, 0.001, s, MODEL_SAVE_PATH)
        m_a[acc] = ('MM_SA', acc, bs_, lr_, e_, seed)
    print(m_a)
    print ('-'*55)
    max_acc = max(m_a, key=float)
    print("Highest accuracy of: " + str(max_acc) + " with parameters: " + str(m_a[max_acc]))
    
