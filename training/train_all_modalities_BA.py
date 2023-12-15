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
import json
import time







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

    # Only for OVO Attention
    if return_attention_weights:
        return model, [attn_weights_1, attn_weights_2, attn_weights_3]
        
    return model



def train(mode, batch_size, epochs, learning_rate, seed, save_path):
    
    
    train_clinical = pickle.load(open("../ADNI/X_train_clinical.pkl", 'rb')).values
    test_clinical= pickle.load(open("../ADNI/X_test_clinical.pkl", 'rb')).values

    clinical_column_names = pickle.load(open("../ADNI/X_test_clinical.pkl", 'rb')).columns
    
    train_snp = pickle.load(open("../ADNI/X_train_snp.pkl", 'rb')).values
    test_snp = pickle.load(open("../ADNI/X_test_snp.pkl", 'rb')).values

    snp_column_names = pickle.load(open("../ADNI/X_test_snp.pkl", 'rb')).columns

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

    # Record the start time
    start_time = time.time()
    
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

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time in seconds
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")

    #model.load_weights(save_path)
    score = model.evaluate([test_clinical, test_snp, test_img], test_label)
    
    # SALIENCY MAPS
    saliency_maps = compute_multi_modal_saliency_maps(model, [test_clinical, test_snp, test_img])
    ####visualize_some_saliency(test_img, saliency_maps, save_path)

    ranking_snps = rank_snps_by_importance(saliency_maps[1])
    save_top_snps(ranking_snps, snp_column_names, save_path)

    # pdb.set_trace()
    
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


def save_top_snps(snp_ranking, snp_col_names, model_save_path):
    extract_mode = lambda s: re.search(r'model_(.*?)\.h5', s).group(1)
    MODE = extract_mode(model_save_path)
    snp_path = f'../reports/saliency_snp_{MODE}.json'
    top10_snps = snp_ranking[:10]
    top10_snps = [snp_col_names[i] for i in top10_snps]
    with open(snp_path, 'w') as file: json.dump(top10_snps, file)
    print(f'Saved Top SNPs to {snp_path}')
    return

def save_top_clinical(clinical_ranking, clinical_col_names, model_save_path):
    extract_mode = lambda s: re.search(r'model_(.*?)\.h5', s).group(1)
    MODE = extract_mode(model_save_path)
    clinical_path = f'../reports/saliency_clinical_{MODE}.json'
    top10_clinical = clinical_ranking[:10]
    top10_clinical = [clinical_col_names[i] for i in top10_clinical]
    with open(clinical_path, 'w') as file: json.dump(top10_clinical, file)
    print(f'Saved Top Clinical Features to {clinical_path}')
    return



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


def rank_snps_by_importance(saliency_scores):
    """
    Rank SNPs based on their importance using mean reciprocal rank.

    NOTE: we can use this same function for the clinical data, too!

    Args:
    saliency_scores (numpy.ndarray): A 2D array of shape (num_inputs, num_snps) containing saliency scores.

    Returns:
    numpy.ndarray: A 1D array of SNP indices ranked by their importance.
    """
    num_inputs, num_snps = saliency_scores.shape

    # Initialize a matrix to hold ranks for each input
    ranks = np.zeros_like(saliency_scores, dtype=float)

    # Loop through each input and rank the SNPs
    for i in range(num_inputs):
        # argsort twice to get rank positions; smaller rank means more important
        ranks[i, :] = np.argsort(np.argsort(-saliency_scores[i, :]))

    # Compute mean reciprocal rank for each SNP
    mrr = np.mean(1.0 / (ranks + 1), axis=0)

    # Get indices of SNPs sorted by their MRR (descending order of importance)
    ranked_snps = np.argsort(-mrr)

    return ranked_snps



def maximize_multi_modal_activation(model, target_layer, input_shapes, iterations=30, step=1.0):
    """
    Generate inputs for each modality that maximize the activation of a specified layer
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
        MODEL_SAVE_PATH = os.path.join(checkpoints_directory, f'model_BA_{s}.h5')
        acc, bs_, lr_, e_ , seed= train('MM_BA', 32, 50, 0.001, s, MODEL_SAVE_PATH)
        m_a[acc] = ('MM_BA', acc, bs_, lr_, e_, seed)
    print(m_a)
    print ('-'*55)
    max_acc = max(m_a, key=float)
    print("Highest accuracy of: " + str(max_acc) + " with parameters: " + str(m_a[max_acc]))
    
