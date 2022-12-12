import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('seaborn-whitegrid')

from joblib import Parallel, delayed
from time import time
from sklearn.cluster import KMeans
from PIL import Image
from io import BytesIO
from sys import getsizeof

def main():
    st.set_page_config(layout='wide')
    st.header('Image Compression')
    st.write('This tool compresses your image in size, preserving the most important colors\
    using k-Means algorithm.')
    img_file_buffer = st.file_uploader('Upload an image', type=['png','jpg'])

    if img_file_buffer is not None:
        # Initializing dicts for holding initial and post-kMeans bytsizes and images for different k-s
        initial_size = 0
        compressed_sizes = {}
        compressed_percents = {}
        images = {}
        
        initial_size = getsizeof(img_file_buffer)
    
        # Converting an image to numpy array
        image = plt.imread(img_file_buffer)#.astype(np.uint8)
               
        # Shape of the numpy array representing the image 
        st.sidebar.header('Additional Information')
        st.sidebar.write('Array shape: ', image.shape)

        # Displaying the image
        st.write(f'Initial size: {initial_size / 1000} kilobytes.')
        st.image(image, caption='Initial Image', clamp=True)
        
        # Image resizing
        resized_image = image / 255
        print('image:\n', image)
        resized_image = resized_image.reshape(image.shape[0] * image.shape[1], image.shape[2])
        
        # Parameter initialization for kMeans

        #k = st.slider('Choose the k in kMeans',0,30,10)
        max_iters = 100

        # Variables for showing the progress
        progress_bar = st.progress(0.0)
        percent_complete = 0
        placeholder = st.empty()

        # Running the algorithm
        start = time()
        start_k = 2
        end_k = 6

        for k in range(start_k, end_k):
            percent_complete = (k - 1) / (end_k - 2) 
            image_recovered, byte_im = compress(image, k, max_iters, resized_image)
            current_size = getsizeof(byte_im)
            compressed_sizes[k] = current_size
            compressed_percents[k] = f'{((current_size - initial_size) * 100) / initial_size:.2f}%'
            images[k] = image_recovered
            progress_bar.progress(percent_complete)
            placeholder.text(f'Progress: {int(percent_complete * 100)}/100')

        end = time()
        st.write(f'The program executed in {end-start:.2f} seconds.')
        st.write(f'Current compression benefit in percents: {compressed_percents.get(end_k - 1)}')

        st.sidebar.write('Compressed size dictionary: ', compressed_percents)

        # Displaying the original and compressed images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Original Image')
            st.image(image, caption='Original', clamp=True)

        with col2:
            st.subheader('Compressed Image')
            st.image(image_recovered, caption='Compressed', clamp=True)

        file_name = '.'.join((img_file_buffer.name).split('.')[:-1])

        # Download button for user
        btn = st.download_button(
            label = 'Download the image',
            data = byte_im,
            file_name = f'{file_name}_{k}.jpeg',
            mime = f'image/jpeg'
        )

        sizes = pd.Series(compressed_sizes)

        # Plotting the graph for different values of k
        plot_graph(sizes, initial_size, start_k, end_k)

def compress(image, k, max_iters, resized_image):
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=max_iters).fit(resized_image)
    idx = kmeans.predict(resized_image)
    centroids = kmeans.cluster_centers_

    # Representing the image in terms of indices
    image_recovered = centroids[idx, :] 

    # Reshaping recovered image into proper dimensions and getting back to the proper shape
    image_recovered = image_recovered.reshape(image.shape)
    image_recovered = (image_recovered * 255).astype(np.uint8)
            
    # Converting ndarray to image
    print('image_recovered:\n', image_recovered)
    #im_download = image_recovered#.astype(np.uint8)
    im = Image.fromarray(image_recovered, mode='RGB')
    #im = Image.fromarray(im_download, mode='RGB')

    # Converting image to bytes
    buf = BytesIO()
    im.save(buf, format='jpeg')
    #im.save(buf, format='png')
    byte_im = buf.getvalue()
    
    return image_recovered, byte_im

def plot_graph(sizes, initial_size, start_k, end_k):
    fig,ax = plt.subplots(figsize=(4,2))
    ax.plot(sizes)
    ax.axhline(initial_size / 1000, color='red', label='initial size')
    ax.set_xlabel('k in kMeans')
    ax.set_ylabel('kilobytes')
    ax.set_title('Compressed image sizes')
    ax.set_xticks(range(start_k, end_k))

    plt.legend()
    plt.tight_layout()
    
    st.pyplot(fig)

if __name__ == '__main__':
    main()
