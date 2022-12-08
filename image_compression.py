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
        # Initializing dicts for holding initial and post-kMeans bytsizes
        initial_size = 0
        compressed_size = {}
        
        initial_size = getsizeof(img_file_buffer)
    
        # Converting an image to numpy array
        image = plt.imread(img_file_buffer).astype(np.uint8)
               
        # Shape of the numpy array representing the image 
        st.sidebar.header('Additional Information')
        st.sidebar.write('Array shape: ', image.shape)

        # Displaying the image
        st.write('Initial size: ', initial_size)
        st.image(image, caption='Initial Image', clamp=True)
        
        # Image resizing
        image = image / 255
        resized_image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
        
        # Parameter initialization for kMeans

        #k = st.slider('Choose the k in kMeans',0,30,10)
        max_iters = 15

        # Variables for showing the progress
        progress_bar = st.progress(0.0)
        percent_complete = 0

        # Running the algorithm
        start = time()
        start_k = 2
        end_k = 12

        for k in range(start_k, end_k):
            percent_complete = (k - 1) / (12 - 2) 
            print(percent_complete)
            image_recovered, byte_im = compress(image, k, max_iters, resized_image)
            compressed_size[k] = getsizeof(byte_im) / 1000
            progress_bar.progress(percent_complete)

        end = time()
        st.write(f'The program executed in {end-start:.2f} seconds.')
        st.write('Compressed size: ', compressed_size.get(end_k - 1))

        # Displaying the compressed image
        st.sidebar.write('Compressed size dictionary: ', compressed_size)
        st.image(image_recovered, caption='Compressed Image', clamp=True)

        # Download button for user
        btn = st.download_button(
            label = 'Download the image',
            data = byte_im,
            file_name = f'{k}.jpeg',
            mime = f'image/jpeg'
        )

        sizes = pd.Series(compressed_size)

        # Plotting the graph for different values of k
        fig,ax = plt.subplots(figsize=(4,2))
        ax.plot(sizes)
        ax.axhline(initial_size / 1000, color='red', label='initial size')

        ax.set_xlabel('k in kMeans')
        ax.set_ylabel('kilobytes')
        ax.set_title('Compressed image sizes')
        ax.set_xticks(range(2,11))

        plt.legend()
        plt.tight_layout()
        
        st.pyplot(fig)

def compress(image, k, max_iters, resized_image):
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=max_iters).fit(resized_image)
    idx = kmeans.predict(resized_image)
    centroids = kmeans.cluster_centers_

    # Representing the image in terms of indices
    image_recovered = centroids[idx, :] 

    # Reshaping recovered image into proper dimensions and getting back to the proper shape
    image_recovered = image_recovered.reshape(image.shape)
    image_recovered = (image_recovered * 255).astype(np.uint8)
    #image_recovered = image_recovered * 255 
            
    # Converting ndarray to image
    im = Image.fromarray(image_recovered, mode='RGB')

    # Converting image to bytes
    buf = BytesIO()
    im.save(buf, format='jpeg')
    byte_im = buf.getvalue()
        
    return image_recovered, byte_im

if __name__ == '__main__':
    main()
