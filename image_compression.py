import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
 
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans

from time import time
from io import BytesIO
from sys import getsizeof

def main():
    st.set_page_config(layout='wide')
    st.title('Image Compression')
    st.write('This tool compresses your image in size, preserving the most important colors\
    using k-Means algorithm.')

    img_file_buffer = st.file_uploader('Upload an image', type=['png','jpg'])

    if img_file_buffer is not None:
        # Dictionary initialization
        compressed_images = {} 
        compressed_bytes = {}   # byte representation of the ndarray images for downloading
        compressed_sizes = {}
        compressed_percents = {}
        
        # Keeping the original image size for calculating the percentage benefit
        initial_size = getsizeof(img_file_buffer)
    
        # Converting an image to numpy array
        bytes_data = img_file_buffer.getvalue()
        image = plt.imread(img_file_buffer)#.astype(np.uint8)
        image_shape = image.shape 

        # Keeping name and type of the image
        file_name = (img_file_buffer.name).split('.')
        name = '.'.join(file_name[:-1])
        image_type = file_name[-1]
        if image_type in ['JPG','jpg','JPEG']:
            image_type = 'jpeg'

        # Shape of the numpy array representing the image 
        st.sidebar.header('Additional Information')
        st.sidebar.write('Image shape: ', image.shape)

        # Displaying the image
        st.write(f'Initial size: {initial_size / 1000} kilobytes.')
        st.image(image, caption='Original', clamp=True)
        
        # Image resizing and pixel normalization
        if image_type == 'jpeg':
            # So that the range of the pixels is [0,1]
            resized_image = image / 255
        elif image_type == 'png':
            # The range is already in [0,1] for png images so we don't need to divide
            resized_image = image

        # 3 dimension -> 2 dimension conversion for kMeans
        resized_image = resized_image.reshape(image.shape[0] * image.shape[1], image.shape[2])

        # Parameter initialization for kMeans
        max_iters = 100

        # Variables for showing the progress
        progress_bar = st.progress(0.0)
        percent_complete = 0
        placeholder = st.empty()

        # Running the algorithm
        start = time()
        start_k = 3
        end_k = 11

        for k in range(start_k, end_k):
            percent_complete = (k - 1) / (end_k - 2) 
            compressed_image, byte_im = compress(resized_image, image_type, image_shape, k, max_iters)
            current_size = getsizeof(byte_im)

            compressed_images[k] = compressed_image
            compressed_bytes[k] = byte_im
            compressed_sizes[k] = current_size / 1000 
            compressed_percents[k] = f'{((current_size - initial_size) * 100) / initial_size:.2f}%'

            progress_bar.progress(percent_complete)
            placeholder.text(f'Progress: {int(percent_complete * 100)}/100')

        end = time()
        st.write(f'The program executed in {end-start:.2f} seconds.')
        st.write(f'Current compression benefit in percents: {compressed_percents.get(end_k - 1)}')

        st.sidebar.write('Compressed size dictionary: ', compressed_percents)

        st.header('Original vs Compressed for different values of k')
        preferred_k = st.slider('Choose the k in kMeans: ', 2, end_k - 1, end_k - 1)

        # Displaying the original and compressed images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Original Image')
            st.image(image, caption='Original', clamp=True)

        with col2:
            st.subheader('Compressed Image')
            st.image(compressed_images[preferred_k], caption='Compressed', clamp=True)

        # Download button for user
        btn = st.download_button(
            label = 'Download the image',
            data = compressed_bytes.get(preferred_k),
            file_name = f'{name}_{preferred_k}.{image_type}',
            mime = f'image/{image_type}'
        )

        sizes = pd.Series(compressed_sizes)

        # Plotting the graph for different values of k
        plot_graph(sizes, initial_size, start_k, end_k)

@st.cache(show_spinner=False)
def compress(resized_image, image_type, initial_image_shape, k, max_iters):
    """Compresses an image using kMeans algorithm

    Args:
        resized_image (ndarray): The resized 2D representation of inital 3D image
        initial_image_shape (tuple): The shape of the inital image for recovery
        k (int): Number of clusters in kMeans
        max_iters: Number of maximum iterations in kMeans

    Returns:
        compressed_image (ndarray): Compressed image of appropriate (initial) size
        byte_im (byteIO): Byte represenation of the compressed image 
    
    """

    kmeans = KMeans(n_clusters=k, max_iter=max_iters).fit(resized_image)
    idx = kmeans.predict(resized_image)
    centroids = kmeans.cluster_centers_

    # Representing the image in terms of indices
    compressed_image = centroids[idx, :] 

    # Reshaping recovered image into proper dimensions and getting back to the proper shape
    compressed_image = compressed_image.reshape(initial_image_shape)
    compressed_image = (compressed_image * 255).astype(np.uint8)
            
    # Converting ndarray to image
    im = Image.fromarray(compressed_image, mode='RGB')
  
    # Converting image to bytes
    buf = BytesIO()
    im.save(buf, format=image_type)
    byte_im = buf.getvalue()
    
    return compressed_image, byte_im

def plot_graph(sizes, initial_size, start_k, end_k):
    """Plots the graph of sizes per different numbers of clusters
    
    Args:
        sizes (float): Sizes of the byte representations of the compressed images
        initial_size (float): The byte size of the original image
        start_k (int): The starting value of clusters for plotting  
        end_k (int): The ending value of cluster for plotting

    Returns:
        None

    """

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
