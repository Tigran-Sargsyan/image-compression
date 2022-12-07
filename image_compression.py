import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from PIL import Image
from io import BytesIO
from sys import getsizeof

def main():
    st.header('Image Compression')
    st.write('This tool compresses your image in size, preserving the most important colors\
    using k-Means algorithm.')
    img_file_buffer = st.file_uploader('Upload an image', type=['png','jpg'])

    if img_file_buffer is not None:
        # Initializing dicts for holding initial and post-kMeans bytsizes
        initial_size, compressed_size = 0,0
        
        initial_size = getsizeof(img_file_buffer)
    
        # Converting an image to numpy array
        image = plt.imread(img_file_buffer).astype(np.uint8)
       
        # Shape of the numpy array representing the image 
        st.write('Array shape: ', image.shape)

        # Displaying the image
        st.write('Initial size: ', initial_size)
        st.image(image, caption='Initial Image', clamp=True)
        
        # Image resizing
        image = image / 255
        resized_image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
        
        # Parameter initialization for kMeans
        k = st.slider('Choose the k in kMeans',0,30,10)
        max_iters = 1

        # Initializing centroids
        #initial_centroids = kMeans_init_centroids(resized_image, k) 
        #centroids, idx = run_kMeans(resized_image, initial_centroids, max_iters)

        # Running the algorithm
        kmeans = KMeans(n_clusters=k, random_state=0, max_iter=max_iters).fit(resized_image)
        idx = kmeans.predict(resized_image)
        centroids = kmeans.cluster_centers_

        # Representing the image in terms of indices
        image_recovered = centroids[idx, :] 

        # Reshape recovered image into proper dimensions and get back to the proper shape
        image_recovered = image_recovered.reshape(image.shape)
        image_recovered = (image_recovered * 255).astype(np.uint8)
        #image_recovered = image_recovered * 255 
        
        # Converting ndarray to image
        im = Image.fromarray(image_recovered, mode='RGB')

        # Converting image to bytes
        buf = BytesIO()
        im.save(buf, format='jpeg')
        byte_im = buf.getvalue()
    
        compressed_size = getsizeof(byte_im)
        
        # Displaying the image
        st.write('compressed size: ', compressed_size)
        st.image(image_recovered, caption='Compressed Image', clamp=True)

        # Download_button for user
        btn = st.download_button(
            label = 'Download the image',
            data = byte_im,
            file_name = f'{k}.jpeg',
            mime = f'image/jpeg'
        )

if __name__ == '__main__':
    main()
