import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

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

        # Converting an image to numpy array
        image = plt.imread(img_file_buffer)

        # Converting ndarray to image
        initial_im = Image.fromarray(image,'RGB')

        # Converting image to bytes
        initial_buf = BytesIO()
        initial_im.save(initial_buf, format='png')
        initial_byte_im = initial_buf.getvalue()

        initial_size = getsizeof(initial_byte_im)

        # Shape of the numpy array representing the image 
        st.write('Array shape: ', image.shape)

        # To View Uploaded Image
        st.write('Initial size: ', initial_size)
        st.image(image, caption='Initial Image', clamp=True)

        image = image / 255
        resized_image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
        
        k = st.slider('Choose the k in kMeans',0,20,5)
        max_iters = 1

        # Initializing centroids
        initial_centroids = kMeans_init_centroids(resized_image, k) 
        centroids, idx = run_kMeans(resized_image, initial_centroids, max_iters)

        # Represent image in terms of indices
        image_recovered = centroids[idx, :] 

        # Reshape recovered image into proper dimensions and get back to the proper shape
        image_recovered = image_recovered.reshape(image.shape)
        image_recovered = image_recovered * 255 

        # Converting ndarray to image
        im = Image.fromarray(image_recovered,'RGB')

        # Converting image to bytes
        buf = BytesIO()
        im.save('file.png')
        byte_im = buf.getvalue()

        compressed_size = getsizeof(byte_im)

        st.write('compressed size: ', compressed_size)
        st.image(image_recovered, caption='Compressed Image', clamp=True)

        # Download_button for user
        btn = st.download_button(
            label = "Download Image",
            data = initial_byte_im,
            file_name = f'{k}.png',
            mime = 'image/png'
        )

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): k centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    """

    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    dists = np.zeros(K)
    
    for i in range(X.shape[0]):
        for j in range(K):
            dists[j] = np.linalg.norm(X[i]-centroids[j]) ** 2  
        idx[i] = np.argmin(dists)
    
    return idx

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    m, n = X.shape
    
    centroids = np.zeros((K, n))
  
    for i in range(K):
        C_k = idx == i
        centroids[i] = np.mean(X[C_k], axis = 0)
    
    return centroids

def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    
    # Randomly reordering the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Taking the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids

def run_kMeans(X, initial_centroids, max_iters = 10, plot_progress = False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initializing values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    
    # Running K-Means
    for i in range(max_iters):
        
        # Output progress
        print("K-Means iteration %d/%d" % (i + 1, max_iters))
        
        # For each example in X, assigning it to the closest centroid
        idx = find_closest_centroids(X, centroids)
            
        # Given the memberships, computing new centroids
        centroids = compute_centroids(X, idx, K)

    return centroids, idx

if __name__ == '__main__':
    main()