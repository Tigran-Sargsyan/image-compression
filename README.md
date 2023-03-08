# Image Compression using K-Means Algorithm

This Streamlit application allows users to compress images and uses the K-Means algorithm for it. After uploading an image, the algorithm compresses it while preserving the most important colors. The user can then download the compressed image.

In addition, the application displays the top three dominant colors for every image uploaded. Multiprocessing has been implemented to speed up the computations.

## How it Works
The application uses the K-Means algorithm to compress the images. K-Means is an unsupervised machine learning algorithm that groups data points into K clusters based on their similarity.

In the context of image compression, K-Means groups the pixels in an image based on their color similarity. The algorithm then selects the most representative color for each cluster and assigns that color to all pixels in the cluster.

The number of clusters (K) determines the level of compression. A higher value of K will preserve more colors and result in less compression, while a lower value of K will result in more compression but less color fidelity.

The top three dominant colors for every image are also displayed using the K-Means algorithm.

Multiprocessing has been implemented to speed up the computations. The application divides the image into smaller segments and compresses them in parallel using multiple CPU cores.


Here you can use the streamlit app via this link directly    
https://tigran-sargsyan-streamlit-app-image-compression-au8hiv.streamlit.app/
