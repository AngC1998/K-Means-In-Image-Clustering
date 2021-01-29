#!/usr/bin/env python
# coding: utf-8

import imageio
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans


def kmeans(X, k, ord=2, tol=1e-6, plot=False):
    """Do k-means clustering!

    O(mnki) cost where i is the number of iterations.

    Parameters
    ----------
    X : (m,n) ndarray
        The data to cluster. Each row is an individual
        entry in the data set (with n features).

    k : int
        How many clusters (we have to choose this).

    ord : float
        The order of the distance norm, e.g., ord=2 is for
        standard Euclidean distance.

    tol : float
        Stopping tolerance. Stop the iteration when
        |centers_old - centers_new| < tol.

    plot : bool
        If true, plot the clusters if the data is two-dimensional
        (meaning n = 2).

    Returns
    -------
    y : (m,) ndarray
         Labels for the clusters, e.g., y[i] = 2 means
         X[i,:] belongs to cluster 2.

    centers : (k,n) ndarray
        The centers of the final clusters.
    """
    m,n = X.shape

    # Get k initial cluster centers by choosing points in X.
    centers = X[np.random.choice(m, k, replace=False)]
    # NOTE: how else could you choose initial centers?

    # Do the iteration: decide which center each row is closest to.
    # At each iteration, we need to figure out the distance between
    # each row of data (m things) and each cluster center (k things)
    # which we can store in an m x k matrix.
    distances = np.empty((m,k))     # distances[i,j] = |X[i] - center[j]|
    diff = np.inf
    while diff > tol:
        # Calculate distances.
        for j in range(k):
            distances[:,j] = la.norm(X - centers[j], axis=1, ord=ord)

        # Choose the smallest distance for each point.
        y = np.argmin(distances, axis=1)

        # Update the centers.
        centers_new = np.empty((k,n))
        for j in range(k):
            centers_new[j] = np.mean(X[y == j], axis=0)

        # Calculate the difference between old and new centers.
        diff = la.norm(centers - centers_new, ord=ord)

        # Update.
        centers = centers_new

    if plot and n == 2:
        plot_clusters(X, y, centers)

    return y, centers


def plot_clusters(X, y, centers):
    xx = np.linspace(X[:,0].min(), X[:,0].max(), 2000)
    ymin, ymax = X[:,1].min(), X[:,1].max()
    k = centers.shape[0]

    for i in range(k):
        # Plot the ith cluster.
        cluster = (y==i)
        plt.plot(X[cluster,0], X[cluster,1], '.', color=f"C{i}", ms=3)

        # Mark the ith cluster center.
        x1, y1 = centers[i]
        plt.plot(x1, y1, 'x', color=f"C{i}", ms=5)

        # Plot linear separation boundaries between clusters.
        for j in range(i+1,k):
            x2, y2 = centers[j]
            yy = (x1 - x2)/(y2 - y1)*(xx - (x1 + x2)/2) + (y1 + y2)/2
            mask = (ymin < yy) & (yy < ymax)
            plt.plot(xx[mask], yy[mask], 'k', lw=.5, alpha=.5)


def example():
    X = np.vstack([np.random.normal(loc=[2, 2], size=(200,2)),
                   np.random.normal(loc=[5, 5], size=(200,2))])
    plt.plot(X[:,0], X[:,1], '.', ms=3)
    plt.plot([2,5], [2,5], 'kx', ms=8)
    plt.title("raw data")
    plt.show()

    y = kmeans(X, k=2, ord=2, plot=True)
    plt.plot([2,5], [2,5], 'kx', ms=8)
    plt.title(r"clustered data, $k = 2$")
    plt.show()


    print("Question: how can k-means go wrong?")

    print("Question 1: what if there are really two clusters,",
          "but we think there are three or more?")
    for k in [3, 4, 5]:
        y = kmeans(X, k=k, plot=True)
        plt.plot([2,5], [2,5], 'kx', ms=8)
        plt.title(rf"clustered data, $k = {k}$")
        plt.show()

    print("Question 2: what if there is a class imbalance?")

    X = np.vstack([np.random.normal(loc=[2, 2], scale=.2, size=(490,2)),
                   np.random.normal(loc=[5, 5], scale=.2, size=( 10,2))])
    plt.plot(X[:,0], X[:,1], '.', ms=3)
    plt.plot([2,5], [2,5], 'kx', ms=8)
    plt.title("raw data")
    plt.show()

    for k in [2, 3]:
        y = kmeans(X, k=k, plot=True)
        plt.plot([2,5], [2,5], 'kx', ms=8)
        plt.title(rf"clustered data, $k = {k}$")
        plt.show()


def cluster_image(im, k, plot=False, saveto=None):
    """Cluster image pixels.

    Parameters
    ----------
    im : (r,c,d) ndarray
        Orignal color image.

    k : int
        Number of clusters.

    plot : bool
        If true, plot original and clustered images.

    saveto : str
        The name of a file to save the new image to.

    Returns
    -------
    imnew : (r,c,d) ndarray
        Clustered color image.

    centers : (k,d) ndarray
        The determined colors.
    """
    # Get the dimensions of the image.
    nrows, ncols, ndim = im.shape
    m = nrows * ncols                   # number of pixels
    n = ndim                            # number of features per pixel
    X = im.reshape((m,n))

    # # Do the clustering algorithm with our function.
    # y, centers = kmeans(X, k)

    # Do the clustering algorithm with Scikit-learn.
    Xclustered = KMeans(n_clusters=k, n_init=5).fit(X)
    centers = Xclustered.cluster_centers_
    y = Xclustered.predict(X)

    # Post-process the cluster centers for image purposes.
    centers = np.clip(centers, 0, 255).astype(np.uint8)

    # Create the new picture.
    Xnew = np.empty_like(X)
    for j in range(k):
        Xnew[y == j] = centers[j]
    imnew = Xnew.reshape(im.shape)

    if plot:
        fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
        ax1.imshow(im)
        ax1.axis("off")
        ax1.set_title("original")
        ax2.imshow(imnew)
        ax2.axis("off")
        ax2.set_title(f"{k} clusters")
        plt.show()

    if saveto:
        imageio.imwrite(saveto, imnew)

    return imnew, centers



def application(infile, k, outfile):
    """Do the whole process for one file."""
    im = imageio.imread(infile) # [::4,::4,:]   # If you want to shrink the original image
    return cluster_image(im, k, plot=True, saveto=outfile)



if __name__ == "__main__":
    # example()
    application("UT_Tower.jpg", 10, "UT_Tower_Three.jpg")
