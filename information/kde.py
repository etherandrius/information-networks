import keras.backend as K
import information.NaftaliTishby as nTishby
from utils import add_noise, noise
from tqdm import tqdm

import numpy as np


def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = K.expand_dims(K.sum(K.square(X), axis=1), 1)
    dists = x2 + K.transpose(x2) - 2*K.dot(X, K.transpose(X))
    return dists


def get_shape(x):
    dims = K.cast(K.shape(x)[1], K.floatx())
    N = K.cast(K.shape(x)[0], K.floatx())
    return dims, N


def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance
    # matrix var * I see Kolchinsky and Tracey, Estimating Mixture Entropy with
    # Pairwise Distances, Entropy, 2017. Section 4.  and Kolchinsky and Tracey,
    # Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = get_shape(x)
    dists = Kget_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims/2.0)*K.log(2*np.pi*var)
    lprobs = K.logsumexp(-dists2, axis=1) - K.log(N) - normconst
    h = -K.mean(lprobs)
    return dims/2 + h


def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with
    # covariance matrix var * I see Kolchinsky and Tracey, Estimating Mixture
    # Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    dims, N = get_shape(x)
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2


def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)


def calculate_information_saxe(input_values, labels, bins=-1):

    data_x = input_values
    data_y = labels

    noise_variance = 0.05
    Y_samples = K.placeholder(ndim=2)
    entropy_func_upper = K.function([Y_samples,], [entropy_estimator_kl(Y_samples, noise_variance),])

    data_y_no_one_hot = np.array([np.where(r==1)[0][0] for r in data_y])
    saved_labelixs = {}
    categories = len(data_y[0])
    for i in range(categories):
        saved_labelixs[i] = data_y_no_one_hot == i

    labelprobs = np.mean(data_y, axis=0)

    #data_x = add_noise(input_values, noise)
    #data_y = add_noise(labels, noise)

    nats2bits = 1.0 / np.log(2)

    print("Setup complete - for information function")

    def information(activation):
        # data_t = [add_noise(a, noise) for a in activation]
        data_t = activation  
        # don't think need to add noise to activations as they are produced
        # randomly by the neural network, adding noise only prevents entropy
        # calculations from failing in situations when 5 points have the exact
        # same values, then a division by zero is possible.

        print("Binning...")
        if bins > 0:
            data_t = [add_noise(np.asarray(nTishby.bin_array(t, bins=bins, low=t.min(), high=t.max())), noise) for t in tqdm(data_t)]
        print("Binning done")

        def info(t):
            h_t = entropy_func_upper([t,])[0]
            h_t_given_x = kde_condentropy(t, noise_variance)
            h_t_given_y = 0
            for i in range(categories):
                h = entropy_func_upper([t[saved_labelixs[i],:],])[0]
                h_t_given_y += labelprobs[i] * h
            h_t_given_y
            return nats2bits * (h_t - h_t_given_x), nats2bits * (h_t - h_t_given_y)

        print("Calculating I")
        I = [(*info(t), 0) for t in tqdm(data_t)]
        print("Calculating I done")
        return np.array(list(zip(*I)))


        #h_t = np.array([entropy_func_upper([t,])[0] for t in data_t])
        #h_t_given_x = np.array([kde_condentropy(t, noise_variance) for t in data_t])



        #return i_x_t, i_y_t, [0]

    return information

