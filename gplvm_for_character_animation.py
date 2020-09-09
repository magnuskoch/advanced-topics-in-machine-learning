import numpy as np
import pickle, pyro, sklearn
from sklearn import decomposition
import matplotlib.pyplot as plt
import GPy
from  GPy.models import bayesian_gplvm

fig, [pca_ax, gplvm_ax] = plt.subplots(1,2)

animation_path = 'animation.bin'

# source is 240 fps. Let's get 24
down_sample = 50

with open(animation_path, 'rb') as fh:
    animation_data = pickle.load(fh)[::down_sample,:]
    mu = np.mean(animation_data)
    sigma = np.std(animation_data)
    animation_data = (animation_data - mu) / sigma

frame_count = animation_data.shape[0]
bone_count = animation_data.shape[1]

animation_data = animation_data.reshape([frame_count, -1])

frame_cap = 1000

if frame_cap > 0:
    animation_data = animation_data[0:frame_cap]
    frame_count = animation_data.shape[0]

def experiment_1(latent_dimension):
    Y = animation_data
    num_inducing = int(frame_count / 100)

    kernel = GPy.kern.RBF(latent_dimension, ARD=False) + GPy.kern.Bias(latent_dimension)

    gplvm = GPy.models.BayesianGPLVM(Y, input_dim=latent_dimension, init='PCA', num_inducing=num_inducing, kernel=kernel)

    max_optimizer_iterations = 3000

    gplvm.optimize('scg', max_iters=max_optimizer_iterations)

    X_latent = gplvm.X
    Y_reconstruction_mu, Y_reconstruction_cov = gplvm.predict(X_latent)

    error = np.linalg.norm(Y - Y_reconstruction_mu, axis=1)
    magnitude = np.linalg.norm(Y, axis=1)

    plt.plot(error, label='reconstruction error')
    plt.plot(magnitude, label='frame magnitude')
    plt.legend()
    plt.show()

def interpolate_latent_x(X_latent, training_indices, latent_dimension, frame_count):
    latent_interpolation = np.empty((frame_count, latent_dimension), dtype=np.float32)

    for index in range(1, len(training_indices)):
        start = training_indices[index-1]
        end = training_indices[index]
        difference = end - start

        start_latent = X_latent[index-1,:]
        end_latent = X_latent[index,:]
        difference_latent = end_latent - start_latent

        for step in range(0, difference + 1):
            t = step / difference
            interpolation = start_latent + difference_latent * t
            latent_interpolation[start+step,:] = interpolation
    return latent_interpolation

def experiment_2(latent_dimension):
    dilation = 50
    training_count = int(frame_count / dilation)
    training_indices = np.linspace(0, frame_count-1, training_count, dtype=np.int16)

    animation_training_data = np.take(animation_data, training_indices, axis=0)

    # define and train gplvm
    Y = animation_training_data
    num_inducing = int(training_count / 1)
    kernel = GPy.kern.RBF(latent_dimension, ARD=False) + GPy.kern.Bias(latent_dimension)

    gplvm = GPy.models.BayesianGPLVM(Y, input_dim=latent_dimension, init='PCA', num_inducing=num_inducing, kernel=kernel)

    max_optimizer_iterations = 1000

    gplvm.optimize('scg', max_iters=max_optimizer_iterations)

    def plot_train_reconstruction():
        X_latent = gplvm.X
        Y_reconstruction_mu, Y_reconstruction_cov = gplvm.predict(X_latent.mean)

        error = np.linalg.norm(Y - Y_reconstruction_mu, axis=1)
        magnitude = np.linalg.norm(Y, axis=1)

        plt.plot(error, label="reconstruction error")
        plt.plot(magnitude, label='Y magnitude')
        plt.legend()
        plt.show()
        assert True

    def plot_interpolation_reconstruction():
        X_latent = gplvm.X
        Y_reconstruction_mu, Y_reconstruction_cov = gplvm.predict(X_latent.mean)

        X_latent_full_range_interpolation = interpolate_latent_x(X_latent.mean, training_indices, latent_dimension, frame_count)
        Y_reconstruction_full_range_mu, Y_reconstruction_full_range_cov = gplvm.predict(X_latent_full_range_interpolation)

        error = np.linalg.norm(animation_data - Y_reconstruction_full_range_mu, axis=1)
        magnitude = np.linalg.norm(animation_data, axis=1)

        plt.scatter(training_indices, np.take(error, training_indices), label="training indices", c='red')
        plt.plot(error, label="reconstruction error")
        plt.plot(magnitude, label='Y magnitude')
        plt.legend()
        plt.show()
        assert True

    plot_train_reconstruction()
    plot_interpolation_reconstruction()

    print('done')
    
experiment_1(3)
experiment_2(3)
experiment_1(20)

