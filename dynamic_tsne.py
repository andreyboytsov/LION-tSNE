# Author: Andrey Boytsov <andrey.boytsov@uni.lu> <andrey.m.boytsov@gmail.com>
# License: BSD 3 clause (C) 2017

# DISCLAIMER: while writing the code I used L. van der Maaten's implementation and scikit-learn implementation for
# reference, clarification and for comparing the example results.
# No code was reused from L. van der Maaten's implementation.
# Method _gradient_descent was copied from scikit-learn implementaion of TSNE in compliance with 3-clause BSD licence.

# As of 02 July 2017 those implementations are available here:
# https://lvdmaaten.github.io/tsne/
# https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/manifold

# - A. Boytsov

# References:
# [1] L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning
# Research 9(Nov):2579-2605, 2008.
# [2] G.E. Hinton and S.T. Roweis. Stochastic Neighbor Embedding. In Advances in Neural Information Processing Systems,
# volume 15, pages 833–840, Cambridge, MA, USA, 2002. The MIT Press.

import numpy as np
from scipy.spatial import distance
from scipy import optimize
from scipy import interpolate
from scipy import linalg

# TODO Add some debug logging, put verbose levels
# TODO Put on GitHub with all the requirements, etc.
# TODO Add some tests
# TODO Random seeds for reproducibility
# TODO Different optimization methods for sigma and KL divergence

# TODO Improve code reuse.

# TODO get machine precision instead ?
EPS = 1e-12 # Precision level

# TODO Make P calculations configurable
# TODO Make sigma calculations configurable

# Main tasks:
# TODO 1. Transform
# TODO 2. Import
# TODO 3. Write tests for KL divergence and gradient (use )


# DISCLAIMER: the method below was copied from scikit-learn implementation (available under 3-clause BSD license)
def _gradient_descent(objective, p0, it, n_iter, objective_error=None,
                      n_iter_check=1, n_iter_without_progress=50,
                      momentum=0.5, learning_rate=1000.0, min_gain=0.01,
                      min_grad_norm=1e-7, min_error_diff=1e-7, verbose=0,
                      args=None, kwargs=None):
    """Batch gradient descent with momentum and individual gains.
    Parameters
    ----------
    objective : function or callable
        Should return a tuple of cost and gradient for a given parameter
        vector. When expensive to compute, the cost can optionally
        be None and can be computed every n_iter_check steps using
        the objective_error function.
    p0 : array-like, shape (n_params,)
        Initial parameter vector.
    it : int
        Current number of iterations (this function will be called more than
        once during the optimization).
    n_iter : int
        Maximum number of gradient descent iterations.
    n_iter_check : int
        Number of iterations before evaluating the global error. If the error
        is sufficiently low, we abort the optimization.
    objective_error : function or callable
        Should return a tuple of cost and gradient for a given parameter
        vector.
    n_iter_without_progress : int, optional (default: 30)
        Maximum number of iterations without progress before we abort the
        optimization.
    momentum : float, within (0.0, 1.0), optional (default: 0.5)
        The momentum generates a weight for previous gradients that decays
        exponentially.
    learning_rate : float, optional (default: 1000.0)
        The learning rate should be extremely high for t-SNE! Values in the
        range [100.0, 1000.0] are common.
    min_gain : float, optional (default: 0.01)
        Minimum individual gain for each parameter.
    min_grad_norm : float, optional (default: 1e-7)
        If the gradient norm is below this threshold, the optimization will
        be aborted.
    min_error_diff : float, optional (default: 1e-7)
        If the absolute difference of two successive cost function values
        is below this threshold, the optimization will be aborted.
    verbose : int, optional (default: 0)
        Verbosity level.
    args : sequence
        Arguments to pass to objective function.
    kwargs : dict
        Keyword arguments to pass to objective function.
    Returns
    -------
    p : array, shape (n_params,)
        Optimum parameters.
    error : float
        Optimum.
    i : int
        Last iteration.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = 0

    for i in range(it, n_iter):
        new_error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if (i + 1) % n_iter_check == 0:
            if new_error is None:
                new_error = objective_error(p, *args)
            error_diff = np.abs(new_error - error)
            error = new_error

            if verbose >= 2:
                m = "[t-SNE] Iteration %d: error = %.7f, gradient norm = %.7f"
                print(m % (i + 1, error, grad_norm))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break
            if error_diff <= min_error_diff:
                if verbose >= 2:
                    m = "[t-SNE] Iteration %d: error difference %f. Finished."
                    print(m % (i + 1, error_diff))
                break

        if new_error is not None:
            error = new_error

    return p, error, i
# End of code copied from scikit-learn


def kl_divergence_and_gradient(y, p_matrix, n_embedded_dimensions=2, lock_grad_indices = []):
    """
    :param y: Current NxK matrix of embedded coordinates. K is the reduced dimensionality (2 or 3 usually).
    Array N*K by 1 is also acceptable (scipy.optimize style).
    :param p_matrix: Matrix P for TSNE (distances represented as Gaussian probabilities)
    :param n_embedded_dimensions: Number of dimensions. Necessary only if 1D representation of Y is passed (like in
    scipy.optimize). If Y is 2D, n_embedded_dimensions will be ignored.
    :param lock_grad_indices: For those indices force gradient to zero. Useful if you add new data and should not touch
    old values.
    :return: Tuple of current KL divergence (target function) and its gradient dC/dY (shaped as y was shaped)
    """
    if len(y.shape) > 1:
        n_embedded_dimensions = y.shape[1]
    y_original_shape = y.shape  # To reshape gradient later if necessary
    y = y.reshape((-1, n_embedded_dimensions))

    q_unnorm = distance.squareform(1 / (1 + distance.pdist(y) ** 2))
    # q_matrix = q_unnorm / np.sum(q_unnorm, axis=1).reshape(y.shape[0], 1) #For asymmetric
    q_matrix = q_unnorm / np.sum(q_unnorm)
    q_matrix = np.maximum(q_matrix, EPS)  # Will have easier time dividing. P is zero on diagonals.
    # It is assumed that p_matrix has 1 in diagonals and other 0-values, in order for x*log(x) to be 0
    p_div_q = p_matrix / q_matrix
    p_div_q = np.maximum(p_div_q, EPS)  # To make sure that p * log(p/q) == 0 if p==0 (like in the limit)
    kl_sum = np.sum(p_matrix * np.log(p_div_q))

    kl_grad = np.zeros(y.shape)
    p_minus_q_mul_q_unnorm = (p_matrix - q_matrix) * q_unnorm
    for i in range(p_matrix.shape[0]):
        # Asymmetric gradient (for further reference)
        # kl_grad[i, :] += 2*(P_matrix[i, j] - q_matrix[i, j] + P_matrix[j, i] - q_matrix[j, i])*(y[i, :] - y[j, :])
        # Pij - Qij + Pji - Qji is (P-Q)+(P-Q).T
        # It should be possible to do it in one loop - see the trick below
        # dC/dYi = 2*sum(Pij - Qij + Pji - Qji)*(Yi - Yj)

        # We need to calculate gradient in one loop, or it will be too slow. Now watch closely:
        # dC/dYi = 4*sum(Pij - Qij)*(Yi - Yj)/(1 - ||Yi-Yj||^2)
        # The coefficient of 4 does not matter really. We can put it inside the learning rate, it we want.
        # That's what most implementations do. But we shall keep it.
        # 1 / (1 - ||Yi-Yj||^2) is, actualy, Q matrix element before normalization. Luckily, we saved q_unnorm before.
        # So, (Pij - Qij) / (1 - ||Yi-Yj||^2) is in matrix p_minus_q_mul_q_unnorm (let's call it S for now)
        # dC/dYi = sum over j (Sij * (Yi - Yj) )
        # Y[i, :] - Y : it is a matrix, where each row is Yi - Yj
        # If you multiply i-th row of matrix S by matrix Y[i, :]-Y, you'll exactly get sum over j (Sij * (Yi - Yj))
        kl_grad[i, :] = 4*p_minus_q_mul_q_unnorm[i, :].dot(y[i, :] - y)
    kl_grad[lock_grad_indices, :] = 0
    kl_grad = kl_grad.reshape(y_original_shape)
    return kl_sum, kl_grad


def get_p_and_sigma(distance_matrix, perplexity, starting_sigma=None, method=None, verbose=0, lock_sigmas=[]):
    """"
    Gets Pij matrix and Gaussian sigma for symmetric version of TSNE.
    P is calculated in asymmetric manner (with conditional probabilities), then symmetrized as Pij = (Pi|j + Pj|i)/(2N)
    (for details see original paper by van der Maaten and Hinton, section 3.1, below equation 3).
    :param distance_matrix: NxN distance matrix
    :param perplexity: Required perplexity. Can be a single value or length-N array.
    :param starting_sigma: values of sigma to start with; Nx1 for asymmetric (even if symmetrized afterwards),
           single value for symmetric #TODO: currently - asymmetric only
    :param method: #TODO ignored for now, forced to advanced symmetrized.
    :param verbose: Logging level. 0 (default) - nothing.
    :param lock_sigmas: Do not calculate sigmas at those indices. Leave them at starting sigmas.
    :return: A tuple consisting of:
        P - Gaussian representation of distances
        sigma - single standard deviation for Gaussian representation of distances (required for TSNE)
    """
    num_pts = distance_matrix.shape[0]  # Number of points
    # Condition below should cover both lists and numpy arrays of any kind
    if not hasattr(perplexity, "__len__"):  # If we got a single value, treat is as a list ...
        perplexity = np.array([perplexity]*num_pts)  # ... of same perplexities for each point
    expected_perplexity_log = np.log2(perplexity)
    # For each point log2(perplexity) should be equal to Shannon entropy
    # and entropy of Gaussian depends on its variance. That's how we determine sigma for Gaussian distribution
    if starting_sigma is None:
        starting_sigma = np.ones((num_pts, 1))
    d_prepared = -distance_matrix**2/2.0  # Saving calculations. You have only to exponentiate and divide by sigma^2

    # Now let's go point-by-point and make H(Pi) = log2(perplexity[i])
    def objective(t, num_row):
        # We'd better extract diagonal element rather than set it to zero (and force x*log(x) -> 0 too).
        # Otherwise it will cause some nasty numerical effects, which produce intermittent bugs
        # So, concatenating current row of D without its diagonal element
        current_d = np.concatenate((d_prepared[num_row, :num_row], d_prepared[num_row, num_row+1:]))
        p_row_current = np.exp(current_d / t**2)  # Getting Gaussian probability densities
        # d_prepared already contains minus, squared distances and /2 . Operations left: /sigma**2 and exponent
        p_row_current = np.maximum(p_row_current, EPS) # Zero can appear if we have several exactly same X points
        p_row_current = p_row_current / np.sum(p_row_current)
        shannon_entropy_row = -p_row_current*np.log2(p_row_current)
        shannon_entropy_summed = np.sum(shannon_entropy_row)
        return shannon_entropy_summed-expected_perplexity_log[num_row]

    best_sigma = np.zeros((num_pts, 1))
    p_final = np.zeros((num_pts, num_pts))
    for i in range(num_pts):
        # Locking row number, finding best beta for row
        if i not in lock_sigmas:
            res = optimize.root(lambda t: objective(t, i), starting_sigma[i])  # TODO play with methods
            best_sigma[i, 0] = np.abs(res.x)  # Getting our best value.
            # So there is a chance that -sigma will be found
        else:
            best_sigma[i,0] = starting_sigma[i]
        # And getting P-matrix. This time no extraction.
        p_final_row = np.exp(d_prepared[i, :] / best_sigma[i, 0]**2)  # Getting Gaussian probability densities
        p_final_row[i] = 0
        p_final_row = p_final_row / np.sum(p_final_row)
        p_final[i, :] = p_final_row
    p_final = p_final + p_final.T  # For advanced symmetric. Symmetrize...
    p_final = p_final / np.sum(p_final)  # ... then renormalize
    if verbose >= 2:
        print("Found sigma. Average: ", np.mean(best_sigma))
    return p_final, best_sigma


class DynamicTSNE:

    def __init__(self, perplexity, n_embedded_dimensions=2, distance_function='Euclidean', symmetric=True):
        """
        :param perplexity: TSNE perplexity. Can be roughly thought as the average number of neighbors.
        :param n_embedded_dimensions: Number of dimensions in embedded space. Usually 2.
        :param distance_function: function that measures distance. Default distance - 'Euclidean'. Can accept string
        description or any function that accepts 2 vectors and return value.
        :param symmetric: 
        """
        self.perplexity = perplexity
        self.n_embedded_dimensions = n_embedded_dimensions
        self.symmetric = symmetric #TODO Asymmetric version is not yet implemented
        if distance_function is None:
            self.distance_function = 'Euclidean'
        elif distance_function == 'Euclidean' or callable(distance_function):
            self.distance_function = distance_function
        else:
            raise ValueError("Distance function not recognized: "+str(distance_function))
        self.X = None  # Just an indication that we did not fit anything yet
        self.Y = None  # Not trained yet
        self.P_matrix = None  # Not calculated yet
        self.sigma = None  # Let's keep it just un case

    def get_distance(self,x1,x2):
        """
        :param x1: K-dimensional sample
        :param x2: Another K-dimensional sample
        :return: Distance according to chosen distance metrics
        """
        if self.distance_function == 'Euclidean':
            x1 = x1.reshape((-1,))
            x2 = x2.reshape((-1,))
            return np.sqrt(np.sum((x1-x2)**2))
        else:
            return self.distance_function(x1, x2)

    def get_distance_matrix(self, x):
        """
        :param x: NxK array of N samples, K dimensions each.
        :return: NxN matrix of distances according to chosen distance metrics
        """
        if self.distance_function == 'Euclidean':
            return distance.squareform(distance.pdist(x)) # distance metrics, now surely NxN
        else:
            # So, we have custom distance function. OK then.
            d = np.zeros((x.shape[0], x.shape[0]))
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    d[i,j] = self.distance_function(x[i,:], x[j,:])
            return d

    def fit(self, x, method='gd_momentum', verbose=0, optimizer_kwargs=None, random_seed=None, from_scratch=False):
        """
        :param x: NxK array. N - number of points, K - original number of dimensions
        :param method: Method for finding minimum KL divergence. Supported methods:
            'gd_momentum' (default) - gradient descent with momentum (see reference [1]).
        TODO For now no other method is supported
        :param verbose: Logging level. 0 - nothing. Higher - more verbose.
        :param optimizer_kwargs: will be passed to the optimizer.
        Selected possible arguments (applicability depends on method):
            early_exaggeration: Increases Pij-s by that factor for first early_exageration_iters iterations. Larger
        value often means larger space between clusters. Set None in order to not use it. Default - 4.
            early_exagegration_iters: see early_exaggeration.
            n_iters: number of gradient descent iterations.  Default - 1000.
        See help of _gradient_descent method for more possible parameters of 'gd_momentum' method and corresponding
        defaults.
        :param random_seed: Random seed. Use for reproducibility.
        :param from_scratch: If there is already some fitted data, should we start over (True), or add this data to
        existing (False)? Default - False TODO: not used at the moment. Treated as True.
        :return: Embedded representation of x.
        """
        # TODO: what if those are additional X? Use another method or field
        if random_seed is not None:
            np.random.seed(random_seed)
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer_kwargs = optimizer_kwargs.copy() # We don't need to change it
        self.X = x
        d = self.get_distance_matrix(self.X)

        self.P_matrix, self.sigma = get_p_and_sigma(d, self.perplexity, verbose=verbose)
        # Making sure that Pij * log(Pij) = 0 if Pij = 0 (just like in lim p->0 p * log(p))

        if method == 'gd_momentum':
            if 'n_iter' not in optimizer_kwargs:
                optimizer_kwargs['n_iter'] = 1000  # Default - 1000 iterations
            # We don't need early_exaggeration parameters to stick around
            early_exaggeration = optimizer_kwargs.pop('early_exaggeration', 4.0)
            early_exaggeration_iters = optimizer_kwargs.pop('early_exaggeration_iters', 100)

            self.Y = np.random.randn(d.shape[0] * self.n_embedded_dimensions)
            it = 0
            if early_exaggeration is not None:
                self.P_matrix = self.P_matrix * early_exaggeration
                optimizer_kwargs_no_iters = optimizer_kwargs.copy()
                optimizer_kwargs_no_iters['n_iter'] = early_exaggeration_iters
                self.Y, final_val, it = _gradient_descent(
                    objective=lambda t: kl_divergence_and_gradient(t, self.P_matrix, self.n_embedded_dimensions),
                    p0=self.Y, it=0, verbose=verbose, **optimizer_kwargs_no_iters)
                self.P_matrix = self.P_matrix / early_exaggeration
                if verbose>=2:
                    print("Early exaggeration is over")

            self.Y, final_val, final_iter = _gradient_descent(
                    objective=lambda t: kl_divergence_and_gradient(t, self.P_matrix, self.n_embedded_dimensions),
                    p0=self.Y, it=it, verbose=verbose, **optimizer_kwargs)
            self.Y = self.Y.reshape((-1, self.n_embedded_dimensions))
        else:
            raise ValueError("Method not recognized: "+str(method))
        return self.Y

    def fit_transform(self, x, method='gd_momentum', verbose=0, optimizer_kwargs=None, random_seed=None,
                      from_scratch=False):
        return self.fit(x, method=method, verbose=verbose, optimizer_kwargs=optimizer_kwargs,
                        random_seed=random_seed, from_scratch=from_scratch)

    def incorporate(self, x, y):
        """
        Takes X and Y without any testing or consideration. Useful for debugging or for using already generated
        visualization.
        :param x: Points in original dimensions.
        :param y: Same points in reduced dimensions.
        
        TODO P-matrix and sigmas are NOT recalculated. It might cause problems on transform.
        """
        # TODO see above.
        self.X = x
        self.Y = y
        self.n_embedded_dimensions = y.shape[1]

    # TODO what if some new X coincide with existing X? Lock it?
    def transform(self, x, y=None, method='gd_momentum', verbose=0, keep_sigmas=True, use_sigmas=None,
                  optimizer_kwargs=None, random_seed=None):
        """
        Transforms the data using existing embedding, but does not save those data for further reference.
        :param x:
        :param y: embeddings to start with. If X is a small update of existing data, it might make sense to start from
        existing embeddings. Acceptable inputs:
            Any 2D array of N by n_embedded_dimensions - exact values of Y to start gradient descent with
            'closest' - start with Y corresponding to closest original X value
            None (default) or 'random' - start at random
        :param method: Method for finding minimum KL divergence. Supported methods:
            'gd_momentum' (default) - gradient descent with momentum (see reference [1]).
        TODO For now no other method is supported
        :param verbose: Logging level. 0 - nothing. Higher - more verbose.
        :param keep_sigmas: Keep variance for old data (hence, new data does not count against number of neighbors).
        Recommended.
        :param use_sigmas: Provide sigmas for Pij matrix instead of trying to calculate. Default - None. Designed
        mainly for debug, just keep default in most cases. You may provide sigmas for all values of for new values only
        (method will figure out depending on size).
        :param optimizer_kwargs: will be passed to the optimizer.
        Selected possible arguments (applicability depends on method):
            early_exaggeration: Increases Pij-s by that factor for first early_exageration_iters iterations. Larger
        value often means larger space between clusters. Set None in order to not use it. Default - 4.
            early_exagegration_iters: see early_exaggeration.
            n_iters: number of gradient descent iterations.  Default - 1000.
        See help of _gradient_descent method for more possible parameters of 'gd_momentum' method and corresponding
        defaults.
        :param random_seed: Random seed. Use for reproducibility.
        :return: Embedded representation of X, using already calculated embeddings.
        """
        if self.X is None:
            raise ValueError("No embedding found. Perhaps, model is not trained.")
        x = np.array(x)
        if random_seed is not None:
            np.random.seed(random_seed)
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer_kwargs = optimizer_kwargs.copy() # We don't need to change it
        # Step 1. Get new P matrix and sigmas.
        # TODO so what shall we do with P and sigma? Recalculate? How.
        # TODO Does it matter? As long as old Y stay, does it matter if old P stay the same? Anyway, P is not going to
        # be updated
        x_merged = np.concatenate((self.X, x), axis=0)
        d_new = self.get_distance_matrix(x_merged)
        # TODO or shall we keep sigmas of existing? Recalculating sigma will keep perplexity. Perplexity is
        # TODO number of neighbors. With new unrelated data number of parents will increase.
        # TODO So, keep old sigmas for old values? Think of it. Preferably, make configurable with good defaults.
        # TODO Imagine transforming the same training data. Expect them to stay. Think from that perspective.
        # TODO If we recalculate ALL sigmas, won't happen.
        # So far decision is just to use interpolation

        # So, we are provided sigmas (I cannot imagine it happening outside debug)
        if use_sigmas is not None:
            use_sigmas = np.array(use_sigmas).reshape((-1,1))  # Reshaping into column
            # Did we provide all sigmas?
            if len(use_sigmas) == (len(self.X) + len(x)):
                # Lock all sigms in place. This way it will just calculate P and not search for any sigmas
                lock_sigmas = list(range(len(use_sigmas)))
                p_new, _ = get_p_and_sigma(d_new, self.perplexity, starting_sigma=use_sigmas,
                                           lock_sigmas=lock_sigmas, verbose=verbose)
            elif len(use_sigmas) == len(x):
                # Only new sigmas provided. What to do with old sigmas?
                # If we are requested to keep them, we keep them
                if keep_sigmas:
                    new_sigmas = np.concatenate((self.sigma, use_sigmas), axis=0)
                    lock_sigmas = list(range(len(new_sigmas)))
                    # Again, lock everything. Do not search, just get P.
                    p_new, _ = get_p_and_sigma(d_new, self.perplexity, starting_sigma=new_sigmas,
                                               lock_sigmas=lock_sigmas, verbose=verbose)
                else:
                    # New sigmas are known, but old ones need to be recalculated.
                    start_sigmas = np.concatenate((np.array([[1.0]] * len(self.X)), use_sigmas), axis=0)
                    lock_sigmas = list(range(len(self.X), len(self.X)+len(use_sigmas)))
                    p_new, _ = get_p_and_sigma(d_new, self.perplexity, starting_sigma=start_sigmas,
                                               lock_sigmas=lock_sigmas, verbose=verbose)

        else:
            if keep_sigmas:
                start_sigmas = np.concatenate((self.sigma, np.array([[1.0]]*len(x))), axis=0)
                lock_sigmas = list(range(len(self.X)))
                p_new, _ = get_p_and_sigma(d_new, self.perplexity, starting_sigma=start_sigmas,
                                           lock_sigmas=lock_sigmas, verbose=verbose)
            else:
                p_new, _ = get_p_and_sigma(d_new, self.perplexity, verbose=verbose)

        # Step 2. Run gradient descent with new P and sigmas
        lock_grad_indices = list(range(len(self.Y))) # All old Y's stay where they are
        if method == 'gd_momentum':
            if 'n_iter' not in optimizer_kwargs:
                optimizer_kwargs['n_iter'] = 1000  # Default - 1000 iterations
            # We don't need early_exaggeration parameters to stick around
            early_exaggeration = optimizer_kwargs.pop('early_exaggeration', 4.0)
            early_exaggeration_iters = optimizer_kwargs.pop('early_exaggeration_iters', 100)

            if y is None or (type(y) == str and y == 'random'): # Type check avoids some warnings
                y_start = np.random.randn(x.shape[0], self.n_embedded_dimensions)
            elif np.array(y).shape == (x.shape[0], self.n_embedded_dimensions):
                y_start = np.array(y).copy()
            elif y == 'closest':
                # We already got distance function. Let's use it
                # Index 0 - number of original X, index 1 - number of transformed X.
                d_considered = d_new[:len(self.X), len(self.X):]
                min_y_index = np.argmin(d_considered, axis=0)
                y_start = self.Y[min_y_index, :]
            else:
                raise ValueError("Could not initialize Y. Method not recognized.")

            new_y = np.concatenate([self.Y, y_start], axis=0)
            new_y = new_y.reshape((-1,))

            it = 0
            if early_exaggeration is not None:
                p_new = p_new * early_exaggeration
                optimizer_kwargs_no_iters = optimizer_kwargs.copy()
                optimizer_kwargs_no_iters['n_iter'] = early_exaggeration_iters
                new_y, final_val, it = _gradient_descent(
                    objective=lambda t:
                        kl_divergence_and_gradient(t, p_new, self.n_embedded_dimensions, lock_grad_indices),
                    p0=new_y, it=0, verbose=verbose, **optimizer_kwargs_no_iters)
                p_new = p_new / early_exaggeration
                if verbose>=2:
                    print("Early exaggeration is over")

            new_y, final_val, final_iter = _gradient_descent(
                    objective=lambda t:
                        kl_divergence_and_gradient(t, p_new, self.n_embedded_dimensions, lock_grad_indices),
                    p0=new_y, it=it, verbose=verbose, **optimizer_kwargs)
            new_y = new_y.reshape((-1, self.n_embedded_dimensions))
            new_y = new_y[-x.shape[0]:, :]  # Keep only new things
        else:
            raise ValueError("Method not recognized: "+str(method))
        return new_y

    def generate_embedding_function(self, embedding_function_type=None, function_kwargs=None):
        """
        Creates embedding function for learned TSNE embedding.
        :param embedding_function_type: Type of interpolator. Supported types:
            'makeshift-lagrange-norm': makeshift interpolator. Works like multidimensional
            version of Lagrange multipliers, but uses distance instead of difference (to account for
            multidimensionality). Distance function is specified in the constructor (default - Euclidean).
            Interpolates each Y dimension separately. If you pass one of the known Xi (used for fitting), it is
            guaranteed to return corresponding Yi. If there are 2 similar fitted X, that correspond to different Ys,
            any Y can be picked. Smooth, if distance function is smooth (looks like, but I need to double-check).

            'weighted-inverse-distance': weighted average of Yi-s, where weights are inversely proportional to distance
            between X and Xi sample. If you pass one of the known Xi (used for fitting), it is guaranteed to return
            corresponding Yi. Function_kwargs can contain 'power' - weights will be proportional to inverse
            distances^power (default - 1.0). Negative power is not recommended.

            'linear': Mutltidimensional piecewise-linear interpolation. Tesselates input
            space into simplices, then interpolates linearly on each simplex. Interpolator is separate for each
            embedded dimension.

            'lagrange-weighted-dimensions': Sum of Lagrange polynomials. For ech dimension of X and each dimension of
            Y it builds Lagrange polynomial, s.t. f(X[i,j]) = W[j]*Y[i] (where i runs over points and j over input
            dimensions). Weights are equal by default, but can be modified by setting function_kwargs['weights'].
            Weights should sum up to one (or they will be renormalized).
            Final interpolation is the sum of interpolators per each dimension of X. So, sum_j(f(X[i,j])) =
            sum(W[j]*Y[i]) = Y[i], as required for interpolation.

            WARNING: This method will fail if there are multiple X points, that have the same values along one of the
            dimensions, but correspond to different Y values (can easily happen for binary attributes or one-hot
            encoding).

            'linear-weighted-dimensions': Sum of linear interpolators. See 'lagrange-weighted-dimensions', it works in a
            similar manner.

            'custom-per-dimension': Sum of custom interpolators. Interpolator class should be in
            function_kwargs['interpolator']. There will be D*K objects of that class, where D is the number of embedded
            dimensions and K is the number of original dimensions. Constructor for each object shoula accept 2 1D arrays
            X and Y. Once constructed, interpolator will be invoked by __call__ function. Scipy 1D interpolators are
            compatible. See 'lagrange-weighted-dimensions' for more details about dimensions and
            weights. Additional keyword arguments to interpolator can be transferred in function_kwargs['interpolator'].

            'rbf' (default, 'default', None): Interpolation based on radial basis functions. Use
            function_kwargs['function'] to specify the function. Default - 'multiquadratic'. Fucntion_kwargs will be
            directly passed to scipy.interpolate.Rbf initialization. See corresponding help for more available options.

        :param function_kwargs: Parameters of embedding function (if applicable).
        :return: Embedding function, which accepts NxK array (K - original number of dimensions). Returns NxD embedded
        array (D is usually 2). Also accepts verbose parameter for logging level.
        """
        #TODO Cache embedders if X did not change?
        if self.X is None:
            raise ValueError("No embedding found. Perhaps, model is not trained.")

        if type(embedding_function_type) == str:
            embedding_function_type = embedding_function_type.lower()

        if embedding_function_type is None or embedding_function_type == 'default':
            embedding_function_type = 'rbf'

        if function_kwargs is None:
            function_kwargs = {}

        if embedding_function_type == 'makeshift-lagrange-norm':
            # TODO duplicate values of X ?
            d = self.get_distance_matrix(self.X)

            # TODO. OK, let's do it with loops and think of vectorized implementation later
            def resulting_embedding_function(x, verbose=0):
                x = np.array(x).reshape((-1,self.X.shape[1]))
                y = np.zeros((len(x), self.n_embedded_dimensions))
                for k in range(x.shape[0]): # For each of k requested points
                    if verbose >= 2:
                        print("makeshift-lagrange-norm: Embedding sample ",k)
                    for i in range(self.X.shape[0]): # Summing over all original Y-s
                        coef = 1.0
                        for j in range(self.X.shape[0]):
                            if i != j:
                                coef *= self.get_distance(x[k, :],self.X[j, :])/d[i, j]
                        y[k, :] += coef * self.Y[i, :]
                return y

            return resulting_embedding_function
        elif embedding_function_type == 'weighted-inverse-distance':
            power = function_kwargs.get('power', 1.0)

            def resulting_embedding_function(x, verbose=0):
                x = np.array(x).reshape((-1,self.X.shape[1]))
                y = np.zeros((len(x), self.n_embedded_dimensions))
                for k in range(x.shape[0]):
                    found = False
                    if verbose >= 2:
                        print("weighted-inverse-distance (power", power, "): Embedding sample ", k)
                    distances = np.zeros(self.X.shape[0])
                    for j in range(len(self.X)):
                        distances[j] = self.get_distance(x[k, :], self.X[j, :])
                        if distances[j] == 0:
                            y[k, :] = self.Y[j, :]
                            if verbose >= 2:
                                print("Exact match on sample", j)
                            found = True
                            break
                    if not found:
                        weights = 1/distances**power
                        weights = weights/np.sum(weights)
                        y[k,:] = weights.dot(self.Y)
                return y
            return resulting_embedding_function
        elif embedding_function_type == 'linear':
            # TODO duplicate values of X ?
            interpolator_list = list()
            for d in range(self.n_embedded_dimensions):
                if self.X.shape[1] == 1:
                    interpolator_list.append(
                        interpolate.interp1d(self.X.reshape((-1, )), self.Y[:, d].reshape((-1, )),kind='linear',
                                             fill_value='extrapolate'))
                else:
                    interpolator_list.append(interpolate.LinearNDInterpolator(self.X, self.Y[:, d]))

            def resulting_embedding_function(x, verbose=0):
                if verbose >= 2:
                    print("linear: Embedding all at once ")
                if self.X.shape[1] == 1:
                    x = np.array(x).reshape((-1, ))
                else:
                    x = np.array(x).reshape((-1, self.X.shape[1]))
                y = np.zeros((len(x), self.n_embedded_dimensions))
                for dim in range(self.n_embedded_dimensions):
                    y[:, dim] = interpolator_list[dim].__call__(x)
                return y

            return resulting_embedding_function
        elif embedding_function_type == 'lagrange-per-dimension':
            # TODO duplicate values of X ?
            dimension_weights = function_kwargs.get('weights', np.array([1/self.X.shape[1]]*self.X.shape[1]))
            dimension_weights = np.array(dimension_weights).reshape((-1,))
            dimension_weights = dimension_weights / np.sum(dimension_weights)
            interpolator_list_per_output_dimensions = list()
            for d in range(self.n_embedded_dimensions):
                this_dimension_interpolator_list = list()
                for k in range(self.X.shape[1]):
                    this_dimension_interpolator_list.append(interpolate.lagrange(self.X[:, k].reshape((-1,)),
                                                                self.Y[:, d].reshape(-1,)*dimension_weights[k]))
                interpolator_list_per_output_dimensions.append(this_dimension_interpolator_list)

            def resulting_embedding_function(x, verbose=0):
                x = np.array(x).reshape((-1, self.X.shape[1]))
                y = np.zeros((len(x), self.n_embedded_dimensions))
                if verbose >= 2:
                    print("lagrange-per-dimension: Embedding all at once")
                for dim in range(self.n_embedded_dimensions):
                    for l in range(self.X.shape[1]):
                        y[:, dim] += interpolator_list_per_output_dimensions[dim][l].__call__(x[:, l].reshape((-1,)))
                return y

            return resulting_embedding_function
        elif embedding_function_type == 'linear-per-dimension':
            # TODO duplicate values of X ?
            dimension_weights = function_kwargs.get('weights', np.array([1 / self.X.shape[1]] * self.X.shape[1]))
            dimension_weights = np.array(dimension_weights).reshape((-1,))
            dimension_weights = dimension_weights / np.sum(dimension_weights)
            interpolator_list_per_output_dimensions = list()
            for d in range(self.n_embedded_dimensions):
                this_dimension_interpolator_list = list()
                for k in range(self.X.shape[1]):
                    this_dimension_interpolator_list.append(interpolate.interp1d(self.X[:, k].reshape((-1,)),
                            self.Y[:, d].reshape(-1,)*dimension_weights[k], kind='linear', fill_value='extrapolate'))
                interpolator_list_per_output_dimensions.append(this_dimension_interpolator_list)

            def resulting_embedding_function(x, verbose=0):
                x = np.array(x).reshape((-1, self.X.shape[1]))
                y = np.zeros((len(x), self.n_embedded_dimensions))
                if verbose >= 2:
                    print("linear-per-dimension: Embedding all at once")
                for dim in range(self.n_embedded_dimensions):
                    for l in range(self.X.shape[1]):
                        y[:, dim] += interpolator_list_per_output_dimensions[dim][l].__call__(x[:, l].reshape((-1,)))
                return y

            return resulting_embedding_function
        elif embedding_function_type == 'custom-per-dimension':
            # TODO duplicate values of X ?
            interpolator_class = function_kwargs['interpolator']
            interpolator_kwargs = function_kwargs['interpolator-kwargs']
            dimension_weights = function_kwargs.get('weights', np.array([1 / self.X.shape[1]] * self.X.shape[1]))
            dimension_weights = np.array(dimension_weights).reshape((-1,))
            dimension_weights = dimension_weights / np.sum(dimension_weights)
            interpolator_list_per_output_dimensions = list()
            for d in range(self.n_embedded_dimensions):
                this_dimension_interpolator_list = list()
                for k in range(self.X.shape[1]):
                    this_dimension_interpolator_list.append(interpolator_class(self.X[:, k].reshape((-1,)),
                                self.Y[:, d].reshape(-1,)*dimension_weights[k], **interpolator_kwargs))
                interpolator_list_per_output_dimensions.append(this_dimension_interpolator_list)

            def resulting_embedding_function(x, verbose=0):
                x = np.array(x).reshape((-1, self.X.shape[1]))
                y = np.zeros((len(x), self.n_embedded_dimensions))
                if verbose >= 2:
                    print("custom-per-dimension: Embedding all at once.")
                for dim in range(self.n_embedded_dimensions):
                    for l in range(self.X.shape[1]):
                        y[:, dim] += interpolator_list_per_output_dimensions[dim][l].__call__(x[:, l].reshape((-1,)))
                return y

            return resulting_embedding_function
        elif embedding_function_type == 'rbf':
            interpolator_list_per_output_dimensions = list()
            for d in range(self.n_embedded_dimensions):
                arglist = list(self.X.T) # Each column separately as X
                arglist.append(self.Y[:,d].reshape((-1,)))  # Column of Y dimension
                interpolator_list_per_output_dimensions.append(interpolate.Rbf(*arglist, **function_kwargs))

            def resulting_embedding_function(x, verbose=0):
                x = np.array(x).reshape((-1, self.X.shape[1]))
                called_arglist = list(x.T)  # Each column separately as X
                y = np.zeros((len(x), self.n_embedded_dimensions))
                if verbose >= 2:
                    print("RBF: Embedding all at once.")
                for dim in range(self.n_embedded_dimensions):
                    y[:, dim] = interpolator_list_per_output_dimensions[dim].__call__(*called_arglist)
                return y

            return resulting_embedding_function
        else:
            raise ValueError("Unknown embedding function type: "+str(embedding_function_type))

