# Author: Andrey Boytsov <andrey.boytsov@uni.lu> <andrey.m.boytsov@gmail.com>
# License: BSD 3 clause (C) 2017

# As of 02 July 2017 those implementations are available here:
# https://lvdmaaten.github.io/tsne/
# https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/manifold

# - A. Boytsov

# References:
# [1] L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning
# Research 9(Nov):2579-2605, 2008.
# [2] G.E. Hinton and S.T. Roweis. Stochastic Neighbor Embedding. In Advances in Neural Information Processing Systems,
# volume 15, pages 833-840, Cambridge, MA, USA, 2002. The MIT Press.

import numpy as np

# TODO get machine precision instead ?
EPS = 1e-12  # Precision level

def product(*args):
    values = map(tuple, args)
    res = [[]]
    for v in values:
        res = [x+[y] for x in res for y in v]
    final_res = list()
    for prod in res:
        final_res.append(tuple(prod))
    return final_res


class LionTSNELightweight:
    """
    Lightweight version of LION-tSNE class. Effectively, LION part, tSNE results .
    """

    def __init__(self, x, y):
        """
        Takes X and Y without any testing or consideration. Useful for debugging or for using already generated
        visualization.
        :param x: Points in original dimensions.
        :param y: Same points in reduced dimensions.
        :param verbose: Logging level. 0 - nothing. Higher - more verbose.
        """
        self.X = x
        self.Y = y
        self.n_embedded_dimensions = y.shape[1]

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
        d = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                d[i,j] = np.sqrt(np.sum((x[i,:] - x[j,:])**2))
        return d

    def generate_lion_tsne_embedder(self, function_kwargs={}, random_state=None, verbose=0,
                                    return_intermediate=False):
        '''
        Method mainly focused on local IDW interpolation and special outlier
        placement procedure.

        TODO: The outlier placement is suboptimal for 3 or more dimensional Y (it places them in a "spiral" on the
        plane, rather than in a grid). Still method is usable.

        TODO: For now only IDW is used as a local interpolation method.

        TODO: Without loss of generality we assume that all input X are distinct (if not - leave only 1 of those x
        and assume that corresponding y is returned for all of them).

        Described in details: <add ref later>

        :param function_kwargs: parameters of the algorithm. Accepted ones:
            'radius_x' : radius in X space for search of nearest neighbors. Ignored if 'radius_x_percentile' is set.

            'power' : power parameter of IDW distribution. Default - 1 (if IDW is used)

            'radius_x_percentile' : percentile of nearest neighbors distribution in X, which is used to set radius
            in search for nearest neighbors. Suppresses 'radius_x'. Accepts value in percents (i.e. for 99% enter 99,
            not 0.99). Default - 99 (if radius_x not set).

            'radius_y' : distance in Y space to indicate outliers. Use if you don't want radius in terms of percentiles.
            Ignored if 'radius_y_percentile' is set.
            NOTE: 'y_safety_margin' will be added to the radius anyway, and it has non-zero default. Multiplication
            coefficient will be applied also.

            'radius_y_percentile' : percentile of nearest neighbors distribution in Y, which is used to set distance
            to embed outliers. Suppresses 'radius_y'. Accepts value in percents (i.e. for 99% enter 99,
            not 0.99). Default - 100 (if radius_y not set), i.e. radius_y = maximum nearest neighbors distance in y
            (before multiplication by radius_y_coefficient and before applying safety margin).

            'y_safety_margin': safety margin for outlier placement in Y space. If lots of similar outliers clump up,
            the closest distance can get lower than radius_y, safety margin can prevent it for a while.
            Default - equal to radius_y_close. Added to radius_y, whether radius_y is given or calculated from
            percentile. Applied after multiplication coefficient.

            'radius_y_close' : if an algorithm requires to "place y close to y_i", it will be placed at a random angle
            at a distance not exceeding radius_y_close. Suppressed by 'radius_y_close_percentile'

            'radius_y_close_percentile' : if an algorithm requires to "place y close to y_i", it will be placed at a
            random angle at a distance not exceeding radius_y_close. Radius_y_close can be set at a percentile of
            nearest neighbor distance in Y. Suppressew 'radius_y_close'. Default - 10 (if 'radius_y_close' not set)

            'radius_y_coefficient' : default - 1.0. Radius_y will be mutiplied by it (before adding safety margin).

            'outlier_placement_method' : Method of placing outliers.
                'cell-based' (default, None) - see article. Splits area between y_min to y_max into 2*r_y sized cells,
                then places outliers at the center of each free cell.
                'circular' : encircles data, adds r_y to it, then finds outlier positions outside that circle with
                proper angular spacing. If ran full cirle, add one more r_y to radius and continues. So, outliers are
                placed on smth like a spiral at the center of the data.

        :param return_intermediate: along with embedding function it will return state of a low of intermediate
            variables. Can be useful for debugging or plotting. Default - false.

        :param random_state: random seed. Default - None.

        :param verbose: Logging level. Default - 0 (log nothing).

        :return: Embedding function, which accepts NxK array (K - original number of dimensions). Returns NxD embedded
        array (D is usually 2). Also accepts verbose parameter for logging level.
        '''
        # TODO Save distance matrix on demand? Can make things faster in many cases, but takes memory.
        # TODO Save P matrix also on demand? It is needed even less, it seems.
        distance_matrix = self.get_distance_matrix(self.X)
        np.fill_diagonal(distance_matrix, np.inf)  # We are not interested in distance to itself
        nn_x_distance = np.min(distance_matrix, axis=1)  # Any axis will do
        outlier_placement_method = function_kwargs.get('outlier_placement_method', None)
        if outlier_placement_method is None:
            outlier_placement_method = 'cell-based'
        outlier_placement_method = outlier_placement_method.lower()

        # TODO Step 1. Extra
        if 'radius_x' in function_kwargs and 'radius_x_percentile' not in function_kwargs:
            radius_x = function_kwargs['radius_x']
        else:
            radius_x_percentile = function_kwargs.get('radius_x_percentile', 99)
            if verbose >= 2:
                print("Setting radius_x at a percentile: ", radius_x_percentile)
            radius_x = np.percentile(nn_x_distance, radius_x_percentile)

        # Some potentially shared calculations for radius_y and radius_y_close
        if 'radius_y_percentile' in function_kwargs or 'radius_y' not in function_kwargs or \
                        'radius_y_close_percentile' in function_kwargs or 'radius_y_close' not in function_kwargs:
            # In that case we will need those things
            y_distance_matrix = self.get_distance_matrix(self.Y)
            np.fill_diagonal(y_distance_matrix, np.inf)  # We are not interested in distance to itself
            nn_y_distance = np.min(y_distance_matrix, axis=1)  # Any axis will do

        if 'radius_y' in function_kwargs and 'radius_y_percentile' not in function_kwargs:
            radius_y = function_kwargs['radius_y']
        else:
            radius_y_percentile = function_kwargs.get('radius_y_percentile', 100)
            radius_y = np.percentile(nn_y_distance, radius_y_percentile)
            if verbose >= 2:
                print("Set radius_y at a percentile: ", radius_y_percentile, "Value: ", radius_y)

        if 'radius_y_close' in function_kwargs and 'radius_y_close_percentile' not in function_kwargs:
            radius_y_close = function_kwargs['radius_y_close']
        else:
            radius_y_close_percentile = function_kwargs.get('radius_y_close_percentile', 10)
            if verbose >= 2:
                print("Setting radius_y_close at a percentile: ", radius_y_close_percentile)
            radius_y_close = np.percentile(nn_y_distance, radius_y_close_percentile)

        if outlier_placement_method == 'circular':
            y_center = np.mean(self.Y, axis=0)
            y_data_radius = np.max(np.sqrt(np.sum((self.Y - y_center) ** 2, axis=1)))  # For outlier placement

        radius_y_coef = function_kwargs.get('radius_y_coefficient', 1.0)
        y_safety_margin = function_kwargs.get('y_safety_margin', radius_y_close)
        radius_y *= radius_y_coef
        radius_y += y_safety_margin

        if verbose >= 2:
            print("Radius_x: ", radius_x)
            print("Radius_y_coef: ", radius_y_coef)
            print("Safety Y margin: ", y_safety_margin)
            print("Radius_y_close: ", radius_y_close)
            print("Radius_y (final): ", radius_y)

        power = function_kwargs.get('power', 1.0)

        if outlier_placement_method == 'cell-based':
            available_cells = list()  # Contain number of cells counting on each axis
            if verbose >= 2:
                print('Generating original set of cells.')
            y_min = np.min(self.Y, axis=0).reshape(-1)  # Let's precompute
            y_max = np.max(self.Y, axis=0).reshape(-1)  # Let's precompute
            if verbose >= 2:
                print('Minimums: ', y_min)
                print('Maximums: ', y_max)
            # Number of cells per dimension.
            original_cell_nums = [int(np.floor((y_max[i] - y_min[i]) / (2 * radius_y))) for i in range(self.Y.shape[1])]
            if verbose >= 2:
                print('Cell nums: ', original_cell_nums)
            # Within y_min to y_max cell can be slighlty larger to divide exactly
            adjusted_cell_sizes = [(y_max[i] - y_min[i]) / original_cell_nums[i] for i in range(self.Y.shape[1])]
            if verbose >= 2:
                print('Adjusted cell sizes: ', adjusted_cell_sizes)
            # How many outer layers did we have to add. For now - none.
            # added_outer_layers = 0 #We do it locally, cause runs are independent
            cell_list = list(product(*[list(range(i)) for i in original_cell_nums]))
            if verbose >= 3:
                print('Cell list: ', cell_list)
            for cell in cell_list:
                if verbose >= 3:
                    print('Checking cell: ', cell)
                # Bounds for each dimension
                cell_bounds_min = [y_min[i] + cell[i] * adjusted_cell_sizes[i] for i in range(self.Y.shape[1])]
                cell_bounds_max = [y_min[i] + (cell[i] + 1) * adjusted_cell_sizes[i] for i in range(self.Y.shape[1])]
                samples_in_cell = np.array([True] * self.Y.shape[0])  #
                for i in range(len(cell_bounds_min)):
                    samples_in_cell = samples_in_cell & \
                                      (cell_bounds_min[i] <= self.Y[:, i]) & (cell_bounds_max[i] >= self.Y[:, i])
                if not samples_in_cell.any():
                    if verbose >= 3:
                        print('No samples in cell: ', cell)
                    available_cells.append(cell)
            del cell_list  # Unnecessary place, we don't need it any longer

        def result_function(x, verbose=0):
            x = np.array(x).reshape((-1, self.X.shape[1]))
            if random_state is not None:
                np.random.seed(random_state)
            # Step 1. Find neighbors in the radius.
            y_result = np.zeros((x.shape[0], self.Y.shape[1]))
            outlier_samples = list()
            local_interpolation_samples = list()
            single_neighbor_samples = list()
            neighbor_indices = list()
            local_interpolation_distances = dict()
            if verbose >= 2:
                print("Radius_x: ", radius_x)
                print("Radius_y: ", radius_y)
                print("Safety Y margin: ", y_safety_margin)
                print("Radius_y_close: ", radius_y_close)
                print("Classifying samples...")
            for i in range(x.shape[0]):
                all_distances = np.sqrt(np.sum((self.X - x[i, :]) ** 2, axis=1))
                exact_matches = np.where(all_distances == 0)[0]
                if len(exact_matches) > 0:
                    if verbose >= 2:
                        print("Sample", i, " - exact match in training set")
                    y_result[i, :] = self.Y[exact_matches[0], :]
                else:
                    cur_neighbor_indices = np.where(all_distances <= radius_x)[0]
                    neighbor_indices.append(cur_neighbor_indices)
                    num_neighbors = len(neighbor_indices[i])
                    if num_neighbors == 0:
                        if verbose >= 2:
                            print("Sample", i, " - outlier")
                        outlier_samples.append(i)
                    elif num_neighbors == 1:
                        if verbose >= 2:
                            print("Sample", i, " - single neighbor")
                        single_neighbor_samples.append(i)
                    else:
                        if verbose >= 2:
                            print("Sample", i, " - ", num_neighbors, "neighbors")
                        local_interpolation_samples.append(i)
                        local_interpolation_distances[i] = all_distances[
                            cur_neighbor_indices]  # Making sure order match

            # First perform local interpolation. It can increase data radius (highly unlikely, though), so watch it.
            if outlier_placement_method == 'circular':
                current_y_data_radius = y_data_radius
            # We already tested for exact match. Can safely assume it is not the case.
            for i in local_interpolation_samples:
                if verbose >= 2:
                    print("Local IDW (power", power, "): Embedding sample ", i)
                weights = 1 / local_interpolation_distances[i] ** power
                weights = weights / np.sum(weights)
                cur_y_result = weights.dot(self.Y[neighbor_indices[i], :])
                if outlier_placement_method == 'circular':
                    current_y_data_radius = max(y_data_radius, np.sqrt(np.sum((cur_y_result - y_center) ** 2)))
                y_result[i, :] = cur_y_result

            # Second take single-neighbor points and either "place them close" or move them to outliers bucket.
            for i in single_neighbor_samples:
                if verbose >= 2:
                    print("Single-neighbor samples: Embedding sample ", i)
                single_neighbor_index = neighbor_indices[i][0]
                single_neighbor_nn_dist = nn_x_distance[single_neighbor_index]
                if single_neighbor_nn_dist <= radius_x:
                    if verbose >= 2:
                        print("Close to cluster, but not close enough. Treating as outlier")
                    outlier_samples.append(i)
                else:
                    if verbose >= 2:
                        print("Close to other outlier. Moving them together.")
                    random_dist = np.random.uniform(low=0, high=radius_y_close)
                    random_angle = np.random.uniform(low=0, high=2 * np.pi)
                    y_result[i, :] = self.Y[single_neighbor_index, :]
                    y_result[i, 0] += random_dist * np.cos(random_angle)  # Considering 0 is X. DOes not matter, really
                    y_result[i, 1] += random_dist * np.sin(random_angle)
                    # TODO: What if Y has more dimensions? More angles, similar procedures. Anyway,this procedure works.
                    # Chance to have exactly the same random values around the same nearest neighbor? Negligible

            # Third, incorporate outliers.
            # 3.1. See if any outliers are close together. If yes, pick one "representative"
            outlier_representatives = list(outlier_samples)  # Later - group them together
            outliers_close_to_representatives = {i: list() for i in outlier_representatives}
            # TODO Group outliers.


            if outlier_placement_method == 'circular':
                # 3.2. Circular - Put those representative on a spiral around the data.
                outliers_angular_distance = 2 * np.arcsin(radius_y / (2 * (current_y_data_radius + radius_y)))
                starting_angle = np.random.uniform(low=0, high=2 * np.pi)
                current_angle = starting_angle
                current_y_radius = current_y_data_radius + radius_y  # Safety margin already included
                outliers_angular_distance = 2 * np.arcsin(radius_y / (2 * (current_y_data_radius + radius_y)))
                for i in outlier_representatives:
                    if verbose >= 2:
                        print("Circular: embedding outlier representative", i)
                    y_result[i, :] = y_center
                    # Considering 0 is X. Does not matter, really
                    y_result[i, 0] += current_y_radius * np.cos(current_angle)
                    y_result[i, 1] += current_y_radius * np.sin(current_angle)
                    current_angle += outliers_angular_distance
                    if current_angle + outliers_angular_distance > starting_angle + 2 * np.pi:
                        current_angle = current_angle - 2 * np.pi
                        starting_angle = current_angle
                        current_y_radius += radius_y
            else:
                # Remember - each run of embedder is independent. So, reinitializing from scratch.
                current_outer_layers = 0
                current_available_cells = list(available_cells)
                for outlier_num in outlier_representatives:
                    if verbose >= 2:
                        print("Embedding outlier representative", outlier_num)
                    # 3.2. Cell-based - put in a free cell. If no free cells - request new ones.
                    if len(current_available_cells) == 0:
                        current_outer_layers += 1
                        if verbose >= 2:
                            print("Out of cells. Adding layer ", current_outer_layers)
                        # Effectively, we need all combinations, where at least one cell coordinate is -L or n+L
                        # And we have variable number of dimensions (usually 2, though), so no nested loops
                        # Iterate everything and leave only cells with -L or n+l as one of coords? Expected too slow.
                        new_available_cells = list()
                        for i in range(self.Y.shape[1]):
                            # From dimensions before current - taking only old dimensions
                            # From current dimension - taking only -L and n+L
                            prods = [list(np.arange(-(current_outer_layers - 1),
                                                    original_cell_nums[k] + (current_outer_layers - 1))) for k in
                                     range(i)]
                            if verbose >= 3:
                                print("For products (1): ", prods)
                            # Keep in mind that last coordinate in the beginning is original_cell_nums[i]-1
                            prods = prods + [[-current_outer_layers, original_cell_nums[i] - 1 + current_outer_layers]]
                            if verbose >= 3:
                                print("For products (2): ", prods)
                            prods = prods + [list(np.arange(-current_outer_layers,
                                                            original_cell_nums[k] + current_outer_layers))
                                             for k in np.arange(i + 1, self.Y.shape[1])]
                            if verbose >= 3:
                                print("For products (3): ", prods)
                            partial_new_cells = list(product(*prods))
                            if verbose >= 3:
                                print("Partial new cells: ", partial_new_cells)
                            new_available_cells.extend(partial_new_cells)
                        current_available_cells = new_available_cells
                        if verbose >= 2:
                            print("New available cells ", current_available_cells)
                    chosen_cell = np.random.randint(0, len(current_available_cells))
                    cell = current_available_cells.pop(chosen_cell)
                    if verbose >= 2:
                        print("Chosen cell:", cell)
                    # Now finding the center.
                    cell_center = np.zeros((1, self.Y.shape[1]))
                    # between y_min and y_max cell sizes are adjusted, so it is just a little bit tricky
                    for i in range(self.Y.shape[1]):
                        if cell[i] < 0:  # Below 0 or above original size - not adjusted
                            cell_center[0, i] = y_min[i] + cell[
                                                               i] * 2 * radius_y + radius_y  # cell[i] is negative, keep in mind
                        elif cell[i] >= original_cell_nums[
                            i]:  # Keep in mind that it passed through many adjusted cells
                            # Cannot just multiply. But can start from max.
                            cell_center[0, i] = y_max[i] + (cell[i] - original_cell_nums[i]) * 2 * radius_y + radius_y
                        else:
                            cell_center[0, i] = y_min[i] + cell[i] * adjusted_cell_sizes[i] + adjusted_cell_sizes[i] / 2
                    y_result[outlier_num, :] = cell_center

            # 3.3. Put other outliers close to their representatives
            for i in outliers_close_to_representatives:
                for j in outliers_close_to_representatives[i]:
                    if verbose >= 2:
                        print("Embedding outlier", j, "close to its representative", i)
                    y_result[j, :] = y_result[i, :]
                    random_dist = np.random.uniform(low=0, high=radius_y_close)
                    random_angle = np.random.uniform(low=0, high=2 * np.pi)
                    y_result[j, :] = self.Y[single_neighbor_index, :]
                    y_result[j, 0] += random_dist * np.cos(random_angle)  # Considering 0 is X. Does not matter, really
                    y_result[j, 1] += random_dist * np.sin(random_angle)

            return y_result

        if return_intermediate and outlier_placement_method == 'cell-based':
            return result_function, cell_bounds_min, cell_bounds_max, adjusted_cell_sizes, available_cells
        else:
            return result_function