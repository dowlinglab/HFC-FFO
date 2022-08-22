#Bisection Method

#Set Upper Bound and Lower Bound
#Lower Bound: 1e-6
#Upper Bound: (Difference betwwen highest upper limit and lowest lower limit)
#Ask NW to find max parameter bound from R-143a.py init file

# Defining Function
def opt_dist(distance, top_samples, constants, target_num, rand_seed = None, eval = False):
    """
    Calculates the distance between points such that exactly a target number of points are chosen for the next iteration
    
    Parameters:
    -----------
        distance: float, The allowable minimum distance between points
        top_samples: pandas data frame, Collection of top liquid/vapor sampes
        constants: utils.r143a.R143aConstants, contains the infromation for a certain refrigerant
        target_num: int, the number of samples to choose next
        rand_seed: int, the seed number to use: None by default
        eval: bool, Determines whether error is calculated or new_points is returned
    
    Returns:
        error: float, The squared error between the target value and number of new_points
        OR
        new_points: pandas data frame, a pandas data frame containing the number of points to be used 
    """
    #Prints distance if there are less top_samples than the target number
    if len(top_samples) <= target_num:
        print("Trying dist =", distance)
        
    top_samp0 = top_samples
    
    #Change seed if wanted
    if rand_seed != None:
        np.random.seed(rand_seed)
    
    #initalize new_points and discarded_points dfs
    new_points = pd.DataFrame()
    discarded_points = pd.DataFrame(columns=top_samples.columns)
    
    #Loop - While len(top_samples > 0)
    while len(top_samples > 0):
        # Shuffle the pareto points
        top_samples = top_samples.sample(frac=1)
        new_points = new_points.append(top_samples.iloc[[0]])
        # Remove anything within distance
        l1_norm = np.sum(
            np.abs(
                top_samples[list(constants.param_names)].values
                - new_points[list(constants.param_names)].iloc[[-1]].values
            ),
            axis=1,
        )
        points_to_remove = np.where(l1_norm < distance)[0]
        discarded_points = discarded_points.append(
            top_samples.iloc[points_to_remove]
        )
        top_samples.drop(
            index=top_samples.index[points_to_remove], inplace=True
        )
    #Calculate the error between the target number of points and number of new points
    error = (target_num - len(new_points))
    
#     print("Error = ",error)
#     return error
    if eval == True:
        return new_points
    else:
        return error
    
# Implementing Bisection Method
def bisection(lower_bound, upper_bound, error_tol, top_samples, constants, target_num, rand_seed = None): 
    """
    approximates a root of a function bounded by lower_bound and upper_bound to within a tolerance 
    
    Parameters:
    -----------
        lower_bound: float, lower bound of the distance, must be > 0
        upper_bound: float, lower bound of the distance, must be > lower_bound
        error_tol: floar, tolerance of error
        top_samples: pandas data frame, Collection of top liquid/vapor sampes
        constants: utils.r143a.R143aConstants, contains the infromation for a certain refrigerant
        target_num: int, the number of samples to choose next
        rand_seed: int, the seed number to use: None by default
        
    Returns:
    --------
        midpoint: The distance that satisfies the error criteria based on the target number

    """
    assert lower_bound > 0, "Lower bound must be greater than 0"
    assert lower_bound < upper_bound, "Lower bound must be less than the upper bound"
    
    # check if lower_bound and upper_bound bound a root
#     print("Low B", lower_bound)
#     print("High B", upper_bound)
    eval_lower_bound = opt_dist(lower_bound, top_samples, constants, target_num, rand_seed)
#     print("Low Eval",eval_lower_bound )
    eval_upper_bound = opt_dist(upper_bound, top_samples, constants, target_num, rand_seed)
#     print("High Eval",eval_upper_bound )
    if np.sign(eval_lower_bound) == np.sign(eval_upper_bound):
        raise Exception(
         "Increase length of upper bound. Given bounds do not include the root!")
        
    # get midpoint
    midpoint = (lower_bound + upper_bound)/2
    eval_midpoint = opt_dist(midpoint, top_samples, constants, target_num, rand_seed)
#     print("Mid", eval_midpoint)
    
    
    if np.abs(eval_midpoint) < error_tol:
        # stopping condition, report midpoint as root
        return midpoint
    elif np.sign(eval_lower_bound) == np.sign(eval_midpoint):
        # case where midpoint is an improvement on lower_bound. 
        # Make recursive call with lower_bound = midpoint
        return bisection(midpoint, upper_bound, error_tol, top_samples, constants, target_num, rand_seed)
    elif np.sign(eval_upper_bound) == np.sign(eval_midpoint):
        # case where midpoint is an improvement on upper_bound. 
        # Make recursive call with upper_bound = midpoint
        return bisection(lower_bound, midpoint, error_tol, top_samples, constants, target_num, rand_seed)

    
#OR WE CAN USE A WHILE LOOP

def bisection(lower_bound, upper_bound, error_tol, top_samples, constants, target_num, rand_seed = None, verbose = False):
    """
    approximates a root of a function bounded by lower_bound and upper_bound to within a tolerance 
    
    Parameters:
    -----------
        lower_bound: float, lower bound of the distance, must be > 0
        upper_bound: float, lower bound of the distance, must be > lower_bound
        error_tol: floar, tolerance of error
        top_samples: pandas data frame, Collection of top liquid/vapor sampes
        constants: utils.r143a.R143aConstants, contains the infromation for a certain refrigerant
        target_num: int, the number of samples to choose next
        rand_seed: int, the seed number to use: None by default
        
    Returns:
    --------
        midpoint: The distance that satisfies the error criteria based on the target number
    """
    #Initialize Termination criteria and counter
    step = 1
    condition = True
    
    #While error > tolerance
    while condition:
        #Set error of upper and lower bound
#         print("Low B", lower_bound)
#         print("High B", upper_bound)
        eval_lower_bound = opt_dist(lower_bound, top_samples, constants, target_num, rand_seed)
        eval_upper_bound = opt_dist(upper_bound, top_samples, constants, target_num, rand_seed)
#         print("Low Eval",eval_lower_bound )
#         print("High Eval",eval_upper_bound )
        
        #Break loop if initial guesses are bad
        if eval_lower_bound*eval_upper_bound >=0:
            print("Increase Length of Upper Bound. Given bounds do not include the root!")
            break
        #Find the midpoint and evaluate it    
        midpoint = (lower_bound + upper_bound)/2
#         print("Mid B", midpoint)
        eval_midpoint = opt_dist(midpoint, top_samples, constants, target_num, rand_seed)
#         print("Mid Eval", eval_midpoint)
            
        if verbose == True:
            print('Iteration-%d, distance = %0.6f and error = %0.6f' % (step, midpoint, eval_midpoint))
            
        # Set the upper or lower bound depending on sign
        if  np.sign(eval_lower_bound) == np.sign(eval_midpoint):
            lower_bound = midpoint
        else:
            upper_bound = midpoint
        
        #Increase counter and terminate loop if error < error_tol
        step = step + 1
        condition = abs(eval_midpoint) > error_tol
 
    return midpoint

distance_opt = bisection()
print('\nRequired Distance is : %0.8f' % distance_opt)