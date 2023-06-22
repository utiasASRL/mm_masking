import pickle
import matplotlib.pyplot as plt
import numpy as np
from  matplotlib.colors import LinearSegmentedColormap

def main():
    task = "icp" # Options are "scale", "icp", "bfar"
    icp_type = "pt2pl" # Options are "pt2pt" and "pt2pl"

    if task == "scale":
        p_names = ["scale"]
    elif task == "icp":
        p_names = ["trim", "huber delta"]
    elif task == "bfar":
        p_names = ["a", "b"]
    

    # Form result directory
    result_dir = 'results/' + task
    sweep_dir = 'results/' + task + '/sweep'
    learn_dir = 'results/' + task + '/learn'
    sweep_naming = sweep_dir + '/' + icp_type
    learn_naming = learn_dir + '/' + icp_type

    # Read in hand sweep results
    with open(sweep_naming + '_sweep_result.pkl', 'rb') as f:
        sweep_result = pickle.load(f)
    
    # Extract stacks from sweep result
    p1_stack = []
    p2_stack = []
    err_stack = []
    for params, err in sweep_result:
        #if err > 1.0: err = 1.0
        p1_stack.append(params[0])
        if len(params) > 1:
            p2_stack.append(params[1])
        err_stack.append(err)

    # Read in learned param results
    learned_params_data = np.load(learn_naming + '_param_hist.npy').squeeze()
    learned_params = learned_params_data[:, :-1]
    learned_param_loss = learned_params_data[:, -1]

    # Plot final result
    #print("Min err. norm: " + str(min_err))
    #print("For params: " + str(min_err_p1_p2))
    plot_sweep_layer(p1_stack, p2_stack, err_stack, learned_params, learned_param_loss, result_dir, p_names)


def plot_sweep_layer(p1_stack, p2_stack, err_stack, learned_params, learned_param_loss, result_dir, p_names=["p1", "p2"]):
    # If there is only one parameter, plot a 2D line
    if len(p2_stack) < 1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(p1_stack, err_stack)
        plt.plot(learned_params[:,0], learned_param_loss, color='black', linewidth=3)
        plt.plot(learned_params[0,0], learned_param_loss[0], 'r+', markersize=10)
        plt.plot(learned_params[-1,0], learned_param_loss[-1], 'g+', markersize=10)
        plt.xlabel(p_names[0])
        plt.ylabel("Error (%)")
        plt.savefig(result_dir + "/sweep_overlay.png")
        return
    
    # Convert the lists to numpy arrays
    p1_arr = np.array(p1_stack)
    p2_arr = np.array(p2_stack)
    res_arr = np.array(err_stack)

    # Reshape the res array to create a grid
    p1_unique = np.unique(p1_arr)
    p2_unique = np.unique(p2_arr)
    res_grid = res_arr.reshape(len(p1_unique), len(p2_unique)).T

    res_grid[res_grid == 100] = None

    # Create a 3D plot of the result surface
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p1_grid, p2_grid = np.meshgrid(p1_unique, p2_unique)

    c = ["darkgreen", "palegreen", "orange", "lightcoral", "red", "darkred"]
    v = [0,.15,.4,0.6,.9,1.]
    l = list(zip(v,c))
    cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
    #surf = ax.plot_surface(p1_grid, p2_grid, res_grid, cmap=cmap)
    contour = ax.contourf(p1_unique, p2_unique, res_grid, cmap=cmap)
    ax.set_xlabel(p_names[0])
    ax.set_ylabel(p_names[1])
    #ax.set_zlabel('Trans. Err. (%)')
    #fig.colorbar(contour)
    cbar = plt.colorbar(contour)
    #ax.view_init(elev=90, azim=2)

    # Plot the learned params trajectory on top of the surface
    # Extract the height from the surface
    x = learned_params[:,0]
    y = learned_params[:,1]
    ax.plot(x, y, color='black', linewidth=2)

    # plot the first point as a red cross
    ax.plot(x[0], y[0], 'r+', markersize=10)

    # plot the last point as a green cross
    ax.plot(x[-1], y[-1], 'g+', markersize=10)

    #ax.plot(learned_ab[:,0], learned_ab[:,1], 0.0*np.ones(learned_ab.shape[0]), color='black', linewidth=3)
    #ax.plot(learned_ab[:,0], learned_ab[:,1], 0.55*np.ones(learned_ab.shape[0]), color='black', linewidth=3)

    # Save figure
    plt.savefig(result_dir + '/sweep_overlay.png')

if __name__ == "__main__":
    main()