import sys
import os
import pickle
import jax

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from examples.params.obstacle_avoidance import problem, plotting_dict

from openscvx.ptr import PTR_main
from examples.plotting import plot_camera_animation, plot_animation, plot_scp_animation
from openscvx.config import Config

################################
# Author: Chris Hayner  
# Autonomous Controls Laboratory
################################

jax.config.update('jax_default_device', jax.devices('cpu')[0])

problem.initialize()
results = problem.solve()

# Check if results folder exists
if not os.path.exists('results'):
    os.makedirs('results')

# Save results
with open('results/results.pickle', 'wb') as f:
    pickle.dump(results, f) 

# Load results
with open('results/results.pickle', 'rb') as f:
    results = pickle.load(f) 

results = problem.post_process(results)
results.update(plotting_dict)
animation_plot = plot_animation(results, problem.params)
# animation_plot.show()
# camera_plot = plot_camera_animation(results, problem.params)
# camera_plot.show()
