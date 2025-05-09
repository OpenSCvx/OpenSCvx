from plotly.subplots import make_subplots
import random
import plotly.graph_objects as go
import numpy as np
import pickle

from openscvx.utils import qdcm, get_kp_pose
from openscvx.config import Config

def full_subject_traj_time(results, params):
    x_full = results["x_full"]
    x_nodes = results["x"]
    t_nodes = x_nodes[:,params.sim.idx_t]
    t_full = results['t_full']
    subs_traj = []
    subs_traj_node = []
    subs_traj_sen = []
    subs_traj_sen_node = []
    
    # if hasattr(params.dyn, 'get_kp_pose'):
    if "moving_subject" in results and "init_poses" in results:
        init_poses = results["init_poses"]
        subs_traj.append(get_kp_pose(t_full, init_poses))
        subs_traj_node.append(get_kp_pose(t_nodes, init_poses))
    elif "init_poses" in results:
        for pose in results["init_poses"]:
            # repeat the pose for all time steps
            pose_full = np.repeat(pose[:,np.newaxis], x_full.shape[0], axis=1).T
            subs_traj.append(pose_full)
            
            pose_node = np.repeat(pose[:,np.newaxis], x_nodes.shape[0], axis=1).T
            subs_traj_node.append(pose_node)
    else:
        raise ValueError("No valid method to get keypoint poses.")

    if "R_sb" in results:
        R_sb = results["R_sb"]
        for sub_traj in subs_traj:
            sub_traj_sen = []
            for i in range(x_full.shape[0]):
                sub_pose = sub_traj[i]
                sub_traj_sen.append(R_sb @ qdcm(x_full[i, 6:10]).T @ (sub_pose - x_full[i, 0:3]))
            subs_traj_sen.append(np.array(sub_traj_sen).squeeze())

        for sub_traj_node in subs_traj_node:
            sub_traj_sen_node = []
            for i in range(x_nodes.shape[0]):
                sub_pose = sub_traj_node[i]
                sub_traj_sen_node.append(R_sb @ qdcm(x_nodes[i, 6:10]).T @ (sub_pose - x_nodes[i, 0:3]).T)
            subs_traj_sen_node.append(np.array(sub_traj_sen_node).squeeze())
        return subs_traj, subs_traj_sen, subs_traj_node, subs_traj_sen_node
    else:
        raise ValueError("`R_sb` not found in results dictionary. Cannot compute sensor frame.")

def save_gate_parameters(gates, params: Config):
    gate_centers = []
    gate_vertices = []
    for gate in gates:
        gate_centers.append(gate.center)
        gate_vertices.append(gate.vertices)
    gate_params = dict(
        gate_centers = gate_centers,
        gate_vertices = gate_vertices
    )

    # Use pickle to save the gate parameters
    with open('results/gate_params.pickle', 'wb') as f:
        pickle.dump(gate_params, f)

def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
            }

def plot_constraint_violation(result, params: Config):
    fig = make_subplots(rows=2, cols=3, subplot_titles=(r'$\text{Obstacle Violation}$', r'$\text{Sub VP Violation}$', r'$\text{Sub Min Violation}$', r'$\text{Sub Max Violation}$', r'$\text{Sub Direc Violation}$', r'$\text{State Bound Violation}$', r'$\text{Total Violation}$'))
    fig.update_layout(template='plotly_dark', title=r'$\text{Constraint Violation}$')

    if "obs_vio" in result:
        obs_vio = result["obs_vio"]
        for i in range(obs_vio.shape[0]):
            color = f'rgb({random.randint(10,255)}, {random.randint(10,255)}, {random.randint(10,255)})'
            fig.add_trace(go.Scatter(y=obs_vio[i], mode='lines', showlegend=False, line=dict(color=color, width = 2)), row=1, col=1)
        i = 0
    else:
        print("'obs_vio' not found in result dictionary.")

    # Make names of each state in the state vector
    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'q0', 'q1', 'q2', 'q3', 'wx', 'wy', 'wz', 'ctcs']

    if "sub_vp_vio" in result and "sub_min_vio" in result and "sub_max_vio" in result:
        sub_vp_vio = result["sub_vp_vio"]
        sub_min_vio = result["sub_min_vio"]
        sub_max_vio = result["sub_max_vio"]
        for i in range(sub_vp_vio.shape[0]):
            color = f'rgb({random.randint(10,255)}, {random.randint(10,255)}, {random.randint(10,255)})'
            fig.add_trace(go.Scatter(y=sub_vp_vio[i], mode='lines', showlegend=True, name = 'LoS ' + str(i) + ' Error', line=dict(color=color, width = 2)), row=1, col=2)
            if params.vp.tracking:
                fig.add_trace(go.Scatter(y=sub_min_vio[i], mode='lines', showlegend=False, line=dict(color=color, width = 2)), row=1, col=3)
                fig.add_trace(go.Scatter(y=sub_max_vio[i], mode='lines', showlegend=False, line=dict(color=color, width = 2)), row=2, col=1)
            else:
                fig.add_trace(go.Scatter(y=[], mode='lines', showlegend=False, line=dict(color=color, width = 2)), row=1, col=3)
                fig.add_trace(go.Scatter(y=[], mode='lines', showlegend=False, line=dict(color=color, width = 2)), row=2, col=1)
        i = 0
    else:
        print("'sub_vp_vio', 'sub_min_vio', or 'sub_max_vio' not found in result dictionary.")
    
    if "sub_direc_vio" in result:
        sub_direc_vio = result["sub_direc_vio"]
        # fig.add_trace(go.Scatter(y=sub_direc_vio, mode='lines', showlegend=False, line=dict(color='red', width = 2)), row=2, col=2)
    else:
        print("'sub_direc_vio' not found in result dictionary.")

    if "state_bound_vio" in result:
        state_bound_vio = result["state_bound_vio"]
        for i in range(state_bound_vio.shape[0]):
            color = f'rgb({random.randint(10,255)}, {random.randint(10,255)}, {random.randint(10,255)})'
            fig.add_trace(go.Scatter(y=state_bound_vio[:,i], mode='lines', showlegend=True, name = state_names[i] + ' Error', line=dict(color=color, width = 2)), row=2, col=3)
    else:
        print("'state_bound_vio' not found in result dictionary.")

    fig.show()

def plot_initial_guess(result, params: Config):
    x_positions = result["x"][0:3]
    x_attitude = result["x"][6:10]
    subs_positions = result["sub_positions"]

    fig = go.Figure(go.Scatter3d(x=[], y=[], z=[], mode='lines+markers', line=dict(color='gray', width = 2)))

    # Plot the position of the drone
    fig.add_trace(go.Scatter3d(x=x_positions[0], y=x_positions[1], z=x_positions[2], mode='lines+markers', line=dict(color='green', width = 5)))

    # Plot the attitude of the drone
    # Draw drone attitudes as axes
    step = 1
    indices = np.array(list(range(x_positions.shape[1])))
    for i in range(0, len(indices), step):
        att = x_attitude[:, indices[i]]

        # Convert quaternion to rotation matrix
        rotation_matrix = qdcm(att)

        # Extract axes from rotation matrix
        axes = 2 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotated_axes = np.dot(rotation_matrix, axes).T

        colors = ['#FF0000', '#00FF00', '#0000FF']

        for k in range(3):
            axis = rotated_axes[k]
            color = colors[k]

            fig.add_trace(go.Scatter3d(
                x=[x_positions[0, indices[i]], x_positions[0, indices[i]] + axis[0]],
                y=[x_positions[1, indices[i]], x_positions[1, indices[i]] + axis[1]],
                z=[x_positions[2, indices[i]], x_positions[2, indices[i]] + axis[2]],
                mode='lines+text',
                line=dict(color=color, width=4),
                showlegend=False
            ))
    
    fig.update_layout(template='plotly_dark')
    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=10, y=10, z=10)))
    fig.update_layout(scene=dict(xaxis=dict(range=[-200, 200]), yaxis=dict(range=[-200, 200]), zaxis=dict(range=[-200, 200])))

    # Plot the keypoint
    for sub_positions in subs_positions:
        fig.add_trace(go.Scatter3d(x=sub_positions[:,0], y=sub_positions[:,1], z=sub_positions[:,2], mode='lines+markers', line=dict(color='red', width = 5), name='Subject'))
    fig.show()

def plot_scp_animation(result: dict,
                       params = None,
                       path=""):
    tof = result["t_final"]
    title = f'SCP Simulation: {tof} seconds'
    drone_positions = result["x_full"][:, :3]
    drone_attitudes = result["x_full"][:, 6:10]
    drone_forces = result["u_full"][:, :3]
    scp_interp_trajs = scp_traj_interp(result["x_history"], params)
    scp_ctcs_trajs = result["x_history"]
    scp_multi_shoot = result["discretization"]
    # obstacles = result_ctcs["obstacles"]
    # gates = result_ctcs["gates"]
    if "moving_subject" in result or "init_poses" in result:
        subs_positions, _, _, _ = full_subject_traj_time(result, params)
    fig = go.Figure(go.Scatter3d(x=[], y=[], z=[], mode='lines+markers', line=dict(color='gray', width = 2), name='SCP Iterations'))
    for j in range(200):
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines+markers', line=dict(color='gray', width = 2)))

    # fig.update_layout(height=1000)

    fig.add_trace(go.Scatter3d(x=drone_positions[:,0], y=drone_positions[:,1], z=drone_positions[:,2], mode='lines', line=dict(color='green', width = 5), name='Nonlinear Propagation'))

    fig.update_layout(template='plotly_dark', title=title)

    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=10, y=10, z=10)))
    fig.update_layout(scene=dict(xaxis=dict(range=[-200, 200]), yaxis=dict(range=[-200, 200]), zaxis=dict(range=[-200, 200])))

    # Extract the number of states and controls from the parameters
    n_x = params.sim.n_states
    n_u = params.sim.n_controls

    # Define indices for slicing the augmented state vector
    i0 = 0
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u
    i5 = i4 + n_x

    # Plot the attitudes of the SCP Trajs
    frames = []
    traj_iter = 0

    for scp_traj in scp_ctcs_trajs:
        drone_positions = scp_traj[:,0:3]
        drone_attitudes = scp_traj[:,6:10]
        frame = go.Frame(name=str(traj_iter))
        data = []
        # Plot the multiple shooting trajectories
        pos_traj = []
        if traj_iter < len(scp_multi_shoot):
            for i_multi in range(scp_multi_shoot[traj_iter].shape[1]):
                pos_traj.append(scp_multi_shoot[traj_iter][:,i_multi].reshape(-1, i5)[:,0:3])
            pos_traj = np.array(pos_traj)
            
            for j in range(pos_traj.shape[1]):
                if j == 0:
                    data.append(go.Scatter3d(x=pos_traj[:,j, 0], y=pos_traj[:,j, 1], z=pos_traj[:,j, 2], mode='lines', legendgroup='Multishot Trajectory', name='Multishot Trajectory ' + str(traj_iter), showlegend=True, line=dict(color='blue', width = 5)))
                else:
                    data.append(go.Scatter3d(x=pos_traj[:,j, 0], y=pos_traj[:,j, 1], z=pos_traj[:,j, 2], mode='lines', legendgroup='Multishot Trajectory', showlegend=False, line=dict(color='blue', width = 5)))
        
            
        for i in range(drone_attitudes.shape[0]):
            att = drone_attitudes[i]

            # Convert quaternion to rotation matrix
            rotation_matrix = qdcm(att)

            # Extract axes from rotation matrix
            axes = 2 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            rotated_axes = np.dot(rotation_matrix, axes.T).T

            colors = ['#FF0000', '#00FF00', '#0000FF']

            for k in range(3):
                axis = rotated_axes[k]
                color = colors[k]

                data.append(go.Scatter3d(
                    x=[scp_traj[i, 0], scp_traj[i, 0] + axis[0]],
                    y=[scp_traj[i, 1], scp_traj[i, 1] + axis[1]],
                    z=[scp_traj[i, 2], scp_traj[i, 2] + axis[2]],
                    mode='lines+text',
                    line=dict(color=color, width=4),
                    showlegend=False
                ))
        traj_iter += 1  
        frame.data = data
        frames.append(frame)
    fig.frames = frames 

    i = 1
    if "obstacles_centers" in result:
        for center, axes, radius in zip(result['obstacles_centers'], result['obstacles_axes'], result['obstacles_radii']):
            n = 20
            # Generate points on the unit sphere
            u = np.linspace(0, 2 * np.pi, n)
            v = np.linspace(0, np.pi, n)

            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))

            # Scale points by radii
            x = 1/radius[0] * x
            y = 1/radius[1] * y
            z = 1/radius[2] * z

            # Rotate and translate points
            points = np.array([x.flatten(), y.flatten(), z.flatten()])
            points = axes @ points
            points = points.T + center

            fig.add_trace(go.Surface(x=points[:, 0].reshape(n,n), y=points[:, 1].reshape(n,n), z=points[:, 2].reshape(n,n), opacity = 0.5, showscale=False))

    if "vertices" in result:
        for vertices in result["vertices"]:
            # Plot a line through the vertices of the gate
            fig.add_trace(go.Scatter3d(x=[vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0], vertices[0][0]], y=[vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1], vertices[0][1]], z=[vertices[0][2], vertices[1][2], vertices[2][2], vertices[3][2], vertices[0][2]], mode='lines', showlegend=False, line=dict(color='blue', width=10)))
            
    # Add the subject positions
    if "n_subs" in result and result["n_subs"] != 0:     
        if "moving_subject" in result:
            if result["moving_subject"]:
                for sub_positions in subs_positions:
                    fig.add_trace(go.Scatter3d(x=sub_positions[:,0], y=sub_positions[:,1], z=sub_positions[:,2], mode='lines', line=dict(color='red', width = 5), showlegend=False))
        else:
            # Plot the subject positions as points
            for sub_positions in subs_positions:
                fig.add_trace(go.Scatter3d(x=sub_positions[:,0], y=sub_positions[:,1], z=sub_positions[:,2], mode='markers', marker=dict(size=10, color='red'), showlegend=False))


    fig.add_trace(go.Surface(x=[-200, 200, 200, -200], y=[-200, -200, 200, 200], z=[[0, 0], [0, 0], [0, 0], [0, 0]], opacity=0.3, showscale=False, colorscale='Greys', showlegend = True, name='Ground Plane'))

    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=10, y=10, z=10)))
    fig.update_layout(scene=dict(xaxis=dict(range=[-200, 200]), yaxis=dict(range=[-200, 200]), zaxis=dict(range=[-200, 200])))

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.8,
            "x": 0.15,
            "y": 0.32,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": f.name,
                    "method": "animate",
                } for f in fig.frames
            ]
        }
    ]

    fig.update_layout(updatemenus = [{"buttons":[
                                        {
                                            "args": [None, frame_args(50)],
                                            "label": "Play",
                                            "method": "animate",
                                        },
                                        {
                                            "args": [[None], frame_args(0)],
                                            "label": "Pause",
                                            "method": "animate",
                                    }],

                                    "direction": "left",
                                    "pad": {"r": 10, "t": 70},
                                    "type": "buttons",
                                    "x": 0.15,
                                    "y": 0.32,
                                }
                            ],
                            sliders=sliders
                        )
    fig.update_layout(sliders=sliders)

    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=10, y=10, z=10)))
    fig.update_layout(scene=dict(xaxis=dict(range=[-200, 200]), yaxis=dict(range=[-200, 200]), zaxis=dict(range=[-200, 200])))

    # Overlay the title onto the plot
    fig.update_layout(title_y=0.95, title_x=0.5)

    # Overlay the sliders and buttons onto the plot
    fig.update_layout(updatemenus = [{"buttons":[
                                        {
                                            "args": [None, frame_args(50)],
                                            "label": "Play",
                                            "method": "animate",
                                        },
                                        {
                                            "args": [[None], frame_args(0)],
                                            "label": "Pause",
                                            "method": "animate",
                                    }],

                                    "direction": "left",
                                    "pad": {"r": 10, "t": 70},
                                    "type": "buttons",
                                    "x": 0.15,
                                    "y": 0.32,
                                }
                            ],
                            sliders=sliders
                        )
    
    

    # Show the legend overlayed on the plot
    fig.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.75))

    # fig.update_layout(height=450, width = 800)

    # Remove the black border around the fig
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # Rmeove the background from the legend
    fig.update_layout(legend=dict(bgcolor='rgba(0,0,0,0)'))

    fig.update_xaxes(
        dtick=1.0,
        showline=False
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        showline=False,
        dtick=1.0
    )

    # Rotate the camera view to the left
    if not "moving_subject" in result:
        fig.update_layout(scene_camera=dict(up=dict(x=0, y=0, z=90), center=dict(x=1, y=0.3, z=1), eye=dict(x=-1, y=2, z=1)))

    fig.show()

def scp_traj_interp(scp_trajs, params: Config):
    scp_prop_trajs = []
    for traj in scp_trajs:
        states = []
        for k in range(params.scp.n):
            traj_temp = np.repeat(np.expand_dims(traj[k], axis = 1), params.prp.inter_sample - 1, axis = 1)
            for i in range(1, params.prp.inter_sample - 1):
                states.append(traj_temp[:,i])
        scp_prop_trajs.append(np.array(states))
    return scp_prop_trajs

def plot_state(result, params: Config):
    scp_trajs = scp_traj_interp(result["x_history"], params)
    x_full = result["x_full"]

    fig = make_subplots(rows=2, cols=7, subplot_titles=('X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity', 'CTCS Augmentation', 'Q1', 'Q2', 'Q3', 'Q4', 'X Angular Rate', 'Y Angular Rate', 'Z Angular Rate'))
    fig.update_layout(title_text="State Trajectories", template='plotly_dark')

    # Plot the position
    x_min = params.sim.min_state[0]
    x_max = params.sim.max_state[0]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,0], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(y=x_full[:,0], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=1)
    fig.add_hline(y=x_min, line=dict(color='red', width=2), row = 1, col = 1)
    fig.add_hline(y=x_max, line=dict(color='red', width=2), row = 1, col = 1)
    fig.update_yaxes(range=[x_min, x_max], row=1, col=1)

    y_min = params.sim.min_state[1]
    y_max = params.sim.max_state[1]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,1], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=2)
    fig.add_trace(go.Scatter(y=x_full[:,1], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=2)
    fig.add_hline(y=y_min, line=dict(color='red', width=2), row = 1, col = 2)
    fig.add_hline(y=y_max, line=dict(color='red', width=2), row = 1, col = 2)

    z_min = params.sim.min_state[2]
    z_max = params.sim.max_state[2]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,2], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=3)
    fig.add_trace(go.Scatter(y=x_full[:,2], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=3)
    fig.add_hline(y=z_min, line=dict(color='red', width=2), row = 1, col = 3)
    fig.add_hline(y=z_max, line=dict(color='red', width=2), row = 1, col = 3)

    # Plot the velocity
    vx_min = params.sim.min_state[3]
    vx_max = params.sim.max_state[3]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,3], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=4)
    fig.add_trace(go.Scatter(y=x_full[:,3], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=4)
    fig.add_hline(y=vx_min, line=dict(color='red', width=2), row = 1, col = 4)
    fig.add_hline(y=vx_max, line=dict(color='red', width=2), row = 1, col = 4)

    vy_min = params.sim.min_state[4]
    vy_max = params.sim.max_state[4]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,4], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=5)
    fig.add_trace(go.Scatter(y=x_full[:,4], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=5)
    fig.add_hline(y=vy_min, line=dict(color='red', width=2), row = 1, col = 5)
    fig.add_hline(y=vy_max, line=dict(color='red', width=2), row = 1, col = 5)

    vz_min = params.sim.min_state[5]
    vz_max = params.sim.max_state[5]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,5], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=6)
    fig.add_trace(go.Scatter(y=x_full[:,5], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=6)
    fig.add_hline(y=vz_min, line=dict(color='red', width=2), row = 1, col = 6)
    fig.add_hline(y=vz_max, line=dict(color='red', width=2), row = 1, col = 6)

    # # Plot the norm of the quaternion
    # for traj in scp_trajs:
    #     fig.add_trace(go.Scatter(y=la.norm(traj[1:,6:10], axis = 1), mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=7)
    # fig.add_trace(go.Scatter(y=la.norm(x_full[1:,6:10], axis = 1), mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=7)
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,-1], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=7)
    fig.add_trace(go.Scatter(y=x_full[:,-1], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=7)
    # fig.add_hline(y=vz_min, line=dict(color='red', width=2), row = 1, col = 6)
    # fig.add_hline(y=vz_max, line=dict(color='red', width=2), row = 1, col = 6)

    # Plot the attitude
    q1_min = params.sim.min_state[6]
    q1_max = params.sim.max_state[6]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,6], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=1)
    fig.add_trace(go.Scatter(y=x_full[:,6], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=1)
    fig.add_hline(y=q1_min, line=dict(color='red', width=2), row = 2, col = 1)
    fig.add_hline(y=q1_max, line=dict(color='red', width=2), row = 2, col = 1)

    q2_min = params.sim.min_state[7]
    q2_max = params.sim.max_state[7]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,7], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=2)
    fig.add_trace(go.Scatter(y=x_full[:,7], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=2)
    fig.add_hline(y=q2_min, line=dict(color='red', width=2), row = 2, col = 2)
    fig.add_hline(y=q2_max, line=dict(color='red', width=2), row = 2, col = 2)

    q3_min = params.sim.min_state[8]
    q3_max = params.sim.max_state[8]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,8], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=3)
    fig.add_trace(go.Scatter(y=x_full[:,8], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=3)
    fig.add_hline(y=q3_min, line=dict(color='red', width=2), row = 2, col = 3)
    fig.add_hline(y=q3_max, line=dict(color='red', width=2), row = 2, col = 3)

    q4_min = params.sim.min_state[9]
    q4_max = params.sim.max_state[9]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,9], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=4)
    fig.add_trace(go.Scatter(y=x_full[:,9], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=4)
    fig.add_hline(y=q4_min, line=dict(color='red', width=2), row = 2, col = 4)
    fig.add_hline(y=q4_max, line=dict(color='red', width=2), row = 2, col = 4)

    # Plot the angular rate
    wx_min = params.sim.min_state[10]
    wx_max = params.sim.max_state[10]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,10], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=5)
    fig.add_trace(go.Scatter(y=x_full[:,10], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=5)
    fig.add_hline(y=wx_min, line=dict(color='red', width=2), row = 2, col = 5)
    fig.add_hline(y=wx_max, line=dict(color='red', width=2), row = 2, col = 5)

    wy_min = params.sim.min_state[11]
    wy_max = params.sim.max_state[11]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,11], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=6)
    fig.add_trace(go.Scatter(y=x_full[:,11], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=6)
    fig.add_hline(y=wy_min, line=dict(color='red', width=2), row = 2, col = 6)
    fig.add_hline(y=wy_max, line=dict(color='red', width=2), row = 2, col = 6)

    wz_min = params.sim.min_state[12]
    wz_max = params.sim.max_state[12]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,12], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=7)
    fig.add_trace(go.Scatter(y=x_full[:,12], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=7)
    fig.add_hline(y=wz_min, line=dict(color='red', width=2), row = 2, col = 7)
    fig.add_hline(y=wz_max, line=dict(color='red', width=2), row = 2, col = 7)
    fig.show()

def plot_control(result, params: Config):
    scp_controls = result["u_history"]
    u = result["u"]

    fx_min = params.sim.min_control[0]
    fx_max = params.sim.max_control[0]

    fy_min = params.sim.min_control[1]
    fy_max = params.sim.max_control[1]

    fz_min = params.sim.min_control[2]
    fz_max = params.sim.max_control[2]

    tau_x_min = params.sim.max_control[3]
    tau_x_max = params.sim.min_control[3]

    tau_y_min = params.sim.max_control[4]
    tau_y_max = params.sim.min_control[4]

    tau_z_min = params.sim.max_control[5]
    tau_z_max = params.sim.min_control[5]

    fig = make_subplots(rows=2, cols=3, subplot_titles=('X Force', 'Y Force', 'Z Force', 'X Torque', 'Y Torque', 'Z Torque'))
    fig.update_layout(title_text="Control Trajectories", template='plotly_dark')

    for traj in scp_controls:
        fig.add_trace(go.Scatter(y=traj[0], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(y=u[0], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=1)
    fig.add_hline(y=fx_min, line=dict(color='red', width=2), row = 1, col = 1)
    fig.add_hline(y=fx_max, line=dict(color='red', width=2), row = 1, col = 1)

    for traj in scp_controls:
        fig.add_trace(go.Scatter(y=traj[1], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=2)
    fig.add_trace(go.Scatter(y=u[1], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=2)
    fig.add_hline(y=fy_min, line=dict(color='red', width=2), row = 1, col = 2)
    fig.add_hline(y=fy_max, line=dict(color='red', width=2), row = 1, col = 2)

    for traj in scp_controls:
        fig.add_trace(go.Scatter(y=traj[2], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=3)
    fig.add_trace(go.Scatter(y=u[2], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=3)
    fig.add_hline(y=fz_min, line=dict(color='red', width=2), row = 1, col = 3)
    fig.add_hline(y=fz_max, line=dict(color='red', width=2), row = 1, col = 3)

    for traj in scp_controls:
        fig.add_trace(go.Scatter(y=traj[3], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=1)
    fig.add_trace(go.Scatter(y=u[3], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=1)
    fig.add_hline(y=tau_x_min, line=dict(color='red', width=2), row = 2, col = 1)
    fig.add_hline(y=tau_x_max, line=dict(color='red', width=2), row = 2, col = 1)

    for traj in scp_controls:
        fig.add_trace(go.Scatter(y=traj[4], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=2)
    fig.add_trace(go.Scatter(y=u[4], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=2)
    fig.add_hline(y=tau_y_min, line=dict(color='red', width=2), row = 2, col = 2)
    fig.add_hline(y=tau_y_max, line=dict(color='red', width=2), row = 2, col = 2)

    for traj in scp_controls:
        fig.add_trace(go.Scatter(y=traj[5], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=3)
    fig.add_trace(go.Scatter(y=u[5], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=3)
    fig.add_hline(y=tau_z_min, line=dict(color='red', width=2), row = 2, col = 3)
    fig.add_hline(y=tau_z_max, line=dict(color='red', width=2), row = 2, col = 3)

    fig.show()

def plot_losses(result, params: Config):
    # Plot J_tr, J_vb, J_vc, J_vc_ctcs
    J_tr = result["J_tr_history"]
    J_vb = result["J_vb_history"]
    J_vc = result["J_vc_history"]
    J_vc_ctcs = result["J_vc_ctcs_vec"]

    fig = make_subplots(rows=2, cols=2, subplot_titles=('J_tr', 'J_vb', 'J_vc', 'J_vc_ctcs'))
    fig.update_layout(title_text="Losses", template='plotly_dark')

    fig.add_trace(go.Scatter(y=J_tr, mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=J_vb, mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=J_vc, mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=1)

    # Set y-axis to log scale for each subplot
    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_yaxes(type='log', row=1, col=2)
    fig.update_yaxes(type='log', row=2, col=1)
    fig.update_yaxes(type='log', row=2, col=2)

    fig.show()