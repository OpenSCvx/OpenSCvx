"""
Minimum Time-to-Climb of a Supersonic Aircraft

This example is taken from:
Bryson, A. E., Desai, M. N. and Hoffman, W. C., "Energy-State
Approximation in Performance Optimization of Supersonic
Aircraft," Journal of Aircraft, Vol. 6, No. 6, November-December,
1969, pp. 481-488.
"""

import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import TrajOptProblem

n = 30  # Number of Nodes

### Atmospheric Data (U.S. 1976 Standard Atmosphere) ###
# Format: [Altitude (m), Density (kg/m^3), Speed of Sound (m/s)]
us1976 = np.array(
    [
        [-2000, 1.478e00, 3.479e02],
        [0, 1.225e00, 3.403e02],
        [2000, 1.007e00, 3.325e02],
        [4000, 8.193e-01, 3.246e02],
        [6000, 6.601e-01, 3.165e02],
        [8000, 5.258e-01, 3.081e02],
        [10000, 4.135e-01, 2.995e02],
        [12000, 3.119e-01, 2.951e02],
        [14000, 2.279e-01, 2.951e02],
        [16000, 1.665e-01, 2.951e02],
        [18000, 1.216e-01, 2.951e02],
        [20000, 8.891e-02, 2.951e02],
        [22000, 6.451e-02, 2.964e02],
        [24000, 4.694e-02, 2.977e02],
        [26000, 3.426e-02, 2.991e02],
        [28000, 2.508e-02, 3.004e02],
        [30000, 1.841e-02, 3.017e02],
    ]
)

alt_atm = us1976[:, 0]
rho_atm = us1976[:, 1]
sos_atm = us1976[:, 2]
### End Atmospheric Data ###

### Propulsion Data ###
mach_tab = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
alt_tab = 304.8 * np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 70])  # Altitudes (ft to m)
thrust_tab = 4448.222 * np.array(
    [  # Thrust table (thousands of pounds to Newtons)
        [24.2, 24.0, 20.3, 17.3, 14.5, 12.2, 10.2, 5.7, 3.4, 0.1],
        [28.0, 24.6, 21.1, 18.1, 15.2, 12.8, 10.7, 6.5, 3.9, 0.2],
        [28.3, 25.2, 21.9, 18.7, 15.9, 13.4, 11.2, 7.3, 4.4, 0.4],
        [30.8, 27.2, 23.8, 20.5, 17.3, 14.7, 12.3, 8.1, 4.9, 0.8],
        [34.5, 30.3, 26.6, 23.2, 19.8, 16.8, 14.1, 9.4, 5.6, 1.1],
        [37.9, 34.3, 30.4, 26.8, 23.3, 19.8, 16.8, 11.2, 6.8, 1.4],
        [36.1, 38.0, 34.9, 31.3, 27.3, 23.6, 20.1, 13.4, 8.3, 1.7],
        [36.1, 36.6, 38.5, 36.1, 31.6, 28.1, 24.2, 16.2, 10.0, 2.2],
        [36.1, 35.2, 42.1, 38.7, 35.7, 32.0, 28.1, 19.3, 11.9, 2.9],
        [36.1, 33.8, 45.7, 41.3, 39.8, 34.6, 31.1, 21.7, 13.3, 3.1],
    ]
)
### End Propulsion Data ###

### Aerodynamic Data ###
mach_aero = np.array([0, 0.4, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8])
cl_alpha_tab = np.array([3.44, 3.44, 3.44, 3.58, 4.44, 3.44, 3.01, 2.86, 2.44])
cd0_tab = np.array([0.013, 0.013, 0.013, 0.014, 0.031, 0.041, 0.039, 0.036, 0.035])
eta_tab = np.array([0.54, 0.54, 0.54, 0.75, 0.79, 0.78, 0.89, 0.93, 0.93])
### End Aerodynamic Data ###

### Vehicle Parameters ###
r_earth = 6378145.0  # Earth radius (m)
mu_earth = 3.986e14  # Earth gravitational parameter (m^3/s^2)
s_ref = 49.2386  # Reference area (m^2)
g0 = 9.80665  # Gravity at sea level (m/s^2)
isp = 1600.0  # Specific impulse (s)
### End Vehicle Parameters ###

### Problem Parameters ###
total_time = 800.0
alt_0 = 0.0
alt_f = 19994.88
speed_0 = 129.314
speed_f = 295.092
fpa_0 = 0.0
fpa_f = 0.0
mass_0 = 19050.864
### End Problem Parameters ###

# Define state components
altitude = ox.State("altitude", shape=(1,))
altitude.max = np.array([21031.2])
altitude.min = np.array([0.0])
altitude.initial = np.array([alt_0])
altitude.final = np.array([alt_f])
altitude.guess = np.linspace([alt_0], [alt_f], n)

speed = ox.State("speed", shape=(1,))
speed.max = np.array([1000.0])
speed.min = np.array([5.0])
speed.initial = np.array([speed_0])
speed.final = np.array([speed_f])
speed.guess = np.linspace([speed_0], [speed_f], n)

fpa = ox.State("fpa", shape=(1,))  # Flight path angle
fpa.max = np.array([40 * np.pi / 180])
fpa.min = np.array([-40 * np.pi / 180])
fpa.initial = np.array([fpa_0])
fpa.final = np.array([fpa_f])
fpa.guess = np.zeros((n, 1))

mass = ox.State("mass", shape=(1,))
mass.max = np.array([20410.0])
mass.min = np.array([100.0])
mass.initial = np.array([mass_0])
mass.final = [("free", mass_0)]
mass.guess = mass_0 * np.ones((n, 1))

# Define control components
alpha = ox.Control("alpha", shape=(1,))  # Angle of attack
alpha.max = np.array([np.pi / 4])
alpha.min = np.array([-np.pi / 4])
alpha.guess = np.zeros((n, 1))

# Define list of all states (needed for TrajOptProblem and constraints)
states = [altitude, speed, fpa, mass]
controls = [alpha]

# Compute radius from altitude
r = altitude[0] + r_earth

# Atmospheric properties
rho = ox.Linterp(alt_atm, rho_atm, altitude[0])
sos = ox.Linterp(alt_atm, sos_atm, altitude[0])
mach = speed[0] / sos

# Aerodynamic coefficients
cl_alpha = ox.Linterp(mach_aero, cl_alpha_tab, mach)
cd0 = ox.Linterp(mach_aero, cd0_tab, mach)
eta = ox.Linterp(mach_aero, eta_tab, mach)

# Thrust
thrust = ox.Linterp(alt_tab, mach_tab, thrust_tab, altitude[0], mach)

# Drag and lift coefficients
cd = cd0 + eta * cl_alpha * alpha[0] ** 2
cl = cl_alpha * alpha[0]

# Dynamic pressure
q = 0.5 * rho * speed[0] ** 2

# Aerodynamic forces
drag = q * s_ref * cd
lift = q * s_ref * cl

# Gravitational acceleration
g = mu_earth / (r**2)

# Create symbolic dynamics
h_dot = speed[0] * ox.Sin(fpa[0])
v_dot = (thrust * ox.Cos(alpha[0]) - drag) / mass[0] - g * ox.Sin(fpa[0])
fpa_dot = (thrust * ox.Sin(alpha[0]) + lift) / (mass[0] * speed[0]) + ox.Cos(fpa[0]) * (
    speed[0] / r - g / speed[0]
)
m_dot = -thrust / (g0 * isp)

# Define dynamics as dictionary mapping state names to their derivatives
dynamics = {
    "altitude": h_dot,
    "speed": v_dot,
    "fpa": fpa_dot,
    "mass": m_dot,
}

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

for control in controls:
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])


problem = TrajOptProblem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time_initial=0.0,
    time_final=("minimize", total_time),
    time_derivative=1.0,  # Real time
    time_min=0.0,
    time_max=total_time,
    constraints=constraints,
    N=n,
)

problem.settings.scp.w_tr = 1e0
problem.settings.scp.lam_cost = 1e-1
problem.settings.scp.lam_vc = 1e2
problem.settings.scp.ep_tr = 1e-3
problem.settings.scp.ep_vb = 1e-4
problem.settings.scp.ep_vc = 1e-8

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    print(f"\nOptimal time to climb: {results['time_final']:.2f} seconds")
    print(f"Final mass: {results['x'][-1, 3]:.2f} kg")
    print(f"Fuel used: {mass_0 - results['x'][-1, 3]:.2f} kg")
