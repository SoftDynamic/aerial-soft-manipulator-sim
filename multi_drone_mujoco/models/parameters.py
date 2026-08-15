"""Physical parameters shared by model construction and simulation."""

from multi_drone_mujoco.utils.enums import DroneModel


################################################################################
# Crazyflie 2.x physical parameters (from Forster 2015 system identification)
################################################################################

DRONE_PARAMS = {
    DroneModel.CF2X: {
        "mass": 0.027,                          # kg
        "arm_length": 0.0397,                   # m (motor-to-center)
        "thrust2weight_ratio": 2.25,
        "ixx": 1.4e-5,                          # kg*m^2
        "iyy": 1.4e-5,
        "izz": 2.17e-5,
        "kf": 3.16e-10,                         # thrust coeff (N/(rad/s)^2)
        "km": 7.94e-12,                         # torque coeff (Nm/(rad/s)^2)
        "prop_radius": 0.02325,                 # m
        "max_speed_kmh": 30.0,
        "gnd_eff_coeff": 11.36859,
        "drag_coeff_xy": 9.1785e-7,
        "drag_coeff_z": 10.311e-7,
        "dw_coeff_1": 2267.18,
        "dw_coeff_2": 0.16,
        "dw_coeff_3": -0.11,
        "collision_h": 0.03,
        "collision_r": 0.06,
        "collision_z_offset": 0.0,
    },
    DroneModel.CF2P: {
        "mass": 0.027,
        "arm_length": 0.0397,
        "thrust2weight_ratio": 2.25,
        "ixx": 1.4e-5,
        "iyy": 1.4e-5,
        "izz": 2.17e-5,
        "kf": 3.16e-10,
        "km": 7.94e-12,
        "prop_radius": 0.02325,
        "max_speed_kmh": 30.0,
        "gnd_eff_coeff": 11.36859,
        "drag_coeff_xy": 9.1785e-7,
        "drag_coeff_z": 10.311e-7,
        "dw_coeff_1": 2267.18,
        "dw_coeff_2": 0.16,
        "dw_coeff_3": -0.11,
        "collision_h": 0.03,
        "collision_r": 0.06,
        "collision_z_offset": 0.0,
    },
    DroneModel.RACE: {
        "mass": 0.250,
        "arm_length": 0.125,
        "thrust2weight_ratio": 4.0,
        "ixx": 4.86e-4,
        "iyy": 4.86e-4,
        "izz": 8.8e-4,
        "kf": 1.28e-8,
        "km": 5.964552e-10,
        "prop_radius": 0.0635,
        "max_speed_kmh": 100.0,
        "gnd_eff_coeff": 11.36859,
        "drag_coeff_xy": 9.1785e-7,
        "drag_coeff_z": 10.311e-7,
        "dw_coeff_1": 2267.18,
        "dw_coeff_2": 0.16,
        "dw_coeff_3": -0.11,
        "collision_h": 0.06,
        "collision_r": 0.15,
        "collision_z_offset": 0.0,
    },
}


CF2X_3G_PAYLOAD_PARAMS = {
    "mass": 0.003,          # kg
    "position": (0.0, 0.0, -0.020),
    "radius": 0.010,        # m
    "half_height": 0.005,   # m (MuJoCo cylinder size convention)
}
