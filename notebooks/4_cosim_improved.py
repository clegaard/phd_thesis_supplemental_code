import numpy as np
from numpy import cos
import matplotlib.pyplot as plt
import math


def step_controller(
    current_communication_time,
    communication_step_size,
    state_communication_point,
    parameters,
):
    internal_step_size = communication_step_size / 2**10

    def controller_dynamics(state):
        error = parameters["θ_setpoint"] - state["θ"]
        error_last = state["error_last"]

        if state["error_last"] is None:
            error_derivative = 0.0
        else:
            error_derivative = (error_last - error) / internal_step_size

        error_integral += error * internal_step_size

        u = (
            parameters["kp"] * error
            + parameters["ki"] * error_integral
            + parameters["kd"] * error_derivative
        )

        dudt = u - state["u"]
        d_error_last_dt = error_last - state["error_last"]
        d_error_integral_dt = error * internal_step_size

    t_internal, state_internal = simulate_euler(
        current_communication_time,
        current_communication_time + internal_step_size,
        internal_step_size,
        controller_dynamics,
        state_communication_point,
    )

    return t_internal, state_internal


def step_robot(
    current_communication_time,
    communication_step_size,
    state_communication_point,
):
    assert math.log(communication_step_size, 2).is_integer()
    internal_step_size = communication_step_size / 2**10

    def robot_dynamics(state):
        V_abs = 12.0  # voltage across coil
        K = 7.45  # torque coefficient
        g = 9.81  # gravitational acceleration
        b = 5.0  # motor shaft friction
        m = 5.0  # mass of joint
        R = 0.15  # electrical resistance ?
        L = 0.036  # motor inductance
        l = 1.0  # length of joint
        J = 0.5 * (m * l**2)  # moment of intertia

        θ, ω, i, u = state
        dθdt = ω
        dωdt = (K * i - b * ω - m * g * l * cos(θ)) / J
        didt = (u * V_abs - R * i - K * ω) / L
        dudt = 0.0
        return np.array((dθdt, dωdt, didt, dudt))

    t_internal, state_internal = simulate_euler(
        current_communication_time,
        current_communication_time + internal_step_size,
        internal_step_size,
        robot_dynamics,
        state_communication_point,
    )

    return t_internal, state_internal


def simulate_euler(t_begin, t_end, step_size, dynamics, state, parameters):
    ts = np.arange(t_begin, t_end + step_size, step_size)

    states = [state]
    for t in t:
        dxdt = dynamics(t, state, parameters)
        state = state + dxdt
        states.append(state)

    return ts, state


def run():
    kp = 1.0
    kd = 0.0
    ki = 0.01

    error_integral = 0.0
    error_last = None

    t_start = 0.0
    t_end = 10.0
    step_size = 0.0001
    θ_setpoint = 1.0

    t = np.arange(t_start, t_end + step_size, step_size)

    controller_parameters = [(1.0, 0.0, 0.01), (1.0, 0.5, 0.01), (1.0, 0.3, 0.01)]

    X = []
    U = []
    for kp, ki, kd in controller_parameters:
        xs = []
        us = []
        x = np.array((0.0, 0.0, 0.0))

        for _ in t:
            θ, *_ = x
            u, error_last, error_integral = step_controller(
                θ, θ_setpoint, step_size, error_last, error_integral, kp, ki, kd
            )
            dxdt = robot_dynamics(x, u, K, b, m, g, l, V_abs, R, L, J)
            x = x + step_size * dxdt
            xs.append(x)
            us.append(u)

        X.append(np.array(xs))
        U.append(us)

    # θ, ω, i = np.array(xs).T

    fig, ax = plt.subplots()
    ax.axhline(θ_setpoint, label="setpoint", color="red", linestyle="dotted")
    for (kp, ki, kd), x, u in zip(controller_parameters, X, U):
        ax.plot(t, x[:, 0], label=rf"$K_p={kp}, K_i={ki}, K_d={kd}$")

    ax.set_ylabel(rf"$\theta(t)$")
    ax.set_xlabel(rf"$t~[s]$")
    ax.legend()

    plt.savefig("figures/cosim/pure_python_simulation.pdf")

    plt.show()


if __name__ == "__main__":
    run()
