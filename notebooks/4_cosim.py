import numpy as np
from numpy import cos
import matplotlib.pyplot as plt
from copy import deepcopy

from dataclasses import dataclass


class Controller:
    def step(self, communication_step_size, robot: "Robot"):
        error = self.θ_setpoint - robot.θ

        if self.error_last is not None:
            error_derivative = (self.error_last - error) / communication_step_size
        else:
            error_derivative = 0.0

        self.u = (
            self.kp * error + self.ki * self.error_integral + self.kd * error_derivative
        )
        self.error_integral += error * communication_step_size
        self.error_last = error

    def __init__(self, kp, ki, kd, θ_setpoint, robot: "Robot"):
        error = θ_setpoint - robot.θ
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.θ_setpoint = θ_setpoint
        self.error_last = error
        self.error_integral = 0.0
        self.u = self.kp * error


@dataclass
class Robot:
    def step(self, communication_step_size, controller: Controller):
        t_elapsed = 0.0

        while t_elapsed < communication_step_size:
            dθdt = self.ω
            dωdt = (
                self.K * self.i
                - self.b * self.ω
                - self.m * self.g * self.l * cos(self.θ)
            ) / self.J
            didt = (
                controller.u * self.V_abs - self.R * self.i - self.K * self.ω
            ) / self.L

            self.θ = self.θ + self.internal_step_size * dθdt
            self.ω = self.ω + self.internal_step_size * dωdt
            self.i = self.i + self.internal_step_size * didt

            t_elapsed += self.internal_step_size
            self._internal_θ = (
                [self.θ] if self._internal_θ is None else self._internal_θ + [self.θ]
            )

            if self._internal_t is None:
                self._internal_t = [t_elapsed]
            else:
                self._internal_t = self._internal_t + [
                    self._internal_t[-1] + self.internal_step_size
                ]

        assert (
            t_elapsed == communication_step_size
        ), f"unable to step for exactly {communication_step_size} "

    θ: float  # angle
    ω: float  # velocity
    i: float  # current
    V_abs = 12.0
    K = 7.45  # torque coefficient
    g = 9.81  # gravitational acceleration
    b = 5.0  # motor shaft friction
    m = 5.0  # mass of joint
    R = 0.15  # electrical resistance ?
    L = 0.036  # motor inductance
    l = 1.0  # length of joint
    J = 0.5 * (m * l**2)  # moment of intertia
    internal_step_size = 1 / 1024
    _internal_θ = None
    _internal_t = None


def run_cosim_for(controller: Controller, robot: Robot, t):
    communication_step_sizes = t[1:] - t[:-1]

    θs = [robot.θ]
    ts = [0.0]

    for communication_step_size in communication_step_sizes:
        controller_cur = deepcopy(controller)

        controller.step(communication_step_size, robot)

        robot.step(communication_step_size, controller_cur)

    θs += robot._internal_θ
    ts += robot._internal_t
    return ts, θs


def run():
    t_start = 0.0
    t_end = 10.0
    communication_step_size = 1 / 1024
    communication_step_size_coarse = 1 / 8  # 1.0

    t = np.arange(t_start, t_end + communication_step_size, communication_step_size)
    t_coarse = np.arange(
        t_start, t_end + communication_step_size_coarse, communication_step_size_coarse
    )

    robot = Robot(θ=0.0, ω=0.0, i=0.0)
    controller = Controller(kp=1.0, ki=0.1, kd=-0.01, θ_setpoint=1.0, robot=robot)

    t_internal, θ_internal = run_cosim_for(controller, deepcopy(robot), t)
    t_internal_coarse, θ_internal_coarse = run_cosim_for(
        controller, deepcopy(robot), t_coarse
    )

    fig, ax = plt.subplots()
    ax.axhline(controller.θ_setpoint, label="setpoint", color="red", linestyle="dotted")
    ax.plot(t_internal, θ_internal, label=r"$h_{internal} = h_{communication}$")
    ax.plot(
        t_internal_coarse,
        θ_internal_coarse,
        label=f"$h_{{communication}}={communication_step_size_coarse}$",
    )
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$\\theta(t)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/cosim/monolithic_vs_cosim_simulation.pdf")
    plt.show()


if __name__ == "__main__":
    run()
