import numpy as np
import matplotlib.pyplot as plt
import sys


def pendulum_ideal(x):
    g = 1.0  # gravity
    l = 1.0  # pendulum arm length
    θ, ω = x

    dθdt = ω
    dωdt = -(g / l) * np.sin(θ)
    return np.array((dθdt, dωdt))


def pendulum_friction(x):
    g = 1.0  # gravity
    l = 1.0  # pendulum arm length
    γ = 1.0  # friction co-efficient
    θ, ω = x

    dθdt = ω
    dωdt = -(g / l) * np.sin(θ) - γ * ω
    return dθdt, dωdt


def lotka_volterra(x):
    x, y = x
    a = 1.0
    b = 1.0
    c = 1.0
    d = 1.0
    dxdt = a * x - b * x * y
    dydt = c * x * y - d * y
    return dxdt, dydt


def run_lotka_volterra():
    mesh_step = 0.3
    x, y = np.meshgrid(
        np.arange(0.0, 5.0 + mesh_step, mesh_step),
        np.arange(0.0, 5.0 + mesh_step, mesh_step),
    )

    dxdt, dydt = lotka_volterra((x, y))

    fig, ax = plt.subplots()
    # ax.quiver(x, y, dxdt, dydt, angles="xy", scale_units="xy", scale=8)
    ax.streamplot(x, y, dxdt, dydt)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Vector Field: Lotka-Volterra")
    plt.savefig("figures/simulation/lotka_volterra_vector_field.pdf")
    plt.show()


def run_pendulum_solver():
    step_size_reference = 0.00001
    step_size = 0.1

    t_start = 0.0
    t_end = 4 * np.pi

    # simulate reference
    t_reference = np.arange(t_start, t_end + step_size_reference, step_size_reference)
    x_cur = np.array((np.pi / 4, 00))  # initial state
    xs_reference = [x_cur]

    for _ in t_reference[1:]:
        x_new = x_cur + step_size_reference * pendulum_ideal(x_cur)
        xs_reference.append(x_new)
        x_cur = x_new

    xs_reference = np.stack(xs_reference, axis=1)  # (n_states, n_steps)

    # simulate Euler
    t = np.arange(t_start, t_end + step_size, step_size)

    x_cur = np.array((np.pi / 4, 00))  # initial state
    xs_euler = [x_cur]

    for _ in t[1:]:
        x_new = x_cur + step_size * pendulum_ideal(x_cur)
        xs_euler.append(x_new)
        x_cur = x_new

    xs_euler = np.stack(xs_euler, axis=1)  # (n_states, n_steps)

    # simulate midpoint

    x_cur = np.array((np.pi / 4, 00))  # initial state
    xs_midpoint = [x_cur]

    for _ in t[1:]:
        x_new = x_cur + step_size * pendulum_ideal(
            x_cur + step_size / 2 * pendulum_ideal(x_cur)
        )
        xs_midpoint.append(x_new)
        x_cur = x_new

    xs_midpoint = np.stack(xs_midpoint, axis=1)  # (n_states, n_steps)

    # plotting

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(
        t_reference,
        xs_reference[0],
        label=f"FE (step-size = {step_size_reference})",
        c="black",
        linewidth=2,
    )
    ax1.plot(
        t,
        xs_euler[0],
        label=f"FE (step-size = {step_size})",
        c="blue",
        linestyle="dotted",
        linewidth=2,
    )
    ax1.plot(
        t,
        xs_midpoint[0],
        label=f"MID (step-size = {step_size})",
        c="red",
        linestyle="dotted",
        linewidth=2,
    )
    ax1.set_ylabel(rf"$\theta$ [rad]")
    ax2.plot(t_reference, xs_reference[1], label="reference", c="black", linewidth=2)
    ax2.plot(
        t,
        xs_euler[1],
        c="blue",
        linestyle="dotted",
        linewidth=2,
    )
    ax2.plot(
        t,
        xs_midpoint[1],
        c="red",
        linestyle="dotted",
        linewidth=2,
    )
    ax2.set_ylabel(rf"$\omega$ [rad/s]")
    ax2.set_xlabel("t [s]")
    ax1.legend()
    plt.tight_layout()
    plt.show()


def run_pendulum_phase():
    mesh_step = 0.3
    θ, ω = np.meshgrid(
        np.arange(-np.pi, np.pi + mesh_step, mesh_step),
        np.arange(-np.pi, np.pi + mesh_step, mesh_step),
    )
    θ_wrapped, ω_wrapped = np.meshgrid(
        np.arange(-2 * np.pi, 2 * np.pi + mesh_step, mesh_step),
        np.arange(-2 * np.pi, 2 * np.pi + mesh_step, mesh_step),
    )

    dθdt_ideal, dωdt_ideal = pendulum_ideal((θ, ω))
    dθdt_ideal_wrapped, dωdt_ideal_wrapped = pendulum_ideal((θ_wrapped, ω_wrapped))
    dθdt_friction, dωdt_friction = pendulum_friction((θ, ω))

    fig, ax = plt.subplots()
    # ax.quiver(θ, ω, dθdt_ideal, dωdt_ideal, angles="xy", scale_units="xy", scale=5)
    ax.streamplot(θ, ω, dθdt_ideal, dωdt_ideal)
    ax.set_xlabel(rf"$\theta$")
    ax.set_ylabel(rf"$\omega$")
    ax.set_title("Vector Field: Ideal Pendulum")
    plt.savefig("figures/simulation/pendulum_ideal_vector_field.pdf")

    fig, ax = plt.subplots()
    ax.quiver(
        θ_wrapped,
        ω_wrapped,
        dθdt_ideal_wrapped,
        dωdt_ideal_wrapped,
        angles="xy",
        scale_units="xy",
        scale=10,
    )
    ax.set_xlim(-2 * np.pi, 2 * np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(rf"$\theta$")
    ax.set_ylabel(rf"$\omega$")
    ax.set_title("Vector Field: Ideal Pendulum")
    plt.savefig("figures/simulation/pendulum_ideal_vector_field_wrapped.pdf")

    fig, ax = plt.subplots()
    # ax.quiver(
    #     θ, ω, dθdt_friction, dωdt_friction, angles="xy", scale_units="xy", scale=5
    # )
    ax.streamplot(θ, ω, dθdt_friction, dωdt_friction)
    ax.set_xlabel(rf"$\theta$")
    ax.set_ylabel(rf"$\omega$")
    ax.set_title("Vector Field: Pendulum with friction")
    plt.savefig("figures/simulation/pendulum_friction_vector_field.pdf")

    plt.show()


def simulate_with_stepsize(f, x0, t_start, t_stop, step_size):
    t = np.arange(t_start, t_stop + step_size, step_size)
    x_euler = [x0]
    x_midpoint = [x0]

    for _ in t[1:]:
        x_euler.append(x_euler[-1] + step_size * f(x_euler[-1]))
        x_midpoint.append(
            x_midpoint[-1]
            + step_size * f(x_midpoint[-1] + step_size / 2 * x_midpoint[-1])
        )

    x_exact = np.array([np.e**t for t in t])
    x_euler = np.stack(x_euler)
    x_midpoint = np.stack(x_midpoint)

    return t, x_exact, x_euler, x_midpoint


def run_convergence():
    def f(x):
        return x

    step_size_min = 2**-25
    step_size_max = 1.0
    n_steps_sizes_in_sweep = 5
    step_size_increment = (step_size_max - step_size_min) / n_steps_sizes_in_sweep
    t_start = 0.0
    t_end = 0.1

    def f(x):
        return x

    x0 = np.array((1.0))

    step_sizes = np.arange(
        step_size_min, step_size_max + step_size_increment, step_size_increment
    )

    errors = {}

    for h in step_sizes:
        t_cur = t_start
        x_cur = x0
        while t_cur < t_end:
            x_cur = x_cur + h * f(x_cur)
            t_cur += h

        # assert t_cur == t_end, "simulation did not stop at the desired end time"

        x_true = np.e**t_cur
        errors[h] = abs(x_cur - x_true)

    fig, ax = plt.subplots()
    ax.loglog(list(errors), list(errors.values()), label="euler")
    # ax.loglog(np.array(list(errors)), errors_midpoint, label="midpoint")
    ax.set_ylabel("Mean-squared error")
    ax.set_xlabel("step-size")
    plt.legend()
    plt.show()


def run_integrator():
    step_size = 0.001
    step_size_coarse = 0.3
    t_start = 0.0
    t_end = 10.0
    t = np.arange(t_start, t_end + step_size, step_size)
    t_coarse = np.arange(t_start, t_end + step_size_coarse, step_size_coarse)

    # fine
    x_cur = np.array((0.5, 0.0))
    x = [x_cur]

    for _ in t[1:]:
        x_new = x_cur + step_size * pendulum_ideal(x_cur)
        x.append(x_new)
        x_cur = x_new

    x = np.stack(x, axis=1)  # (n_states, n_steps)

    # coarse
    x_cur = np.array((0.5, 0.0))
    x_coarse = [x_cur]
    x_straight = []
    n_steps_per_coarse = step_size_coarse // step_size

    for _ in t_coarse[1:]:
        x_new = x_cur + step_size_coarse * pendulum_ideal(x_cur)
        x_coarse.append(x_new)
        x_cur = x_new

    x_coarse = np.stack(x_coarse, axis=1)  # (n_states, n_steps_coarse)

    # plotting
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(t, x[0])
    ax1.scatter(t_coarse, x_coarse[0], marker="x", c="red")

    ax2.plot(t, x[1])
    ax1.set_ylabel(rf"$\theta$ [rad]")
    ax2.set_ylabel(rf"$\omega$ [rad/s]")
    ax2.set_xlabel("t [s]")
    plt.show()


if __name__ == "__main__":
    # run_lotka_volterra()
    # run_convergence()
    # run_pendulum_solver()
    run_pendulum_phase()
