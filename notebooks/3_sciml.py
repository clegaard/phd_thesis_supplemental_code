import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap, random, tree_map
from jax.lax import scan
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

key = random.PRNGKey(1)


def solve_euler(f, t, y0, *args, **kwargs):
    step_sizes = t[1:] - t[:-1]
    y_cur = y0
    Y = [y_cur]

    for t, step_size in zip(t[1:], step_sizes):
        dydt = f(t, y_cur, *args)
        y_new = y_cur + step_size * dydt
        Y.append(y_new)
        y_cur = y_new

    return jnp.stack(Y, axis=1)


def solve_euler_scan(f, t, y0, *args, **kwargs):
    step_sizes = t[1:] - t[:-1]

    def f_scan(y_cur, t_and_step):
        t, step_size = t_and_step
        dydt = f(t, y_cur, *args)
        y_new = y_cur + step_size * dydt
        return y_new, y_new

    _, Y = scan(f_scan, init=y0, xs=(t[1:], step_sizes))
    Y = jnp.concatenate((y0.reshape(-1, 1), Y.T), axis=1)
    return Y


def f(t, x, params):
    θ, ω = x
    dθdt = ω
    dωdt = -params["g"] / params["l"] * jnp.sin(θ)
    return jnp.array((dθdt, dωdt))


def run_parameter_estimation():
    h = 0.001
    t_train_start = 0.0
    t_train_end = h
    t_train = jnp.arange(t_train_start, t_train_end + h, h)
    t_validation_start = t_train_start
    t_validation_end = 4 * jnp.pi
    t_validation = jnp.arange(t_validation_start, t_validation_end + h, h)

    x0 = jnp.array((-0.5, 0.0))

    solve_euler_scan_grid = vmap(solve_euler_scan, (None, None, 0, None))

    res = 0.1

    θ_grid, ω_grid = jnp.meshgrid(
        jnp.arange(-jnp.pi, jnp.pi + res, res),
        jnp.arange(-jnp.pi, jnp.pi + res, res),
    )
    θ_grid = θ_grid.reshape(-1)
    ω_grid = ω_grid.reshape(-1)
    x0_grid = jnp.stack((θ_grid, ω_grid), axis=1)

    def update(x_target, t, x0, params):
        def loss(params):
            y_predicted = solve_euler_scan(f, t, x0, params)
            return jnp.linalg.norm(x_target - y_predicted)

        loss, grads = value_and_grad(loss)(params)

        params = tree_map(lambda p, g: p - lr * g, params, grads)
        return loss, params

    def update_grid(x_target, t, x0, params):
        def loss(params):
            y_predicted = solve_euler_scan_grid(f, t, x0, params)
            return jnp.linalg.norm(x_target - y_predicted)

        loss, grads = value_and_grad(loss)(params)

        params = tree_map(lambda p, g: p - lr * g, params, grads)
        return loss, params

    parameters_true = {"g": 1.0, "l": 1.0}
    parameters_estimated = {"g": 1.0, "l": 0.5}

    x_true_grid = solve_euler_scan_grid(f, t_train, x0_grid, parameters_true)

    n_epochs = 10000
    lr = 0.001  # learning rate
    losses = []

    update = jit(update_grid)

    for _ in tqdm(range(n_epochs)):
        # value, parameters_estimated = update(x_true, t, x0, parameters_estimated)
        value, parameters_estimated = update(
            x_true_grid, t_train, x0_grid, parameters_estimated
        )
        losses.append(value)

    x_true_validation = solve_euler_scan(f, t_validation, x0, parameters_true)
    x_predicted_validation = solve_euler_scan(f, t_validation, x0, parameters_estimated)

    fig, ax = plt.subplots()
    ax.plot(t_validation, x_true_validation[0], label=rf"$\theta$", color="blue")
    ax.plot(t_validation, x_true_validation[1], label=rf"$\omega$", color="orange")
    ax.plot(
        t_validation,
        x_predicted_validation[0],
        label=rf"$\hat{{\theta}}$",
        color="blue",
        linestyle="dotted",
    )
    ax.plot(
        t_validation,
        x_predicted_validation[1],
        label=rf"$\hat{{\omega}}$",
        color="orange",
        linestyle="dotted",
    )
    ax.set_xlabel("t[s]")
    ax.set_ylabel("x(t)")
    ax.legend()
    ax.set_title(
        rf"Pendulum $\hat{{g}}$={parameters_estimated['g']:.2f},$\hat{{l}}$={parameters_estimated['l']:.2f}"
    )
    plt.savefig("./figures/sciml/parameter_estimation_pendulum_simulation.pdf")

    fig, ax = plt.subplots()
    ax.semilogy(losses)
    ax.set_xlabel("epoch")
    _ = ax.set_ylabel("loss(epoch)")
    plt.savefig("./figures/sciml/parameter_estimation_pendulum_losses.pdf")

    plt.show()


def run_phase_plot():
    res = 0.1
    g = 1.0
    l = 1.0
    xx, yy = jnp.meshgrid(
        jnp.arange(-jnp.pi, jnp.pi + res, res),
        jnp.arange(-jnp.pi, jnp.pi + res, res),
    )

    dx, dy = f((xx, yy), g, l)

    fig, ax = plt.subplots()

    xx = np.asarray(xx)
    yy = np.asarray(yy)

    ax.streamplot(xx, yy, dx, dy)
    ax.set_xlabel("θ")
    ax.set_ylabel("ω")

    plt.show()


if __name__ == "__main__":
    run_parameter_estimation()
