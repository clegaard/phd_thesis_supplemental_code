from jax import jit, value_and_grad, random, vmap
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt


def loss(x, y, a, b):
    y_predicted = a * x + b
    return jnp.linalg.norm(y - y_predicted)


def update(x, y, a, b):
    learning_rate = 0.0001

    value, (dlda, dldb) = value_and_grad(loss, argnums=(2, 3))(x, y, a, b)
    a -= learning_rate * dlda
    b -= learning_rate * dldb
    return value, a, b


def run():
    key = random.PRNGKey(1)

    a = 1.0
    b = 2.0
    x = jnp.arange(-3.0, 3.0, 0.001)
    e = random.normal(key, x.shape) * 1e-1
    y = a * x + b + e

    update_jit = jit(update)

    a_estimate = -5.0  # 0.5
    b_estimate = 5.0  # 1.0
    n_epochs = 1000
    losses = []

    ab_grid = jnp.meshgrid(jnp.arange(-5.0, 5.0, 0.1), jnp.arange(-5.0, 5.0, 0.1))

    loss_grid = vmap(vmap(lambda a, b: loss(x, y, a, b), (0, 0)), (0, 0))(*ab_grid)

    a_estimates = []
    b_estimates = []

    for i in tqdm(range(n_epochs)):
        value, a_estimate, b_estimate = update_jit(x, y, a_estimate, b_estimate)
        a_estimates.append(a_estimate)
        b_estimates.append(b_estimate)
        losses.append(value)
    y_estimate = a_estimate * x + b_estimate

    fig, ax = plt.subplots()
    ax.plot(x, y, label="f(x)")
    ax.plot(x, y_estimate, label=rf"$\hat{{f}}(x)$")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    fig, ax = plt.subplots()
    ax.semilogy(jnp.array(losses))
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss(epoch)")
    print(f"a: {a_estimate}, b: {b_estimate}")

    fig, ax = plt.subplots()
    im = ax.imshow(loss_grid, extent=(-5, 5, -5, 5), origin="lower")
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    fig.colorbar(im)
    ax.plot(a_estimates, b_estimates, color="red")
    ax.set_title("$\mathcal{L}(a,b)$")

    plt.show()


if __name__ == "__main__":
    run()
