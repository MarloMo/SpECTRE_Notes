# Enable interactive plot
# %matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import numpy as np

sigma = 1.0
mu = 5.0
amplitude = 1.0


def initial_wave_profile_psi(x):
    # The psi field, the scalar field itself.

    return amplitude / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 /
                                                             (2 * sigma**2))


def initial_wave_profile_pi(x):
    # The pi field, the conjugate momentum. Represents the time derivative.

    return 0 * x  #amplitude / (sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))


def initial_wave_profile_phi(x):
    # The phi field. Represents the spatial derivative.

    return -amplitude * (x - mu) / (sigma * np.sqrt(2 * np.pi)) * np.exp(
        -(x - mu)**2 / (2 * sigma**2))


def faster_finite_differentiation(xs, profile):
    #Second order even at the boundaries.
    #assume regular separation:
    delta_x = xs[1] - xs[0]
    derivative = np.array([
        (-profile[2] + 4.0 * profile[1] - 3.0 * profile[0]) / (2.0 * delta_x)
    ])
    derivative = np.append(derivative,
                           (profile[2:] - profile[:-2]) / (2.0 * delta_x))
    derivative = np.append(derivative, [
        (profile[-3] - 4.0 * profile[-2] + 3.0 * profile[-1]) / (2.0 * delta_x)
    ])
    return derivative


def updated_psi(psi, pi, phi, delta_t):
    return psi + -pi * delta_t


def constraint_error(psi, phi, xs):
    # phi should be equal to the spatial derivative of psi at all times
    return faster_finite_differentiation(xs, psi) - phi


def updated_psi_second_order_in_time(psi, pi, phi, delta_t, previous_psi):
    return 1.0 / 3.0 * (-2.0 * pi * delta_t + 4.0 * psi - previous_psi)


def updated_pi_LF(psi, pi, phi, delta_t, apply_BCs=False):
    dphi_dx = faster_finite_differentiation(xs, phi)
    if not apply_BCs:
        new_pi = [pi[0] - dphi_dx[0] * delta_t]
    if apply_BCs:
        new_pi = [pi[0]]
    for i in range(1, len(psi) - 1):
        new_pi.append(0.5 * (pi[i - 1] + pi[i + 1]) - dphi_dx[i] * delta_t)
    if not apply_BCs:
        new_pi.append(pi[-1] - dphi_dx[-1] * delta_t)
    if apply_BCs:
        new_pi.append(pi[0])
    return np.array(new_pi)


def updated_phi_LF(psi, pi, phi, delta_t, damping=False, apply_BCs=False):
    gamma_2 = 0.01
    dpi_dx = faster_finite_differentiation(xs, pi)
    new_phi = [phi[0] - dpi_dx[0] * delta_t]
    if apply_BCs:
        new_phi = [phi[1]]
    for i in range(1, len(psi) - 1):
        new_phi.append(0.5 * (phi[i - 1] + phi[i + 1]) - dpi_dx[i] * delta_t)
    if not apply_BCs:
        new_phi.append(phi[-1] - dpi_dx[-1] * delta_t)
    if apply_BCs:
        new_phi.append(-phi[1])
    if damping == False:
        return np.array(new_phi)
    if damping == True:
        return np.array(new_phi) + gamma_2 * constraint_error(psi, phi, xs)


## Set up discretized fields:
N = 201
xs = np.linspace(-5, 15, N)
psis = initial_wave_profile_psi(xs)
pis = initial_wave_profile_pi(xs)
phis = initial_wave_profile_phi(xs)
constraints = constraint_error(psis, phis, xs)
delta_x = xs[1] - xs[0]
v = 1.0
delta_t = 0.1

# Useful quantities:
grid_speed = delta_x / delta_t
courant_factor = v / grid_speed
crossing_time = N * delta_x / v / 2.0
final_time = crossing_time
num_timesteps = int(final_time / delta_t)
print(courant_factor)

fig, ax = plt.subplots()
ax.set_xlim(-5, 15)
ax.set_ylim(-0.75, 0.75)

line1, = ax.plot([])
line2, = ax.plot([])
line3, = ax.plot([])
line4, = ax.plot([])
line5, = ax.plot([])
line6, = ax.plot([])

psis_all = [psis]
pis_all = [pis]
phis_all = [phis]
temp_psis = [psis]
temp_pis = [pis]
temp_phis = [phis]
constraints_all = [constraints]


def animate(frame_num):
    # set damping function to True to see the damping effect
    damping = True
    apply_BCs = True
    temp_psis[0] = updated_psi(psis_all[0], pis_all[0], phis_all[0], delta_t)
    temp_pis[0] = updated_pi_LF(psis_all[0], pis_all[0], phis_all[0], delta_t,
                                apply_BCs)
    temp_phis[0] = updated_phi_LF(psis_all[0], pis_all[0], phis_all[0],
                                  delta_t, damping, apply_BCs)
    psis_all[0] = temp_psis[0]
    pis_all[0] = temp_pis[0]
    phis_all[0] = temp_phis[0]
    if (frame_num == 50):
        pis_all[0] = -pis_all[0]
    line1.set_data((xs, psis_all[0]))
    line2.set_data((xs, pis_all[0]))
    line3.set_data((xs, phis_all[0]))
    line4.set_data((xs, constraints_all[0]))
    line5.set_data((xs, pis_all[0] + phis_all[0]))
    line6.set_data((xs, pis_all[0] - phis_all[0]))
    return [line1, line2, line3, line4, line5, line6]


anim = animation.FuncAnimation(fig,
                               animate,
                               frames=100,
                               interval=50,
                               blit=True)
plt.show()
