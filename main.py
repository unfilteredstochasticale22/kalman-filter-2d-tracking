import numpy as np
import matplotlib.pyplot as plt
import os

# Create folder for plots
os.makedirs("plots", exist_ok=True)

N = 100          # number of time steps
dt = 1.0         # time step

true_pos = np.array([0.0,0.0])
true_vel = np.array([1.0,0.5])

measurement_std = np.array([1,1.5])  # sensor noise standard deviation

true_positions = []
measurements = []

for k in range(N):
    true_pos = true_pos + true_vel * dt

    noisy_measurement = true_pos + measurement_std * np.random.randn(2)

    true_positions.append(true_position)
    measurements.append(noisy_measurement)

true_positions = np.array(true_positions)

measurements = np.array(measurements)
x = np.array([[0.0,0.0],
              [0.0,0.0]])

# Initial covariance
# Large values mean high initial uncertainty
P = np.eye(4) * 1000.0

# State transition matrix
# position_next = position + velocity * dt
# velocity_next = velocity
A = np.array([[1.0,0, dt,0],
              [0.0,1.0, 0,dt],
              [0.0,0.0, 1.0,0],
             [0.0,0.0, 0.0,1.0]])

H = np.array([[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0]])


Q=np.eye(4)*0.01
R = np.diag(measurement_std ** 2)
I = np.eye(4)

for z in measurements:
z = z.reshape(2, 1)
x = A @ x
    P = A @ P @ A.T + Q
 y = z - H @ x                      # innovation / residual
    S = H @ P @ H.T + R                # innovation covariance
    K = P @ H.T @ np.linalg.inv(S)     # Kalman gain

    x = x + K @ y
    P = (I - K @ H) @ P

    position_estimate = [x[0, 0], x[1, 0]]
velocity_estimate = [x[2, 0], x[3, 0]]

estimated_positions = np.array(estimated_positions)
estimated_velocities = np.array(estimated_velocities)

time = np.arange(N) * dt
# -----------------------------
# 5. Plot 2D position tracking
# -----------------------------

plt.figure(figsize=(6, 6))

plt.plot(
    true_positions[:, 0],
    true_positions[:, 1],
    label="True trajectory"
)

plt.scatter(
    measurements[:, 0],
    measurements[:, 1],
    s=15,
    label="Noisy measurements"
)

plt.plot(
    estimated_positions[:, 0],
    estimated_positions[:, 1],
    label="Kalman estimate"
)

plt.xlabel("x position")
plt.ylabel("y position")
plt.title("2D Kalman Filter: Position Tracking")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig("plots/trajectory_tracking.png")
plt.show()


# -----------------------------
# 6. Plot velocity estimate
# -----------------------------

plt.figure(figsize=(10, 4))

plt.plot(
    time,
    np.ones(N) * true_velocity[0],
    label="True vx"
)

plt.plot(
    time,
    estimated_velocities[:, 0],
    label="Estimated vx"
)

plt.plot(
    time,
    np.ones(N) * true_velocity[1],
    label="True vy"
)

plt.plot(
    time,
    estimated_velocities[:, 1],
    label="Estimated vy"
)

plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("2D Kalman Filter: Velocity Estimate")
plt.grid(True)
plt.legend()
plt.savefig("plots/velocity_estimate.png")
plt.show()


# -----------------------------
# 7. Plot estimation error
# -----------------------------

position_error = estimated_positions - true_positions
position_error_norm = np.linalg.norm(position_error, axis=1)

plt.figure(figsize=(10, 4))
plt.plot(time, position_error_norm)

plt.xlabel("Time")
plt.ylabel("Position error norm")
plt.title("2D Position Estimation Error")
plt.grid(True)
plt.savefig("plots/position_error_norm.png")
plt.show()


# -----------------------------
# 8. Print results
# -----------------------------

print("Final true position:", true_positions[-1])
print("Final estimated position:", estimated_positions[-1])

print("True velocity:", true_velocity)
print("Final estimated velocity:", estimated_velocities[-1])

print("Mean position error norm:", np.mean(position_error_norm))
print("Final position error norm:", position_error_norm[-1])
