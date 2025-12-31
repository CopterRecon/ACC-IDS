import numpy as np
import matplotlib.pyplot as plt

# Parameters
u = 3.4       # m/s^2 (deceleration)
h = 2.0       # s   (headway time)
dt = 1.0      # s   (time step Δt)
d = 170.0     # m   (current distance d(t))

# ---------------------------------------------------
# Safe distance function
# d_safe(t) = 0.278*h*v_h + (0.039*v_h^2)/u
# ---------------------------------------------------
def safe_distance(v_h):
    return 0.278 * h * v_h + (0.039 * v_h**2) / u


def velocity_Threshold(v_l_vals):

    # Precompute common term
    b = 0.278 * h + dt
    
    # v_thr formula:
    # v_thr = (u/0.078) * [ -b + sqrt( b^2 + (0.156/u) * (d + v_l * dt) ) ]
    inside_sqrt = b**2 + (0.156 / u) * (d + v_l_vals * dt)
    return (u / 0.078) * (-b + np.sqrt(inside_sqrt))
    

# -----------------------------
# Plot v_thr vs v_l
# -----------------------------
# v_l range (leader speed)
v_l_vals = np.linspace(0, 150, 300)
v_thr_vals = velocity_Threshold(v_l_vals)
plt.figure(figsize=(8, 5))
plt.plot(v_l_vals, v_thr_vals, label=r"$v_{\mathrm{thr}}$")
plt.xlabel("Leader speed $v_l$")
plt.ylabel(r"$v_{\mathrm{thr}}$")
plt.title(r"$v_{\mathrm{thr}}$ vs $v_l$  (with $d(t)=170$)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Example usage:
# print(safe_distance(100))   # Safe distance at v_h(t)=100