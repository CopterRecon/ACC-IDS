import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ================================
# Kalman Filter
# ================================
class KF:
    def __init__(self, x0, P0, Q, R):
        self.x = x0; self.P = P0; self.Q = Q; self.R = R
    def predict(self): self.P += self.Q
    def update(self, z):
        K = self.P/(self.P+self.R)
        self.x = self.x + K*(z-self.x)
        self.P = (1-K)*self.P
        return self.x
    
        
# ------------------------
# Unit conversions
# ------------------------
KMH_TO_MS = 1000.0 / 3600.0
MS_TO_KMH = 3600.0 / 1000.0

# ------------------------
# Vehicle dynamics (km/h I/O)
# ------------------------
def simulate_vehicle_speed_kmh(vehicle, mass, current_speed_kmh, throttle_input, braking_input, delta_time, u=3.4, a_max=1.5):  # a_max in m/s^2 (comfort accel)
    throttle = max(0.0, min(1.0, float(throttle_input)))
    brake    = max(0.0, min(1.0, float(braking_input)))
    dt = max(0.0, float(delta_time))
    
    v_ms = max(0.0, float(current_speed_kmh)) * KMH_TO_MS
    
    C_drag, A_frontal, rho_air, C_rolling = 0.30, 2.0, 1.225, 0.010
    max_engine_force = 5000.0
    
    tractive_force = max_engine_force * throttle
    drag_force = 0.5 * rho_air * (v_ms ** 2) * C_drag * A_frontal
    rolling_force = mass * 9.81 * C_rolling
    
    max_brake_force = mass * u
    brake_force = max_brake_force * brake
    
    net_force = tractive_force - drag_force - rolling_force - brake_force
    acc = net_force / mass
    
    # cap braking by pedal*u, cap accel by a_max
    if brake > 0:
        acc = max(acc, -(brake * u))
    acc = min(acc, a_max)
    
    v_new_ms = max(0.0, v_ms + acc * dt)
    if dt > 0:
        acc = (v_new_ms - v_ms) / dt
        
    return v_new_ms * MS_TO_KMH, acc
    
    
def get_vehicle_speed(vehicle, vspeed_kmh, throttle, braking, time_step, u=3.4):
    # vspeed is km/h
    if vspeed_kmh >= 110:
        braking = 1.0
        current_speed_kmh = vspeed_kmh
    else:
        braking = float(round(random.betavariate(0.4, 1)))  # 0 or 1
        throttle = min(1.0, random.random() * 4)            # cap at 1
    
        current_speed_kmh, acc_mps2 = simulate_vehicle_speed_kmh(
        vehicle, 1200, vspeed_kmh, throttle, braking, time_step, u=u
    )
    return current_speed_kmh, braking
    
    
    # ------------------------
    # Distances (meters) using km/h speeds
    # ------------------------
def get_safe_distance(vspeed_kmh, h, u):
    """
    Safe distance in meters:
    d = v*h + v^2/(2u)
    v in m/s, h in s, u in m/s^2
    """
    v_ms = vspeed_kmh * KMH_TO_MS
    return (v_ms * h) + (v_ms**2) / (2.0 * u)
    
    
def get_Gapdistance(previous_distance_m, host_speed_kmh, lead_speed_kmh, time_step):
    """
    Gap update in meters:
        d_{k+1} = d_k + (v_lead - v_host)*dt
    speeds in km/h converted to m/s
    """
    return previous_distance_m + ((lead_speed_kmh - host_speed_kmh) * KMH_TO_MS * time_step)
    
    
    # ------------------------
    # Recommended: make compute_v_thr exact with conversions
    # ------------------------
def compute_v_thr(d, v_l_kmh, u, h, dt):
    """
    Threshold host speed v_thr in km/h from quadratic:
        (KMH_TO_MS^2/(2u)) v^2 + (KMH_TO_MS*(h+dt)) v - (d + KMH_TO_MS*dt*v_l) = 0
    """
    p = (KMH_TO_MS**2) / (2.0 * u)
    b = KMH_TO_MS * (h + dt)
    c = -(d + KMH_TO_MS * dt * v_l_kmh)
    
    disc = b*b - 4.0*p*c
    disc = np.maximum(disc, 0.0)
    v_thr = (-b + np.sqrt(disc)) / (2.0 * p)
    return v_thr    
            
# -------------------------------
# Lemma 4.2 Equation (Eq. )
# -------------------------------
def compute_z_threshold(v_h_pred_k1, K_k1, d_hat_k, v_l_hat_k, u, h, dt):
    """
    Compute the threshold on the measurement z_{k+1} (measured host speed)
    such that the updated Kalman-filter estimate v_h(t+1|t+1) exceeds v_thr.

    Inputs:
    - v_h_pred_k1 : predicted host speed  v_h(t+1 | t)
    - K_k1        : Kalman gain at time k+1 (scalar, for speed update)
    - d_hat_k     : estimated gap d_k at time k
    - v_l_hat_k   : estimated lead speed v_l,k at time k
    - u           : deceleration parameter
    - h           : headway time
    - dt          : sampling time Δt

    Returns:
    - z_thr_k1    : minimum measurement z_{k+1} such that v_h(t+1|t+1) > v_thr
    """
    # 1) Compute threshold speed v_thr at k+1
    v_thr_k1 = compute_v_thr(d_hat_k, v_l_hat_k, u, h, dt)
    
    # 2) Compute measurement threshold z_{k+1}
    #    z_thr = [v_thr - (1 - K)*v_pred] / K
    z_thr_k1 = (v_thr_k1 - (1.0 - K_k1) * v_h_pred_k1) / K_k1
    
    return z_thr_k1


'''
     Plot the host speed threshold

'''
def plotSpeedThreshold():

    # Choose one lead speed, or multiple for comparison
    lead_speeds = [20, 70, 110]  # km/h
    
    # Plot v_thr as a function of gap d(t)
    d_min, d_max = 0, 80
    d_vals = np.linspace(d_min, d_max, 500)
    
    plt.figure()
    for v_l in lead_speeds:
        thr_vals = compute_v_thr(d_vals, v_l=v_l, u=a, h=h, dt=time_step)
        plt.plot(d_vals, thr_vals, label=f"Speed lead vehicle = {v_l} km/h")
    
    plt.xlabel("Current distance between the host and lead vehicles - d(t)  (m)")
    plt.ylabel("Threshold host speed  (km/h)")
    #plt.title("")
    plt.grid(True)
    plt.legend()
    plt.show()

    return

'''
    This is the code for Equation 17
'''
def g(v_kmh: np.ndarray, v_l, gapDistance, dt) -> np.ndarray:

    # Eq. (17) coefficients
    p = 0.039 / a
    b = 0.278 * (h + dt)
    c = -(gapDistance + 0.278 * dt * v_l)
    
    return p * v_kmh**2 + b * v_kmh + c


'''
    Check the safe distance
'''
def required_gap_eq17(v_h_kmh, v_l_kmh, u, h, dt):
    # meters, consistent with compute_v_thr()
    return (KMH_TO_MS*(h+dt)*v_h_kmh) + ((KMH_TO_MS**2)/(2*u))*(v_h_kmh**2) - (KMH_TO_MS*dt*v_l_kmh)

'''
    Controler for the lead vehicle
'''
def lead_controls(v_kmh, v_set_kmh=90.0, a_comfort=1.2, k_acc=0.12, u=3.4):
    err = v_set_kmh - v_kmh
    a_des = k_acc * err * KMH_TO_MS
    a_des = max(-u, min(a_comfort, a_des))
    
    if a_des >= 0:
        throttle = a_des / a_comfort
        brake = 0.0
    else:
        throttle = 0.0
        brake = (-a_des) / u
        
    return float(np.clip(throttle, 0, 1)), float(np.clip(brake, 0, 1))

'''
   Controler for the host vehicle
'''

def host_controls(v_host_kmh, v_lead_kmh, gap_m, u, h, dt,
                  cruise_kmh=120.0, a_comfort=1.5, k_acc=0.15):
    v_thr = compute_v_thr(gap_m, v_lead_kmh, u=u, h=h, dt=dt)
    v_target = min(cruise_kmh, v_lead_kmh, v_thr)
    
    # desired acceleration (m/s^2) from speed error (km/h)
    err = v_target - v_host_kmh
    a_des = k_acc * err * KMH_TO_MS   # convert km/h error to m/s then scale
    
    # cap to comfort accel/decel
    a_des = max(-u, min(a_comfort, a_des))
    
    # map accel to throttle/brake
    if a_des >= 0:
        throttle = a_des / a_comfort if a_comfort > 0 else 0.0
        brake = 0.0
    else:
        throttle = 0.0
        brake = (-a_des) / u if u > 0 else 0.0
        
    # extra safety if gap below required
    d_req = required_gap_eq17(v_host_kmh, v_lead_kmh, u=u, h=h, dt=dt)
    if gap_m < d_req:
        throttle = 0.0
        brake = max(brake, 0.8)
        
    return float(np.clip(throttle, 0, 1)), float(np.clip(brake, 0, 1))
'''

'''
def plotDistanceGapVSSpeedthreshold(v_l, gapDistance,dt ):

    # Eq. (17) coefficients
    p = 0.039 / a
    b = 0.278 * (h + dt)
    c = -(gapDistance + 0.278 * dt * v_l)
    # Compute positive root (threshold) of g(v)=0
    disc = b**2 - 4 * p * c
    if disc < 0:
        raise ValueError(f"Discriminant is negative ({disc}). Check parameters.")
    
    v_thr = (-b + np.sqrt(disc)) / (2 * p)

    # Plot over a host-speed range
    v = np.linspace(0, 120, 800)  # km/h (adjust as desired)

    plt.figure()
    plt.plot(v, g(v, v_l,gapDistance,dt), label="g(v) (Eq. 17)")
    plt.axhline(0)
    plt.axvline(v_thr, linestyle="--", label=f"v_thr ≈ {v_thr:.2f} km/h")
    plt.xlabel("Threshold host speed  (km/h)")
    plt.ylabel("Gap btw. current distance and safe distance - g(v)")
    #plt.title("Gap distance plotted as a quadratic in host speed")
    plt.grid(True)
    plt.legend()
    plt.show()

    return
    # Example Usage:
# mass_of_car = 1200 kg, , time_step = 0.01 seconds
Hostcurrent_speed = 25.0 
Leadcurrent_speed = 30.0 
time_step = 0.1   # in s
throttle = 1.0 # Full throttle
braking = 0.0
h=2.0 #
a=3.4
Speed_conversion = 0.278 # Constant to convert speed from km/h to m/s

SimulationData=[]

kf = KF(Hostcurrent_speed, 10, 0.01, 0.1)


# Example: compute v_thr at a specific d and v_l
example_d = 41.46068235294118
example_vl = 30
print("v_thr =", compute_v_thr(example_d, example_vl, u=a,h=h, dt=time_step), "km/h")

#plotSpeedThreshold()


#plotDistanceGapVSSpeedthreshold(v_l = 30, gapDistance = 41 , dt = time_step)


#Set the initial distance between the two vehicle to be the safe distance
safe_Distance = get_safe_distance(Hostcurrent_speed, u=a, h=h)  
gap_distance=1.1 * safe_Distance

print(f" Initial safe distance {safe_Distance} for speed {Hostcurrent_speed} Gap {gap_distance}")


ntimes =0
rtimes =0
nexceed=0

TwoCarsSpeedDataset=[]

SimulationData = []
ntimes = 0
rtimes = 0
nexceed = 0

for j in range(1000):
    
    # --- Lead vehicle control ---
    lead_throttle, lead_brake = lead_controls(Leadcurrent_speed, v_set_kmh=90.0)
    Leadcurrent_speed, lead_acc = simulate_vehicle_speed_kmh(
        "Lead", 1200, Leadcurrent_speed, lead_throttle, lead_brake, time_step, u=a
    )
    
    # --- Host (ACC) control ---
    host_throttle, host_brake = host_controls(
        Hostcurrent_speed, Leadcurrent_speed, gap_distance,
        u=a, h=h, dt=time_step, cruise_kmh=120.0
    )
    Hostcurrent_speed, host_acc = simulate_vehicle_speed_kmh(
        "Host", 1200, Hostcurrent_speed, host_throttle, host_brake, time_step, u=a
    )
    
    # --- Update gap (meters) ---
    gap_distance = get_Gapdistance(gap_distance, Hostcurrent_speed, Leadcurrent_speed, time_step)
    
    # --- Safety metrics ---
    d_req = required_gap_eq17(Hostcurrent_speed, Leadcurrent_speed, a, h, time_step)
    PotentialCrash = int(gap_distance < d_req)
    
    threshold_speed = compute_v_thr(gap_distance, Leadcurrent_speed, u=a, h=h, dt=time_step)
    speedRisk = int(Hostcurrent_speed > threshold_speed)
    
    # --- Counters ---
    ntimes += PotentialCrash
    rtimes += speedRisk
    
    # --- STORE DATA (this was missing) ---
    SimulationData.append([
        j,
        Hostcurrent_speed,
        Leadcurrent_speed,
        gap_distance,
        d_req,
        threshold_speed,
        host_throttle,
        host_brake,
        PotentialCrash,
        speedRisk
    ])
    
    if gap_distance < 2:
        print(f"Step {j}: Gap < 2 m -> Accident... Exit")
        break
    
print(f"Safe distance is violated {ntimes} times")
print(f"Threshold speed is violated {rtimes} times")
print(f"nexceed is violated {nexceed} times")


for item in SimulationData:
    print(f"Record: {item}")
    
    
plt.figure()

steps = [row[0] for row in SimulationData]
hostspeed = [row[1] for row in SimulationData]

plt.plot(steps, hostspeed, label="Host speed (km/h)")
plt.xlabel("Time step")
plt.ylabel("Speed (km/h)")
plt.grid(True)
plt.legend()
plt.show()


# Extract series from SimulationData
steps = [row[0] for row in SimulationData]
gap_m = [row[3] for row in SimulationData]
d_req_m = [row[4] for row in SimulationData]

host_kmh = [row[1] for row in SimulationData]
vthr_kmh = [row[5] for row in SimulationData]

# -----------------------
# Plot 1: Gap vs Required Gap
# -----------------------
plt.figure()
plt.plot(steps, gap_m,  label="Gap distance d (m)")
plt.plot(steps, d_req_m,  label="Required gap d_req (m)")
plt.xlabel("Time step")
plt.ylabel("Distance (m)")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------
# Plot 2: Host speed vs Threshold speed
# -----------------------
plt.figure()
plt.plot(steps, host_kmh,  label="Host speed v_h (km/h)")
plt.plot(steps, vthr_kmh, label="Threshold speed v_thr (km/h)")
plt.xlabel("Time step")
plt.ylabel("Speed (km/h)")
plt.grid(True)
plt.legend()
plt.show()
