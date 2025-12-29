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
    
def simulate_vehicle_speed(vehicle,mass, current_speed, throttle_input, braking_input, delta_time):
    # Constants and vehicle parameters (simplified)
    C_drag = 0.3  # Drag coefficient
    A_frontal = 2.0  # Frontal area (m^2)
    rho_air = 1.225  # Air density (kg/m^3)
    C_rolling = 0.01 # Rolling resistance coefficient
    max_engine_force = 5000 # Maximum available engine force (Newtons) - depends on gear/torque curve
        
    # Calculate forces
    # 1. Tractive Force (simplified from throttle input)
    # A realistic model would use engine torque curves and gear ratios
    tractive_force = max_engine_force * throttle_input

    # 2. Aerodynamic Drag Force (proportional to velocity squared)
    drag_force = 0.5 * rho_air * current_speed**2 * C_drag * A_frontal

    # 3. Rolling Resistance Force (simplified, can be considered constant at low speeds)
    rolling_resistance_force = mass * 9.81 * C_rolling # mass * gravity * coefficient

    # 4. Braking Force (simplified from braking input)
    braking_force = 10000 * braking_input # Max braking force (Newtons)

    # Net Force
    net_force = tractive_force - drag_force - rolling_resistance_force - braking_force

    # Acceleration (a = F / m)
    acceleration = net_force / mass


    if (braking_input > 0):
        acceleration = -a
        #print(f"Breaking for {vehicle} requested: current_speed:{current_speed}, braking_input:{braking_input} new_speed {new_speed} current_speed {current_speed}, acceleration {acceleration}")

        # Update Speed (using simple Euler integration)
        # New velocity = Old velocity + acceleration * time step
    new_speed = current_speed + (acceleration * delta_time)

    # Speed cannot be negative
    if new_speed < 0:
        new_speed = 0

    return new_speed, acceleration


##  Simulate the speed of the vehicle
##
def get_vehicle_speed(vehicle, current_speed, throttle, braking, time_step, faultinjection):
    if (current_speed >= 90):
        braking = 1
    else:    
        braking = round(random.betavariate(0.4,1))
        throttle = random.random()*4
    
    # The maximum throttle is 1
    if (throttle > 1.0):
        throttle = 1.0
        
    current_speed, acc = simulate_vehicle_speed(vehicle,1200, current_speed, throttle, braking, time_step)
    
#        if (tcurrent_speed <= 90):
#            current_speed, acc = tcurrent_speed, tacc
    
    #print(f"{vehicle} Speed: {current_speed:.2f} m/s, brak {braking:.2f} thro {throttle:.2f} Acceleration: {acc:.2f} m/s^2")
    return current_speed, braking

    # End 

# Compute teh safe distance

def get_safe_distance(current_speed):
    
    distance = (Speed_conversion * h * current_speed) + ((0.039 * math.sqrt(current_speed))/a)
    return distance

# Compute GAP distance between the two vehicle
def get_Gapdistance(previous_distance, host_speed, lead_speed):
    distance = previous_distance + ((lead_speed - host_speed) * time_step)
    return distance

# -------------------------------
# Lemma 4.1 Equation (Eq. 24)
# -------------------------------

def compute_v_thr(d, v_l, u, h, dt):
    """
    Compute v_thr at step k+1 given:
    - d_hat_k     : estimated gap d_k
    - v_l_hat_k   : estimated lead speed v_l,k
    - u           : deceleration parameter
    - h           : headway time
    - dt          : sampling time Δt
    """
    b = Speed_conversion * h + dt
    inside = (b**2) + ((0.156 / u) * (d + (v_l * dt)))
    if (inside < 0):
        inside (f"inside {inside:.2f} d_hat_k {d:.2f}, dt {dt:.2f}, v_l:{v_l:.2f}, h:{h}, a:{a}")
    #    v_thr = 0
    else:  
        v_thr = (u / 0.078) * (-b + np.sqrt(inside))
    
    return v_thr

def v_threshold(d, dt, v_l, h, a):
    term1 = 0.278 * h + dt
    term2 = term1**2 + (0.156 / a) * (d + v_l * dt)
    if (term2 < 0):
        print (f"term 2 {term2:.2f} d {d:.2f}, dt {dt:.2f}, v_l:{v_l:.2f}, h:{h}, a:{a}")
        
    return (a / 0.078) * (-term1 + np.sqrt(term2))

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


    # Example Usage:
# mass_of_car = 1200 kg, initial_speed = 0 m/s, time_step = 0.01 seconds
Hostcurrent_speed = 20.0
Leadcurrent_speed = 30.0
time_step = 0.01
throttle = 1.0 # Full throttle
braking = 0.0
h=2.0 #
a=3.4
Speed_conversion = 0.278 # Constant to convert speed from km/h to m/s

SimulationData=[]

kf = KF(Hostcurrent_speed, 10, 0.01, 0.1)

#Set the initial distance between the two vehicle to be the safe distance
safe_Distance = get_safe_distance(Hostcurrent_speed)  
gap_distance=1.1 * safe_Distance

ntimes =0
rtimes =0

for j in range(1000): # Run for 10 seconds (1000 steps of 0.01s)
    faultinjection = 0

    Hostcurrent_speed, braking = get_vehicle_speed("Host", Hostcurrent_speed, throttle, braking, time_step, faultinjection)
    #kf.predict()
    #kf.update(Hostcurrent_speed)
    
    
    Leadcurrent_speed, lbrake = get_vehicle_speed("Lead",Leadcurrent_speed, throttle, 0, time_step, faultinjection)

    safe_Distance = get_safe_distance(Hostcurrent_speed)    
    gap_distance = get_Gapdistance(gap_distance, Hostcurrent_speed, Leadcurrent_speed) 

    #threshold_speed = compute_v_thr(gap_distance, Leadcurrent_speed, a, h, time_step)

    threshold_speed = v_threshold(gap_distance,time_step, Leadcurrent_speed,h,a)

    ## Decide on the next speed
    if (gap_distance < safe_Distance):
        Crash = 1
        ntimes= ntimes+1
        
    else:
        Crash = 0
        
    if (Hostcurrent_speed > threshold_speed):
        speedRisk = 1
        rtimes= rtimes+1
    else:
        speedRisk = 0
    
    SimulationData.append([Hostcurrent_speed,Leadcurrent_speed,threshold_speed,safe_Distance,gap_distance,braking,Crash,speedRisk])    
    
    #print(f"{j} - Hostcurrent_speed:{Hostcurrent_speed:.2f}, Leadcurrent_speed {Leadcurrent_speed:.2f}, threshold_speed:{threshold_speed:.2f}, safe distance {safe_Distance:.2f}, gap_distance:{gap_distance:.2f}, Distance risk:{Crash}, speedRisk:{speedRisk}")
    
print(f"Safe distance is violated {ntimes} times")
print(f"Threshold speed is violated {rtimes} times")

for item in SimulationData:
    print(f"Record: {item}")
    