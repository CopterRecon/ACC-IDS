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
    
def simulate_vehicle_speed(mass, current_speed, throttle_input, braking_input, delta_time):
    # Constants and vehicle parameters (simplified)
    C_drag = 0.3  # Drag coefficient
    A_frontal = 2.0  # Frontal area (m^2)
    rho_air = 1.225  # Air density (kg/m^3)
    C_rolling = 0.01 # Rolling resistance coefficient
    max_engine_force = 5000 # Maximum available engine force (Newtons) - depends on gear/torque curve

    if (braking_input > 0):
        print(f"Breaking requested: current_speed:{current_speed}, braking_input:{braking_input}")
        throttle_input = 0
        
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

    # Update Speed (using simple Euler integration)
    # New velocity = Old velocity + acceleration * time step
    new_speed = current_speed + acceleration * delta_time

    # Speed cannot be negative
    if new_speed < 0:
        new_speed = 0

    return new_speed, acceleration


##  Simulate the speed of the vehicle
##
def get_vehicle_speed(vehicle, current_speed, throttle, braking, time_step, faultinjection):
    if (current_speed >= 90) or (braking ==1):
        braking = 1
    else:    
        braking = round(random.betavariate(0.4,1))
        throttle = random.random()*6
    
    current_speed, acc = simulate_vehicle_speed(1200, current_speed, throttle, braking, time_step)
    
#        if (tcurrent_speed <= 90):
#            current_speed, acc = tcurrent_speed, tacc
    
    #print(f"{vehicle} Speed: {current_speed:.2f} m/s, brak {braking:.2f} thro {throttle:.2f} Acceleration: {acc:.2f} m/s^2")
    return current_speed

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
def v_threshold(d, dt, v_l, h, a):
    term1 = 0.278 * h + dt
    term2 = term1**2 + (0.156 / a) * (d + v_l * dt)
    if (term2 < 0):
        print (f"term 2 {term2:.2f} d {d:.2f}, dt {dt:.2f}, v_l:{v_l:.2f}, h:{h}, a:{a}")
    
    return (a / 0.078) * (-term1 + np.sqrt(term2))



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


kf = KF(Hostcurrent_speed, 10, 0.01, 0.1)

#Set the initial distance between the two vehicle to be the safe distance
safe_Distance = get_safe_distance(Hostcurrent_speed)  
gap_distance=1.1 * safe_Distance

for j in range(1000): # Run for 10 seconds (1000 steps of 0.01s)
    faultinjection = 0

    Hostcurrent_speed = get_vehicle_speed("Host", Hostcurrent_speed, throttle, braking, time_step, faultinjection)

    Leadcurrent_speed = get_vehicle_speed("Lead",Leadcurrent_speed, throttle, 0, time_step, faultinjection)

    safe_Distance = get_safe_distance(Hostcurrent_speed)    
    gap_distance = get_Gapdistance(gap_distance, Hostcurrent_speed, Leadcurrent_speed) 
    
    threshold_speed = v_threshold(gap_distance, time_step, Leadcurrent_speed, h, a)

    ## Decide on the next speed
    if (gap_distance < safe_Distance):
        braking = 1
    else:
        braking = 0
        
    if (Hostcurrent_speed > threshold_speed):
        speedRisk = 1
    else:
        speedRisk = 0
    
    print(f"{j} - Hostcurrent_speed:{Hostcurrent_speed:.2f}, Leadcurrent_speed {Leadcurrent_speed:.2f}, threshold_speed:{threshold_speed:.2f}, safe distance {safe_Distance:.2f}, gap_distance:{gap_distance:.2f}, Distance risk:{braking}, speedRisk:{speedRisk}")
        
        
    