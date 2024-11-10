from numpy import ones, exp, diff
from pydrying.dry import thin_layer, material
import matplotlib.pyplot as plt
import numpy as np

# Define thermophysical properties
def Diff(T, X):
    return 1e-9 * ones(len(T))  # Diffusion coefficient

def aw(T, X):
    return (1.0 - exp(-0.6876 * (T + 45.5555) * X * X))  # Sorption isotherm

def Lambda_const(T, X, Lambda_val=0.02):
    return Lambda_val  # Lambda passed as argument to modify during experiments

# Common geometric properties
material_shape = 0  # Flat material
caracteristic_length = 1e-2  # 1 cm thickness
drying_time = 7200*5  # 2 hours in seconds
heat_transfer_coefficient = 25  # W/m^2/K

# Create a function to handle the simulation and return results
def run_drying_simulation(label, air_conditions, material_params, Lambda_val=0.02):
    # Define the drying material
    drying_material = material(Diff=Diff, aw=aw,
                               Lambda=lambda T, X: Lambda_const(T, X, Lambda_val),
                               m=material_params['shape'],
                               L=material_params['length'],
                               X0=material_params['X0'])

    # Solve the thin-layer drying problem
    problem = thin_layer(material=drying_material, air=air_conditions,
                         h=material_params['heat_transfer'],
                         tmax=material_params['drying_time'])
    problem.solve()

    # Debugging output to ensure parameters are varying correctly
    print(f"Running Simulation: {label}, Air: {air_conditions}, Shape: {material_params['shape']}, "
          f"Lambda: {Lambda_val}, Initial Moisture: {material_params['X0']}, "
          f"Heat Transfer: {material_params['heat_transfer']}")

    return problem

# Plot Moisture Content Over Time
def plot_moisture_content(problem, label):
    plt.plot(problem.res.t, problem.res.Xmoy, label=label)

# Plot Surface Temperature Over Time
def plot_surface_temperature(problem, label):
    plt.plot(problem.res.t, problem.res.T[-1, :], label=label)

# Plot Drying Rate (derivative of moisture content)
def plot_drying_rate(problem, label):
    drying_rate = -np.gradient(problem.res.Xmoy, problem.res.t)  # Calculate drying rate
    plt.plot(problem.res.t[:len(drying_rate)], drying_rate, label=label)

# Plot Heat Flux (heat transfer coefficient * temperature difference)
def plot_heat_flux(problem, air_conditions, material_params, label):
    heat_flux = material_params['heat_transfer'] * (air_conditions['T'] - problem.res.T[-1, :])  # Heat flux calculation
    plt.plot(problem.res.t, heat_flux, label=label)

# Experiment 1: Varying Initial Air Temperature
air_temperatures = [30, 60, 100]  # Air temperatures in °C
material_params_base = {'shape': material_shape, 'length': caracteristic_length,
                        'heat_transfer': heat_transfer_coefficient,
                        'drying_time': drying_time, 'X0': 0.5}  # 50% initial moisture

# Initialize figures for combined plots
plt.figure(figsize=(10, 6))
plt.title("Effect of Air Temperature on Moisture Content")
plt.xlabel("Drying time in s")
plt.ylabel("Moisture content (dry basis)")

for T_air in air_temperatures:
    air_conditions = {'T': T_air, 'RH': 0.2}  # Relative humidity of 20%
    problem = run_drying_simulation(f'Air Temp: {T_air} °C', air_conditions, material_params_base)

    plot_moisture_content(problem, f'Air Temp: {T_air} °C')

plt.legend()
plt.show()

# Similarly for surface temperature
plt.figure(figsize=(10, 6))
plt.title("Effect of Air Temperature on Surface Temperature")
plt.xlabel("Drying time in s")
plt.ylabel("Surface temperature (°C)")

for T_air in air_temperatures:
    air_conditions = {'T': T_air, 'RH': 0.2}
    problem = run_drying_simulation(f'Air Temp: {T_air} °C', air_conditions, material_params_base)

    plot_surface_temperature(problem, f'Air Temp: {T_air} °C')

plt.legend()
plt.show()

# Drying Rate Plot
plt.figure(figsize=(10, 6))
plt.title("Effect of Air Temperature on Drying Rate")
plt.xlabel("Drying time in s")
plt.ylabel("Drying rate (1/s)")

for T_air in air_temperatures:
    air_conditions = {'T': T_air, 'RH': 0.2}
    problem = run_drying_simulation(f'Air Temp: {T_air} °C', air_conditions, material_params_base)

    plot_drying_rate(problem, f'Air Temp: {T_air} °C')

plt.legend()
plt.show()

# Heat Flux Plot
plt.figure(figsize=(10, 6))
plt.title("Effect of Air Temperature on Heat Flux")
plt.xlabel("Drying time in s")
plt.ylabel("Heat Flux (W/m^2)")

for T_air in air_temperatures:
    air_conditions = {'T': T_air, 'RH': 0.2}
    problem = run_drying_simulation(f'Air Temp: {T_air} °C', air_conditions, material_params_base)

    plot_heat_flux(problem, air_conditions, material_params_base, f'Air Temp: {T_air} °C')

plt.legend()
plt.show()

# Experiment 2: Varying Heat Transfer Coefficient
heat_transfer_coeffs = [5, 50, 150]  # Heat transfer coefficients
material_params_base['X0'] = 0.5  # Reset initial moisture if changed

# Moisture Content Plot
plt.figure(figsize=(10, 6))
plt.title("Effect of Heat Transfer Coefficient on Moisture Content")
plt.xlabel("Drying time in s")
plt.ylabel("Moisture content (dry basis)")

for h in heat_transfer_coeffs:
    material_params = material_params_base.copy()
    material_params['heat_transfer'] = h
    air_conditions = {'T': 60, 'RH': 0.2}
    problem = run_drying_simulation(f'Heat Transfer: {h} W/m²K', air_conditions, material_params)

    plot_moisture_content(problem, f'Heat Transfer: {h} W/m²K')

plt.legend()
plt.show()

# Surface Temperature Plot
plt.figure(figsize=(10, 6))
plt.title("Effect of Heat Transfer Coefficient on Surface Temperature")
plt.xlabel("Drying time in s")
plt.ylabel("Surface temperature (°C)")

for h in heat_transfer_coeffs:
    material_params = material_params_base.copy()
    material_params['heat_transfer'] = h
    air_conditions = {'T': 60, 'RH': 0.2}
    problem = run_drying_simulation(f'Heat Transfer: {h} W/m²K', air_conditions, material_params)

    plot_surface_temperature(problem, f'Heat Transfer: {h} W/m²K')

plt.legend()
plt.show()

# Drying Rate Plot
plt.figure(figsize=(10, 6))
plt.title("Effect of Heat Transfer Coefficient on Drying Rate")
plt.xlabel("Drying time in s")
plt.ylabel("Drying rate (1/s)")

for h in heat_transfer_coeffs:
    material_params = material_params_base.copy()
    material_params['heat_transfer'] = h
    air_conditions = {'T': 60, 'RH': 0.2}
    problem = run_drying_simulation(f'Heat Transfer: {h} W/m²K', air_conditions, material_params)

    plot_drying_rate(problem, f'Heat Transfer: {h} W/m²K')

plt.legend()
plt.show()

# Heat Flux Plot
plt.figure(figsize=(10, 6))
plt.title("Effect of Heat Transfer Coefficient on Heat Flux")
plt.xlabel("Drying time in s")
plt.ylabel("Heat Flux (W/m²)")

for h in heat_transfer_coeffs:
    material_params = material_params_base.copy()
    material_params['heat_transfer'] = h
    air_conditions = {'T': 60, 'RH': 0.2}
    problem = run_drying_simulation(f'Heat Transfer: {h} W/m²K', air_conditions, material_params)

    plot_heat_flux(problem, air_conditions, material_params, f'Heat Transfer: {h} W/m²K')

plt.legend()
plt.show()

# Experiment 3: Varying Thermal Conductivity
thermal_conductivities = [0.01, 0.05, 0.2]  # Thermal conductivity values

# Moisture Content Plot
plt.figure(figsize=(10, 6))
plt.title("Effect of Thermal Conductivity on Moisture Content")
plt.xlabel("Drying time in s")
plt.ylabel("Moisture content (dry basis)")

for Lambda_val in thermal_conductivities:
    problem = run_drying_simulation(f'Thermal Conductivity: {Lambda_val} W/mK',
                                    air_conditions, material_params_base, Lambda_val=Lambda_val)

    plot_moisture_content(problem, f'Lambda: {Lambda_val} W/mK')

plt.legend()
plt.show()

# Surface Temperature Plot
plt.figure(figsize=(10, 6))
plt.title("Effect of Thermal Conductivity on Surface Temperature")
plt.xlabel("Drying time in s")
plt.ylabel("Surface temperature (°C)")

for Lambda_val in thermal_conductivities:
    problem = run_drying_simulation(f'Thermal Conductivity: {Lambda_val} W/mK',
                                    air_conditions, material_params_base, Lambda_val=Lambda_val)

    plot_surface_temperature(problem, f'Lambda: {Lambda_val} W/mK')

plt.legend()
plt.show()

# Drying Rate Plot
plt.figure(figsize=(10, 6))
plt.title("Effect of Thermal Conductivity on Drying Rate")
plt.xlabel("Drying time in s")
plt.ylabel("Drying rate (1/s)")

for Lambda_val in thermal_conductivities:
    problem = run_drying_simulation(f'Thermal Conductivity: {Lambda_val} W/mK',
                                    air_conditions, material_params_base, Lambda_val=Lambda_val)

    plot_drying_rate(problem, f'Lambda: {Lambda_val} W/mK')

plt.legend()
plt.show()

# Heat Flux Plot
plt.figure(figsize=(10, 6))
plt.title("Effect of Thermal Conductivity on Heat Flux")
plt.xlabel("Drying time in s")
plt.ylabel("Heat Flux (W/m²)")

for Lambda_val in thermal_conductivities:
    problem = run_drying_simulation(f'Thermal Conductivity: {Lambda_val} W/mK',
                                    air_conditions, material_params_base, Lambda_val=Lambda_val)

    plot_heat_flux(problem, air_conditions, material_params_base, f'Lambda: {Lambda_val} W/mK')

plt.legend()
plt.show()
