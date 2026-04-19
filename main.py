import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------
# LOAD DATA
# -------------------------------
data = pd.read_csv("data.csv", skiprows=2)

data.columns = ['Date', 'Rainfall', 'Inflow', 'Outflow', 'Streamflow', 'Q1', 'Q2']

# FIX: handle '-' values
data = data.replace('-', pd.NA)

for col in ['Rainfall', 'Inflow', 'Streamflow', 'Q1', 'Q2']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna()

# -------------------------------
# MODEL
# -------------------------------
X = data[['Rainfall', 'Inflow', 'Q1', 'Q2']]
y = data['Streamflow']

model = LinearRegression()
model.fit(X, y)

coeffs = model.coef_
intercept = model.intercept_

# Predictions
y_pred = model.predict(X)

# Metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# Dependency %
total = sum(abs(coeffs))
deps = [(abs(c)/total)*100 for c in coeffs]

# Error %
error_percent = (mae / y.mean()) * 100

# -------------------------------
# MENU
# -------------------------------
print("\n==============================")
print("STREAMFLOW PREDICTION SYSTEM")
print("==============================")
print("1. Show Model + Graph")
print("2. Manual Prediction (Teacher Input)")
print("3. Automatic Future Prediction")

choice = int(input("\nEnter your choice: "))

# -------------------------------
# OPTION 1: MODEL + GRAPH
# -------------------------------
if choice == 1:
    print("\nREGRESSION MODEL")
    print(f"Q(t) = {coeffs[0]:.3f}*Rain + {coeffs[1]:.3f}*Inflow + "
          f"{coeffs[2]:.3f}*Q(t-1) + {coeffs[3]:.3f}*Q(t-2) + {intercept:.3f}")

    print("\nMODEL PERFORMANCE")
    print(f"R2 Score: {r2:.3f}")
    print(f"MAE: {mae:.3f} MCM")

    print("\nDEPENDENCY (%)")
    print(f"Rainfall: {deps[0]:.2f}%")
    print(f"Inflow: {deps[1]:.2f}%")
    print(f"Q(t-1): {deps[2]:.2f}%")
    print(f"Q(t-2): {deps[3]:.2f}%")

    print(f"\nAverage Error: {error_percent:.2f}%")

    # Graph
    subset = 300
    plt.figure()
    plt.plot(y.values[:subset], label="Actual")
    plt.plot(y_pred[:subset], label="Predicted")
    plt.title("Actual vs Predicted Streamflow")
    plt.xlabel("Time")
    plt.ylabel("Streamflow (MCM)")
    plt.legend()
    plt.show()

# -------------------------------
# OPTION 2: MANUAL TESTING
# -------------------------------
elif choice == 2:
    print("\nMANUAL PREDICTION")

    rain = float(input("Enter Rainfall (mm): "))
    inflow = float(input("Enter Inflow (MCM): "))
    q1 = float(input("Enter Q(t-1) (MCM): "))
    q2 = float(input("Enter Q(t-2) (MCM): "))

    # FIX: Use DataFrame (removes warning)
    input_data = pd.DataFrame([[rain, inflow, q1, q2]],
                              columns=['Rainfall', 'Inflow', 'Q1', 'Q2'])

    pred = model.predict(input_data)[0]

    print(f"\nPredicted Streamflow = {pred:.2f} MCM")

# -------------------------------
# OPTION 3: FUTURE PREDICTION
# -------------------------------
elif choice == 3:
    print("\nFUTURE PREDICTION (NEXT 5 DAYS)")

    rain = data['Rainfall'].iloc[-1]
    inflow = data['Inflow'].iloc[-1]
    q1 = data['Streamflow'].iloc[-1]
    q2 = data['Streamflow'].iloc[-2]

    future = []

    for i in range(5):
        input_data = pd.DataFrame([[rain, inflow, q1, q2]],
                                  columns=['Rainfall', 'Inflow', 'Q1', 'Q2'])

        next_val = model.predict(input_data)[0]
        future.append(next_val)

        q2 = q1
        q1 = next_val

    for i, val in enumerate(future, 1):
        print(f"Day {i}: {val:.2f} MCM")

    # Graph
    plt.figure()
    plt.plot(data['Streamflow'].values, label="Actual")

    future_index = range(len(data), len(data)+5)
    plt.plot(future_index, future, linestyle='--', label="Future")

    plt.title("Future Prediction")
    plt.xlabel("Time")
    plt.ylabel("Streamflow (MCM)")
    plt.legend()
    plt.show()

else:
    print("Invalid choice")