import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv("Salary_positions.csv")

X = df[['Level']]
y = df['Salary']

lin_reg = LinearRegression()
lin_reg.fit(X, y)
r2_lin = r2_score(y, lin_reg.predict(X))

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
r2_poly = r2_score(y, poly_reg.predict(X_poly))

print(f"Simple Linear Regression R2: {r2_lin:.4f}")
print(f"Polynomial Linear Regression R2: {r2_poly:.4f}")

if r2_poly > r2_lin:
    print("Polynomial Regression is more accurately fitting to the given data.")
else:
    print("Simple Linear Regression is more accurately fitting to the given data.")

test_levels = pd.DataFrame({'Level': [11, 12]})
pred_lin = lin_reg.predict(test_levels)
pred_poly = poly_reg.predict(poly.transform(test_levels))

print(f"\nSimple Linear Regression - Level 11: {pred_lin[0]:.2f}, Level 12: {pred_lin[1]:.2f}")
print(f"Polynomial Linear Regression - Level 11: {pred_poly[0]:.2f}, Level 12: {pred_poly[1]:.2f}")
