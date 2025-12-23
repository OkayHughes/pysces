from .config import np, npt


def init_deriv(gll_points):
  # uses the lagrange interpolating polynomials
  leg_eval = np.zeros(shape=(npt, npt))
  leg_der = np.zeros(shape=(npt, npt))

  for deg in range(npt):
    c = np.zeros(npt)
    c[deg] = 1.0
    leg_eval[:, deg] = np.polynomial.legendre.legval(gll_points, c)
    der = np.polynomial.legendre.legder(c, 1)
    leg_der[:, deg] = np.polynomial.legendre.legval(gll_points, der)

  coeffs = np.linalg.inv(leg_eval)
  return np.dot(leg_der, coeffs)


gll_points = np.array([1.0, np.sqrt(1 / 5), -np.sqrt(1 / 5), -1.0])

deriv = {"gll_points": np.array([1.0, np.sqrt(1 / 5), -np.sqrt(1 / 5), -1.0]),
         "gll_weights": np.array([1 / 6, 5 / 6, 5 / 6, 1 / 6]),
         "deriv": init_deriv(gll_points)}
