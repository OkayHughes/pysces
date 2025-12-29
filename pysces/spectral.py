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


gll_points = {3: {"points": np.array([-1.0, 0.0, 1.0]),
                  "weights": np.array([1.0/3.0, 4.0/3.0, 1.0/3.0])},
              4: {"points": np.array([1.0, np.sqrt(1 / 5), -np.sqrt(1 / 5), -1.0]),
                  "weights": np.array([1 / 6, 5 / 6, 5 / 6, 1 / 6])},
              5: {"points": np.array([1.0, np.sqrt(3 / 7), 0.0, -np.sqrt(3 / 7), -1.0]),
                  "weights": np.array([1.0 / 10.0, 49.0/90.0, 32.0 / 45.0, 49.0/90.0, 1.0/10.0])},
              6: {"points": np.array([1.0,
                                      np.sqrt(1.0 / 3.0 + 2.0 * np.sqrt(7) / 21.0),
                                      np.sqrt(1.0 / 3.0 - 2.0 * np.sqrt(7) / 21.0),
                                      -np.sqrt(1.0 / 3.0 - 2.0 * np.sqrt(7) / 21.0),
                                      -np.sqrt(1.0 / 3.0 + 2.0 * np.sqrt(7) / 21.0),
                                      -1.0]),
                  "weights": np.array([1.0/15.0,
                                       (14.0 - np.sqrt(7))/30.0,
                                       (14.0 + np.sqrt(7))/30.0,
                                       (14.0 + np.sqrt(7))/30.0,
                                       (14.0 - np.sqrt(7))/30.0,
                                       1.0/15.0])}}

deriv = {"gll_points": gll_points[npt]["points"],
         "gll_weights": gll_points[npt]["weights"],
         "deriv": init_deriv(gll_points[npt]["points"])}
