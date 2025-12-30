

def bilinear(v0, v1, v2, v3, alpha, beta):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  #   v0---α---v1
  #   |    :    |
  #   |    β    |
  #   |    :    |
  #   v2---α---v3
  aprime = (alpha + 1) / 2
  bprime = (beta + 1) / 2
  top_point = aprime * v0 + (1 - aprime) * v1
  bottom_point = aprime * v2 + (1 - aprime) * v3
  return (bprime * top_point + (1 - bprime) * bottom_point)


def bilinear_jacobian(v0, v1, v2, v3, alpha, beta):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
    the 1st param name `first`
  second :
    the 2nd param
  third : {'value', 'other'}, optional
    the 3rd param, by default 'value'

  Returns
  -------
  string
    a value in a string

  Raises
  ------
  KeyError
    when a key error
  """
  aprime = (alpha + 1) / 2
  bprime = (beta + 1) / 2
  dphys_dalpha = 1 / 2.0 * (bprime * (v0 - v1) + (1 - bprime) * (v2 - v3))
  dphys_dbeta = 1 / 2.0 * (aprime * v0 + (1 - aprime) * v1 - (aprime * v2 + (1 - aprime) * v3))
  return dphys_dalpha, dphys_dbeta
