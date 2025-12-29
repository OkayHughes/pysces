from pysces.config import np
from pysces.mathops import bilinear, bilinear_jacobian
from pysces.cubed_sphere import gen_cube_topo
from pysces.spectral import init_spectral
from .context import test_npts

def test_bilinear():
  NFACES = 1000
  np.random.seed(0)
  face_position_2d = np.random.uniform(size=(NFACES, 4, 2))
  jac_test = np.zeros(shape=(NFACES, 2, 2))
  diff_minus = np.zeros(shape=(NFACES, 2))
  diff_plus = np.zeros(shape=(NFACES, 2))
  ncheck = 5
  nfrac = np.linspace(-1, 1, ncheck)
  for i in range(ncheck):
    for j in range(ncheck):
      alpha = nfrac[i]
      beta = nfrac[j]
      eps = 1e-6
      diff_plus = bilinear(face_position_2d[:, 0, :],
                           face_position_2d[:, 1, :],
                           face_position_2d[:, 2, :],
                           face_position_2d[:, 3, :], alpha + eps, beta)
      diff_minus = bilinear(face_position_2d[:, 0, :],
                            face_position_2d[:, 1, :],
                            face_position_2d[:, 2, :],
                            face_position_2d[:, 3, :], alpha - eps, beta)
      dres_dalpha = (diff_plus - diff_minus) / (2 * eps)
      diff_plus = bilinear(face_position_2d[:, 0, :],
                           face_position_2d[:, 1, :],
                           face_position_2d[:, 2, :],
                           face_position_2d[:, 3, :], alpha, beta + eps)
      diff_minus = bilinear(face_position_2d[:, 0, :],
                            face_position_2d[:, 1, :],
                            face_position_2d[:, 2, :],
                            face_position_2d[:, 3, :], alpha, beta - eps)
      dphys_dalpha, dphys_dbeta = bilinear_jacobian(face_position_2d[:, 0, :],
                                                    face_position_2d[:, 1, :],
                                                    face_position_2d[:, 2, :],
                                                    face_position_2d[:, 3, :], alpha, beta)
      jac_test[:, :, 0] = dphys_dalpha
      jac_test[:, :, 1] = dphys_dbeta
      dres_dbeta = (diff_plus - diff_minus) / (2 * eps)
      assert (np.max(np.abs(dres_dalpha - jac_test[:, :, 0])) < 1e-7)
      assert (np.max(np.abs(dres_dbeta - jac_test[:, :, 1])) < 1e-7)


def test_bilinear_cs():
  for npt in test_npts:
    spectrals = init_spectral(npt)
    for nx in [7, 8]:
      face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
      NFACES = face_position.shape[0]
      jac_test = np.zeros(shape=(NFACES, 2, 2))
      diff_minus = np.zeros(shape=(NFACES, 2))
      diff_plus = np.zeros(shape=(NFACES, 2))
      for i in range(npt):
        for j in range(npt):
          alpha = spectrals["gll_points"][i]
          beta = spectrals["gll_points"][j]
          eps = 1e-4
          diff_plus = bilinear(face_position_2d[:, 0, :],
                              face_position_2d[:, 1, :],
                              face_position_2d[:, 2, :],
                              face_position_2d[:, 3, :], alpha + eps, beta)
          diff_minus = bilinear(face_position_2d[:, 0, :],
                                face_position_2d[:, 1, :],
                                face_position_2d[:, 2, :],
                                face_position_2d[:, 3, :], alpha - eps, beta)
          dres_dalpha = (diff_plus - diff_minus) / (2 * eps)
          diff_plus = bilinear(face_position_2d[:, 0, :],
                              face_position_2d[:, 1, :],
                              face_position_2d[:, 2, :],
                              face_position_2d[:, 3, :], alpha, beta + eps)
          diff_minus = bilinear(face_position_2d[:, 0, :],
                                face_position_2d[:, 1, :],
                                face_position_2d[:, 2, :],
                                face_position_2d[:, 3, :], alpha, beta - eps)
          dphys_dalpha, dphys_dbeta = bilinear_jacobian(face_position_2d[:, 0, :],
                                                        face_position_2d[:, 1, :],
                                                        face_position_2d[:, 2, :],
                                                        face_position_2d[:, 3, :], alpha, beta)
          jac_test[:, :, 0] = dphys_dalpha
          jac_test[:, :, 1] = dphys_dbeta
          dres_dbeta = (diff_plus - diff_minus) / (2 * eps)
          assert (np.max(np.abs(dres_dalpha - jac_test[:, :, 0])) < 1e-7)
          assert (np.max(np.abs(dres_dbeta - jac_test[:, :, 1])) < 1e-7)
