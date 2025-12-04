from spherical_spectral_element.config import jnp, np, jax_wrapper
from spherical_spectral_element.theta_l.vert_remap_reference import for_loop_remap
from spherical_spectral_element.theta_l.vertical_remap import zerroukat_remap


def get_testbed(seed=True, random=False):
  nF = 2
  
  npt = 4
  nlev = 10
  reference_levs = jnp.linspace(0, 1, nlev+1)
  if seed:
    np.random.seed(0)
  else:
    np.random.seed(1)
  ints = jnp.concatenate(([0.0],
                          jnp.sort(jax_wrapper(np.random.uniform(size=nlev-1))),
                          [1.0]))
  deltas = (ints[1:] - ints[:-1])[jnp.newaxis, jnp.newaxis, jnp.newaxis, :] * jnp.ones((nF, npt, npt, 1))
  deltas_ref = (reference_levs[1:] - reference_levs[:-1])[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]* jnp.ones((nF, npt, npt, 1))
  if random:
    Qs = jax_wrapper(np.random.uniform(size=(nF, npt, npt, nlev, 2)))
  else:
    Qs = jnp.ones((nF, npt, npt, nlev, 2))
  
  return deltas, deltas_ref, Qs, Qs * deltas[:, :, :, :, jnp.newaxis]


def test_for_remap():
    for _ in range(100):
      for random in [True, False]:
        deltas, deltas_ref, Qs, Qdps = get_testbed(seed=False, random=random)
        Qdp_out = for_loop_remap(Qdps, deltas, deltas_ref)
        Qdp_out_filt = for_loop_remap(Qdps, deltas, deltas_ref, filter=True)
        if not random:
          assert(jnp.allclose(Qdp_out, deltas_ref[:, :, :, :, jnp.newaxis]))
        assert(jnp.allclose(jnp.sum(Qdps, axis=-2), jnp.sum(Qdp_out, axis=-2)))
        assert(jnp.allclose(jnp.sum(Qdps, axis=-2), jnp.sum(Qdp_out_filt, axis=-2)))
        assert(not jnp.any(Qdp_out_filt < 0))
        #if jnp.any(Qdp_out < 0):
        #  print("Warning: negative remapped value")


def test_remap():
  for _ in range(100):
    deltas, deltas_ref, Qs, Qdps = get_testbed(seed=True, random=True)
    #print("for loop output")
    #Qdps = Q_mix * deltas[:, :, :, :, jnp.newaxis]
    Qdp_out_for = for_loop_remap(Qdps, deltas, deltas_ref, filter=False)

    dims = {"num_level": deltas.shape[-1]}
    #print("numpy loop output")
    Qdp_out = zerroukat_remap(Qdps, deltas, deltas_ref, dims)
    assert (jnp.max(jnp.abs(Qdp_out_for - Qdp_out)) < 1e-10)
    assert(jnp.allclose(jnp.sum(Qdps, axis=-2), jnp.sum(Qdp_out, axis=-2)))

    
    #print(f"Diff between quantity: {jnp.max(jnp.abs(Qdp_out_for - Qdp_out))}")
    #print(Qdp_out_for)
    #print(Qdp_out)
    #print(jnp.sum(Qdp_out))
