"""Real MJX random-push checks: sampling, 80 force projections, force on/off.
No PPO training. Uses CPU; the initial MJX compilation may take minutes.
"""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from finger_env import FingerHeightEnv, PUSH_POINTS, PUSH_DIRECTIONS
from finger_env_mjx import FingerHeightMjxEnv
config=Path(__file__).with_name('config_push.json')
e=FingerHeightMjxEnv(config_path=config)
n=FingerHeightEnv(config_path=config)
reset=jax.jit(e.reset)
s=reset(jax.random.PRNGKey(1))
print('real MJX reset PASS',flush=True)
batch=jax.jit(jax.vmap(e.reset))(jax.random.split(jax.random.PRNGKey(6),128))
forces=np.linalg.norm(batch.info['push_force'],axis=1)
assert forces.min()>=1-1e-6 and forces.max()<=4+1e-6
assert len(np.unique(batch.info['push_point_index']))==10
assert len(np.unique(batch.info['push_direction_index']))==8
assert np.any(batch.info['push_sampled']) and not np.all(batch.info['push_sampled'])
print('128 JIT/vmap resets: all 10 points and 8 directions sampled; 1-4 N PASS',flush=True)
# Align explicit perturbation across NumPy/JAX; their RNG algorithms differ.
n.reset(seed=1)
n.push_sampled=True
n.push_point_index=4
n.push_direction_index=6
n.push_force=np.array([0.,0.,-1.])
n.push_start_tick=0
n.push_duration_ticks=20
s=s.replace(info={**s.info,'push_sampled':jp.array(True),'push_point_index':jp.int32(4),
 'push_direction_index':jp.int32(6),'push_force':jp.array([0.,0.,-1.]),'push_start_tick':jp.int32(0),'push_duration_ticks':jp.int32(20)})
apply=jax.jit(e._apply_push)
for i in range(10):
 for j in range(8):
  body,local=list(PUSH_POINTS.values())[i]
  bid=mujoco.mj_name2id(n.model,mujoco.mjtObj.mjOBJ_BODY,body)
  point=n.data.xpos[bid]+n.data.xmat[bid].reshape(3,3)@local
  q=np.zeros(3)
  mujoco.mj_applyFT(n.model,n.data,PUSH_DIRECTIONS[j],np.zeros(3),point,bid,q)
  d,_=apply(s.data,{**s.info,'push_point_index':jp.int32(i),'push_force':jp.array(PUSH_DIRECTIONS[j],jp.float32)})
  np.testing.assert_allclose(d.qfrc_applied,q,rtol=1e-5,atol=1e-8)
print('80 native/MJX force projection comparisons PASS',flush=True)
step=jax.jit(e.step)
print('Compiling real MJX step with random push...',flush=True)
for i in range(3):
 s=step(s,jp.array([0.]))
 s.obs.block_until_ready()
 obs,_,terminated,_,info=n.step([0.])
 assert not terminated
 np.testing.assert_allclose(s.data.qpos,n.data.qpos,atol=1e-3,rtol=1e-3)
 np.testing.assert_allclose(s.metrics['height'],info['height'],atol=1e-4)
 assert int(s.info['push_applied_ticks'])==20
 if i>=1:
  np.testing.assert_array_equal(s.data.qfrc_applied,np.zeros(3))
print('real MJX force-on/force-off steps; native parity; exactly 20 ticks PASS',flush=True)
