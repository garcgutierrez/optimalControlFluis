"""Optimal transient growth map G_max(lambda_x, lambda_z) at Re_tau = 2000."""
import common
import numpy as np, time, sys
from multiprocessing import Pool
from lq import Plant

def work(args):
    lx, lz = args
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=common.NCHEB, actuator='wall', profile=common.profile())
    tmax, Gmax, z0, tg, g = pl.transient_growth(tgrid=np.geomspace(1e-4, 20.0, 90))
    return lx, lz, tmax, Gmax

if __name__ == '__main__':
    nx, nz = int(sys.argv[1]), int(sys.argv[2])
    LX = np.geomspace(0.1, 100.0, nx)
    LZ = np.geomspace(0.02, 10.0, nz)
    common.profile()
    jobs = [(lx, lz) for lx in LX for lz in LZ]
    t0 = time.time()
    with Pool(18) as p:
        res = p.map(work, jobs, chunksize=1)
    res = np.array(res)
    G = res[:, 3].reshape(nx, nz); TM = res[:, 2].reshape(nx, nz)
    np.savez(f'{common.DATA}/tg_map_{nx}x{nz}.npz', LX=LX, LZ=LZ, G=G, tmax=TM)
    print('done', time.time()-t0)
    i, j = np.unravel_index(np.argmax(G), G.shape)
    print('max G', G.max(), 'at lx', LX[i], 'lz', LZ[j])
    np.set_printoptions(linewidth=250, precision=2)
    print('rows: lx; cols: lz', LZ)
    for a, row in zip(LX, G): print(f'{a:8.3f}', row)
