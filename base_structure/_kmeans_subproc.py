"""Subprocess entrypoint to run KMeans safely in a separate process.
This script is invoked with: python -m base_structure._kmeans_subproc <in.npy> <out.npy> <n_clusters> <mb_batch> <use_mini>
It writes the cluster centers to <out.npy> on success and exits non-zero on failure.
"""
import sys
import argparse
import traceback
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inp')
    p.add_argument('out')
    p.add_argument('n_clusters', type=int)
    p.add_argument('mb_batch', type=int)
    p.add_argument('use_mini', type=int)
    args = p.parse_args()
    try:
        X = np.load(args.inp)
        if args.use_mini:
            mb = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=0, batch_size=args.mb_batch, max_iter=10)
            mb.fit(X)
            centers = mb.cluster_centers_
        else:
            km = KMeans(n_clusters=args.n_clusters, n_init=1, random_state=0, max_iter=300)
            km.fit(X)
            centers = km.cluster_centers_
        np.save(args.out, centers)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
