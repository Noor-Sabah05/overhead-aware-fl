# Fashion-MNIST dataset

The experiment uses Fashion-MNIST: 60,000 training images and 10,000 test images, with ten classes of 28 x 28 grayscale clothing images.

From the repository root, prepare the data with:

```bash
python src/experiment.py prepare --config configs/short.json
```

The loader downloads the four official gzip archives from the Fashion-MNIST repository, checks their expected MD5 values, and saves the archives, prepared NPZ cache, and JSON metadata in this directory. Use `--cache /path/to/cache` to select another location.

The selected configuration controls the training subset and fixed subset seed; the official test set is retained. Normalization uses mean 0.2860 and standard deviation 0.3530. Downloaded data and caches are ignored by Git. Existing experiment provenance is recorded in each result directory's `manifest.json`.