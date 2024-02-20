# Set OpenBLAS to use 4 threads
export OPENBLAS_NUM_THREADS=300

# Check the current setting
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"

python ctgan_model.py