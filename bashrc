...
# The line below has been commented
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export GTK_PATH=/usr/lib/x86_64-linux-gnu/gtk-2.0

# These two lines have been added
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/envs/transcriber/lib:$LD_LIBRARY_PATH
