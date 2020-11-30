1. Install OpenMPI/MPI on the cluster/machine
2. pip install mpi4py

For executing the scripts use command: mpiexec -n 4 python script.py
-n defines number of processes to spawn
Number of threads to spawn has been hard coded in the script. (Currently, it spawns 4 threads)

processesV1.py - Parallel execution only using processes
processesWithThreadsV1.py - Parallel execution using processes which further spawn threads

NOTE: mpi4py only works on python >=3.4