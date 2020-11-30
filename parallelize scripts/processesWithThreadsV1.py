import numpy as np
from mpi4py import MPI
from queue import Queue
from threading import Thread
from time import sleep

THREAD_COUNT = 4
QUEUE_SIZE = 10

class DownloadWorker(Thread):

#   queue should have input to process
    def __init__(self, queue, returnQueue, rank, threadNum):
        Thread.__init__(self)
        self.queue = queue
        self.rQueue = returnQueue
        self.rank = rank
        self.threadNum = threadNum

    def run(self):
        value = 0
        while True:
            try:
            # Get the work from the queue and expand the tuple
            # Sleep was added to simulate intensive work
                # sleep(0.000001) 
                value = self.queue.get()
                # print(self.rank, self.threadNum, value)
            finally:
                self.rQueue.put(value)
                self.queue.task_done()

comm = MPI.COMM_WORLD

comm.Barrier()
t_start = MPI.Wtime()

# this array lives on each processor
data = np.zeros(QUEUE_SIZE)
l = []

# start = MPI.Wtime()
totals = None
queue = Queue()
rQueue = Queue()
threads = []
# Create 4 worker threads
for x in range(4):
    worker = DownloadWorker(queue, rQueue, comm.rank, x)
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    worker.daemon = True
    worker.start()
    # threads.append(worker)

# Put the tasks into the queue as a tuple
for i in range(1, 11):
    queue.put(i)

# Causes the main thread to wait for the queue to finish processing all the tasks
queue.join()

value = sum(rQueue.queue)
# print(data)
# print(MPI.Wtime()-start)

if comm.rank > 0:
    comm.send(value, dest=0, tag=11)
else:
    for i in range(1, comm.size):
        value += comm.recv(source=i, tag=11)

    print('[%i]'%comm.rank, value)

comm.Barrier()
t_diff = MPI.Wtime() - t_start
if comm.rank==0: print (t_diff)