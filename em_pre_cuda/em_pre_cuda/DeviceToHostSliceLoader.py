import torch
from torch.multiprocessing import Queue
from threading import Thread
from torch import cuda


class DeviceToHostSliceLoader(object):
    def __init__(self, slice_grid, num_threads = 2):
        self.queue = Queue()
        slice_grid.set_post_queue_copy(lambda slc_idx, calc_stream, slc: self.queue.put((slc_idx, calc_stream, slc)))
        self.num_threads = num_threads

    def set_post_queue_copy(self, func):
        self.slc_grid_put = func

    def start(self):
        def thread_target():
            while True:
                if not self.queue.empty():
                    slc_idx, stream, slc_gpu = self.queue.get()
                    stream.synchronize()
                    slc_cpu = slc_gpu.cpu()
                    self.slc_grid_put(slc_idx, slc_cpu)

        threads = [Thread(target=thread_target) for _ in range(self.num_threads)]
        map(lambda x: x.setDaemon(True), threads)
        map(lambda x: x.start(), threads)