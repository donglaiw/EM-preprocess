from threading import Thread
import torch
import torch.cuda as cuda
from torch.multiprocessing import Lock, Queue

class SliceGrid(object):
    def __init__(self, num_devices, slice_range, median_rad, h_to_d_loader, num_threads = 2):
        self.num_threads = num_threads
        self.queue = Queue()
        self.slice_range = slice_range
        self.num_dev = num_devices
        self.slc_per_dev = len(slice_range) / num_devices
        self.med_rad = median_rad
        self.grid = [[None] * (2 * median_rad + 1) for _ in slice_range]
        self.lock_array = [Lock() for _ in slice_range]
        h_to_d_loader.set_post_queue_copy(lambda slc_idx, dev_idx, stream, slc_cuda:
                                          self.queue.put((slc_idx, dev_idx, stream, slc_cuda)))

    def set_post_queue_copy(self, func):
        self.d_to_h_put = func

    def set_pre_process_method(self, method):
        self.pre_process = method

    def start(self):
        def thread_target():
            while self.grid.count(0) != len(self.grid):
                if not self.queue.empty():
                    slc_idx, dev_idx, stream, slc_cuda = self.queue.get()
                    phoney_idx = self.slice_range.index(slc_idx)
                    j = 0
                    for i in range(0, 2 * self.med_rad + 1):
                        j = phoney_idx - self.med_rad + i
                        if j / self.slc_per_dev == dev_idx and j >= 0 and j < len(self.slice_range):
                            self.grid[j][i] = (stream, slc_cuda)
                            self.lock_array[j].acquire()
                            if self.grid[j] != 0 and self.grid[j].count(None) == 0:
                                calc_stream = cuda.Stream(dev_idx)
                                for element in self.grid[j]:
                                    element[0].synchronize()
                                with cuda.stream(calc_stream):
                                    input = torch.stack(element[1] for element in self.grid[j])
                                    self.d_to_h_put(slc_idx, calc_stream, self.pre_process(input))
                                self.grid[j] = 0
                            self.lock_array[j].release()
        threads = [Thread(target=thread_target) for _ in range(self.num_threads)]
        map(lambda x: x.setDaemon(True), threads)
        map(lambda x: x.start(), threads)
