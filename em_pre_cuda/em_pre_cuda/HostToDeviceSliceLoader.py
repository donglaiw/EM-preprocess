import torch
from torch.multiprocessing import Queue
from threading import Thread
from torch import cuda


class HostToDeviceSliceLoader(object):
    def __init__(self, streams_per_device, disk_host_loader, num_threads = 2):
        self.queue = Queue()
        self.streams = [[cuda.Stream(i)] * streams_per_device for i in range(disk_host_loader.num_queues())]
        disk_host_loader.set_post_queue_copy(lambda slc_idx, q_idx, slc: self.queue.put((slc_idx, q_idx, slc)))
        self.num_threads = num_threads

    def set_post_queue_copy(self, func):
        self.slc_grid_put = func

    def start(self):
        def thread_target():
            i = 0
            while True:
                if not self.queue.empty():
                    slc_idx, q_idx, slc_cpu = self.queue.get()
                    str_idx = i % len(self.streams[q_idx])
                    cur_stream = self.streams[q_idx][str_idx]
                    with cuda.device(q_idx):
                        with cuda.stream(cur_stream):
                            slc_cuda = slc_cpu.cuda()
                    self.slc_grid_put(slc_idx, q_idx, cur_stream, slc_cuda)
                i = i + 1

        threads = [Thread(target=thread_target) for _ in range(self.num_threads)]
        map(lambda x: x.setDaemon(True), threads)
        map(lambda x: x.start(), threads)