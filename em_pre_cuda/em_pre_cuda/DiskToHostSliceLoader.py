from torch.multiprocessing import Queue
from threading import Thread


class DiskToHostSliceLoader(object):

    def __init__(self, load_slice_from_disk, slice_range, num_threads=8, num_devices=1):
        self.disk_load = load_slice_from_disk
        self.num_threads = num_threads
        self.queues = [Queue() for _ in range(num_devices)]
        num_slices = len(slice_range)
        slc_per_que = num_slices / num_devices
        for i, queue in enumerate(self.queues):
            queue_range = slice_range[slc_per_que * i: slc_per_que * (i + 1)]
            if i == len(self.queues) - 1:
                right = slice_range[slc_per_que * (i + 1): slice_range[-1]]
                queue_range = queue_range + right
            map(queue.put, queue_range)

    def num_queues(self):
        return len(self.queues)
    def set_post_queue_copy(self, func):
        self.h_to_d_put = func

    def start(self):

        def thread_target():
            i = 0
            while ([queue.empty() for queue in self.queues]).count(True) != len(self.queues):
                q_idx = i % len(self.queues)
                if not self.queues[q_idx].empty():
                    slc_index = self.queues[q_idx].get()
                    slc = self.disk_load(slc_index)
                    self.h_to_d_put(slc_index, q_idx, slc)
                i = i + 1
            return
        threads = [Thread(target=thread_target) for _ in range(self.num_threads)]
        map(lambda x: x.setDaemon(True), threads)
        map(lambda x: x.start(), threads)
