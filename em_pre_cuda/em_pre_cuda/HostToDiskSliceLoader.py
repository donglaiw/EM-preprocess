from torch.multiprocessing import Queue
from threading import Thread


class HostToDiskSliceLoader(object):

    def __init__(self, save_slice_to_disk, slice_range, num_threads=8):
        self.disk_save = save_slice_to_disk
        self.num_threads = num_threads
        self.queue = Queue()
        self.slice_range = slice_range

    def start(self):
        def thread_target():
            while len(self.slice_range) != 0:
                if not self.queue.empty():
                    slc_index = self.queue.get()
                    self.disk_save(slc_index)
                    self.slice_range.remove(slc_index)
        threads = [Thread(target=thread_target) for _ in range(self.num_threads)]
        map(lambda x: x.setDaemon(True), threads)
        map(lambda x: x.start(), threads)