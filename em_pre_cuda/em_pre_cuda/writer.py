import torch
from torch.multiprocessing import Pool, Lock, Condition, Manager
import cv2

class SlicerWriter:
    def __init__(self, num_slices):
        self.num_slices = num_slices
        self.queue_lock = Lock()
        self.cond_empty = Condition(self.queue_lock)
        manager = Manager()
        self.queue = manager.Queue()

    def start(self, num_processes):
        self.processes = Pool(num_processes)
        self.processes.map_async(self._writer_target(), range(self.num_slices))

    def post(self, image, path):
        self.queue_lock.acquire()
        self.queue.put((image, path))
        self.queue_lock.release()

    def _writer_target(self):
        self.queue_lock.acquire()
        image, path = self.queue.get()
        self.queue_lock.release()
        if image.is_cuda:
            image = image.cpu()
        cv2.imwrite(path, image.numpy())
