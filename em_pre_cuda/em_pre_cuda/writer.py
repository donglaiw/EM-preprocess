import torch
from torch.multiprocessing import Pool, Condition, Manager, Queue
import cv2
from os import path, makedirs

class SlicerWriter:
    def __init__(self, writer, num_processes=5, num_slices=100):
        self.writer = writer
        self.num_slices = num_slices
        manager = Manager()
        self.cond_empty = manager.Condition()
        self.queue = manager.Queue()
        self.processes = Pool(num_processes)

    def start(self):
        self.processes.map_async(self.writer_process, range(self.num_slices))


    def post(self, image, idx):
        self.queue.put((image, idx))

    def writer_process(self):
        self.cond_empty.acquire()
        while self.queue.empty():
            self.cond_empty.wait()
        image, idx = self.queue.get()
        self.writer(image.cpu().numpy(), idx)


def png_slice_writer(output_dir, image_name, num_processes=5, num_slices=100):
    def writer(image, idx):
        output_dir_c = output_dir
        img_name_c = image_name
        if '%' in output_dir_c:
            output_dir_c = output_dir_c % idx
        else:
            img_name_c = img_name_c % idx
        img_path = (output_dir + image_name) % idx
        if not path.exists(output_dir_c):
            makedirs(output_dir_c)
        cv2.imwrite(img_path, image)
        
    return SlicerWriter(writer, num_processes, num_slices)
