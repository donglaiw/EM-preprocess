from threading import Thread
from torch.multiprocessing import Queue, Lock
import torch.cuda as cuda
import torch


class DiskToHostSliceLoader(object):
    """
    An object responsible for loading EM slices from disk to host memory and saving them to a desired data structure
    using a desired "post" method.
    """
    def __init__(self, load_method, slice_range=None, num_threads=8, num_partitions=1, partition_overlap=2):
        """
        Constructor.
        :param load_method: The method used to load each slice as a PyTorch Tensor with the slice's index as
        its only argument.
        :param slice_range: The list of slice indices wished to be loaded.
        :param num_threads: Number of threads used to load the slices.
        :param num_partitions: Number of partitions to divide the slice_range into.
        Each partition will be loaded independently. Useful when using multiple GPUs.
        :param partition_overlap: How many of the previous partition's slices to be loaded by the current partition. (If
        used with de_flickering, set this to the temporal filter's radius size.)
        For example if one wishes to load 100 slices with num_partitions = 3 and partition_overlap = 2,
        """
        # Arg Check:---------------------------------------------------------------------------------------------------#
        assert type(load_method) == function and load_method.func_code.co_argcount == 1
        if slice_range is None:
            slice_range = range(100)
        assert type(slice_range) == list and len(slice_range) > 0
        assert (type(num_threads) == int or type(num_threads) == long) and num_threads > 0
        assert (type(partition_overlap) == int or type(partition_overlap == long)) and partition_overlap > -1
        # -------------------------------------------------------------------------------------------------------------#
        self.load = load_method
        self.parts = [Queue()] * num_partitions
        self.post = None
        slc_per_part = len(slice_range) / num_partitions
        for i, part in enumerate(self.parts):
            part_range = slice_range[slc_per_part * i: slc_per_part * (i + 1)]
            if i != 0:
                left = slice_range[slc_per_part * i - partition_overlap: slc_per_part * i]
                part_range = left + part_range
            if i == len(self.parts) - 1:
                right = slice_range[slc_per_part * (i + 1): slice_range[-1]]
                part_range = part_range + right
            map(part.put, part_range)

        def thread_target(start_idx):
            counter = start_idx
            while ([partition.empty() for partition in self.parts]).count(True) != len(self.parts):
                part_idx = counter % len(self.parts)
                if not self.parts[part_idx].empty():
                    slc_index = self.parts[part_idx].get()
                    try:
                        slc = self.load(slc_index)
                        self.post(slc_index, slc)
                    except RuntimeError:
                        print "Failed to load slice number %d" % slc_index
                        pass
                counter = counter + 1
            return

        self.threads = [Thread(target=thread_target, args=(i,)) for i in range(num_threads)]

    def set_post_method(self, post_method):
        """
        Sets the method used to post the loaded slices along with their indices. The output of this function can be
        anything the user desires, but the input arguments must be in the form of (slice_index, _slice).
        :param post_method: The method to be called upon loading a slice with arguments (slice_index, _slice).
        :return: None
        """
        # Arg Check:---------------------------------------------------------------------------------------------------#
        assert type(post_method) == function and post_method.func_code.co_argcount == 2
        # -------------------------------------------------------------------------------------------------------------#
        self.post = post_method

    def start(self):
        """
        Starts the loading operation. Before calling this method, the post_method property should be set.
        :return: None.
        """
        # Arg Check:---------------------------------------------------------------------------------------------------#
        if type(self.post) is None:
            raise TypeError("The post method should be set before starting.")
        # -------------------------------------------------------------------------------------------------------------#
        map(lambda x: x.start(), self.threads)

    def join(self):
        """
        Waits on the loading process to be finished.
        :return: None.
        """
        try:
            map(lambda x: x.join(), self.threads)
        except RuntimeError:
            raise RuntimeError("Did you start the loader before calling join?")


class HostToDeviceSliceLoader(object):
    """
    An object responsible for loading EM slices from memory to (multiple) GPUs asynchronously and saving the handles
    to a desired data structure via a "post" method. (Divide the partititon into gpus.)
    """
    def __init__(self, slice_range=None, num_devices=None, num_threads=8):
        """
        Constructor.
        :param num_devices: Number of CUDA capable devices to be used.
        :param num_threads: Number of threads to be used.
        """
        if slice_range is None:
            self.left_slices = range(100)
        else:
            assert type(slice_range) == list
            self.left_slices = slice_range
        if not cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        if cuda.device_count() == 0:
            raise RuntimeError("No GPU available.")
        if num_devices is None:
            num_devices = cuda.device_count()
        else:
            assert (type(num_devices) is int or type(num_devices) is long) and num_devices > 0 and \
            num_devices <= cuda.device_count()
        assert (type(num_threads) is int or type(num_threads) is long) and num_threads > 0
        self.jobs_queue = Queue()
        self.load_method = None
        self.post_method = None

        def thread_target():
            while len(self.left_slices) != 0:
                if not self.jobs_queue.empty():
                    slc_idx, slc_cpu = self.jobs_queue.get()
                    cur_stream = cuda.Stream(dev_id)
                    with cuda.device(dev_id):
                        with cuda.stream(cur_stream):
                            slc_cuda = slc_cpu.cuda(non_blocking=True)
                    self.slc_grid_put(slc_idx, dev_id, cur_stream, slc_cuda)

        self.threads = [Thread(target=thread_target) for _ in range(num_threads)]




    def set_post_queue_copy(self, func):
        self.slc_grid_put = func

    def start(self):
        map(lambda x: x.start(), self.threads)

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
        h_to_d_loader.set_post_method(lambda slc_idx, dev_idx, stream, slc_cuda:
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
                                    input = torch.stack([element[1] for element in self.grid[j]])
                                    self.d_to_h_put(slc_idx, calc_stream, self.pre_process(input))
                                self.grid[j] = 0
                            self.lock_array[j].release()
        threads = [Thread(target=thread_target) for _ in range(self.num_threads)]
        map(lambda x: x.setDaemon(True), threads)
        map(lambda x: x.start(), threads)

class DeviceToHostSliceLoader(object):
    def __init__(self, slice_grid, num_threads = 2):
        self.queue = Queue()
        slice_grid.set_post_method(lambda slc_idx, calc_stream, slc: self.queue.put((slc_idx, calc_stream, slc)))
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
        #map(lambda x: x.setDaemon(True), threads)
        map(lambda x: x.start(), threads)


class HostToDiskSliceLoader(object):

    def __init__(self, d_to_h, save_slice_to_disk, slice_range, num_threads=8):
        self.disk_save = save_slice_to_disk
        self.num_threads = num_threads
        self.queue = Queue()
        d_to_h.set_post_method(lambda slc_idx, slc_cpu: self.queue.put((slc_idx, slc_cpu)))
        self.slice_range = slice_range

    def set_post_queue_copy(self, func):
        self.put = func

    def start(self):
        def thread_target():
            while len(self.slice_range) != 0:
                if not self.queue.empty():
                    slc_index = self.queue.get()
                    self.disk_save(slc_index)
                    self.slice_range.remove(slc_index)
        self.threads = [Thread(target=thread_target) for _ in range(self.num_threads)]
        #map(lambda x: x.setDaemon(True), threads)
        map(lambda x: x.start(), self.threads)

    def join(self):
        map(lambda x: x.join(), self.threads)


class Manager(object):
    def __init__(self, disk_loader, disk_saver, slice_range, med_rad=2, num_devices=1):
        self.d_to_r = DiskToHostSliceLoader(disk_loader, slice_range)
        print "d_to_r"
        self.r_to_d = HostToDeviceSliceLoader(self.d_to_r)
        self.grid = SliceGrid(num_devices, slice_range, med_rad, self.r_to_d)
        self.d_to_r = DeviceToHostSliceLoader(self.grid)
        self.r_to_d = HostToDiskSliceLoader(self.d_to_r, disk_saver, slice_range)

    def set_target(self, func):
        self.grid.set_pre_process_method(func)


    def start(self):
        self.d_to_r.start()
        print "d_to_r"
        self.r_to_d.start()
        print "r_to_d"
        self.grid.start()
        self.d_to_r.start()
        self.r_to_d.start()

    def wait(self):
        self.r_to_d.join()