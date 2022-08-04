from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger(object):
    """
    TensorBoardLogger is a very simple wrap of tensorboard SummaryWriter class that logs the training and validation loss
    """

    def __init__(self, directory):
        self.epoch_step = 0
        self.iter_step = 0
        self.sw = SummaryWriter(log_dir=directory)

    def log_iter(self, losses, prefix=""):
        prefix = prefix + "_iter"

        for k, v in losses:
            assert isinstance(v, (int, float, complex)) and not isinstance(v, bool), "loss value must be a number"
            self.log_scalar(prefix + k, v, self.iter_step)
        self.iter_step += 1

    def log_epoch(self, losses, prefix=""):
        prefix = prefix + "_epoch"

        for k, v in losses:
            assert isinstance(v, (int, float, complex)) and not isinstance(v, bool), "loss value must be a number"
            self.log_scalar(prefix + k, v, self.epoch_step)
        self.epoch_step += 1

    def log_scalar(self, name, value, step):
        self.sw.add_scalar(name, value, step)

    def log_histogram(self, name, values, step):
        self.sw.add_histogram(name, values, step)

    def log_image(self, name, img, step):
        self.sw.add_image(name, img, step)

    def close(self):
        self.sw.close()

    def __del__(self):
        self.close()
