from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    """Using tensorboard"""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
