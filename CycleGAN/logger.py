try:
    import tensorflow as tf
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tf = None

class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        if tf is None:
            self.writer = None
        else:
            self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if self.writer is None:
            return
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
