import logging
import os

class Logger:
    def __init__(self, log_file, level=logging.ERROR):
        self.log_file = log_file
        self.level = level
        self.configure_logging()

    def configure_logging(self):
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Clear the log file
        with open(self.log_file, 'w'):
            pass

        # Basic configuration for logging
        logging.basicConfig(filename=self.log_file, 
                            level=self.level, 
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    def log_error(self, message, exc_info=False):
        logging.error(message, exc_info=exc_info)

    def log_info(self, message):
        logging.info(message)
