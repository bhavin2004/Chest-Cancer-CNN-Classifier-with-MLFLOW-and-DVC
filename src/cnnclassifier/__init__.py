import os,sys,logging
from datetime import datetime
from pathlib import Path

log_file_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

logging_str = "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] - %(message)s"

log_dir = 'logs/'+log_file_name


log_file_path = Path(os.path.join(os.getcwd(),log_dir, log_file_name+'.log'))

# Create logs directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Configure logging to write to a file and console
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)


logger = logging.getLogger('cnnclassifierLogger')


def get_error_message(error_msg,error_detail) -> str:
    """
    Generates a detailed error message including the exception type, value, and traceback.
    
    Args:
        error_msg (Exception): The exception that occurred.
        error_detail (sys): The sys module to access traceback information.
        
    Returns:
        str: A formatted error message.
    """
    _,_,exc_tb = error_detail.exc_info()
    
    filename = exc_tb.tb_frame.f_code.co_filename # type: ignore
    
    error_message = f"Error occurred in script: {filename} at line number: {exc_tb.tb_lineno}\nAnd Error message: {str(error_msg)}"
    
    return error_message


class CustomException(Exception):
    """
    Custom exception class that logs the error message and traceback.
    
    Args:
        Exception: The base exception class.
    """
    
    def __init__(self, error_msg: Exception, error_detail):
        super().__init__(error_msg)
        self.error_message = get_error_message(error_msg, error_detail)
        logger.error(self.error_message)
    
    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        a=10/0
    except Exception as e:
        # print(str(e))  # This will print the formatted error message
        
        logger.info("Custom exception handled successfully.")
        raise CustomException(e, sys) 