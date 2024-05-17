class CustomError(Exception):
    def __init__(self, message):
        super(CustomError, self).__init__(message)
        self.message = "Custom Error! "+message
