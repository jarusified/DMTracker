import datetime

def epoch_to_timestamp(epoch):
    """
    Converts an epoch time to a timestamp.
    """
    if(isinstance(epoch, str)):
        epoch = float(epoch)/10**9

    print(epoch)

    return datetime.datetime.fromtimestamp(epoch).strftime('%c')

def add_duration_to_timestamp(epoch, duration):
    """
    Adds a duration to an epoch time.
    """
    if(isinstance(epoch, str)):
        epoch = float(epoch)/10**9

    return epoch + duration