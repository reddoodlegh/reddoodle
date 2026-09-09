"""
Differential module containing functions for calculating the discrete derivative of a signal.
"""
def diff(timeArray : list, signalArray : list):
    """
    Calculate the discrete derivative of a signal.

    Args:
        timeArray (list): List of time values.
        signalArray (list): List of signal values.

    Returns:
        list: List of discrete derivatives.
    """
    vt = []
    if len(timeArray) == len(signalArray):
        for i in range(1 , len(timeArray)):
            vt.append((signalArray[i] - signalArray[i-1])/ (timeArray[i] - timeArray[i-1]))
            print(vt[i-1])
    else:
        print("Arrays are unequal length")
        return
    return vt