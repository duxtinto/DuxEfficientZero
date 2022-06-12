import cv2


def arr_to_str(arr):
    """To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
    This function encodes the observation numpy arr to the jpeg strings
    """
    img_str = cv2.imencode('.jpg', arr)[1].tobytes()

    return img_str
