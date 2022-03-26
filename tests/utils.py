def try_assert(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            assert False, f"{func.__name__} fucntion error: {e}"
    return wrapper
