import inspect


def store_attr():
    curframe = inspect.currentframe()
    if curframe:
        if curframe.f_back:
            prev_locals = curframe.f_back.f_locals
            for k, v in curframe.f_back.f_locals.items():
                if k != "self":
                    setattr(prev_locals["self"], k, v)
