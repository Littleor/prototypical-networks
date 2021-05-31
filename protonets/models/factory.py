MODEL_REGISTRY = {}


# 一个装饰器 用于注册model
def register_model(model_name):
    def decorator(f):
        MODEL_REGISTRY[model_name] = f  # 将module对应的函数保存到MODEL_REGISTRY中
        return f

    return decorator


def get_model(model_name, model_opt):
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name](**model_opt)
    else:
        raise ValueError("Unknown model {:s}".format(model_name))
