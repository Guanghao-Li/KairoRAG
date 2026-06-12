"""Provider 层统一异常。"""


class KairoProviderError(RuntimeError):
    """外部 provider 初始化或调用失败时抛出的清晰错误。"""
