from typing import Any, Self

class YoutubeDL:
    def __init__(
        self, params: dict[str, Any] | None = None, auto_init: bool = True
    ) -> None: ...
    def extract_info(
        self,
        url: str,
        download: bool = True,
        ie_key: str | None = None,
        extra_info: dict[str, Any] | None = None,
        process: bool = True,
        force_generic_extractor: bool = False,
    ) -> dict[str, Any]: ...
    def prepare_filename(
        self,
        info_dict: dict[str, Any],
        dir_type: str = "",
        *,
        outtmpl: str | None = None,
        warn: bool = False,
    ) -> str: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args: Any) -> None: ...
