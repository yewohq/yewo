"""Choose the public Android collection for injected media."""

import mimetypes


def android_media_directory(local_path: str) -> str:
    media_type, _ = mimetypes.guess_type(local_path)
    if media_type:
        category = media_type.split("/", 1)[0]
        if category == "audio":
            return "Music"
        if category == "video":
            return "Movies"
        if category == "image":
            return "Pictures"
    return "Download"
