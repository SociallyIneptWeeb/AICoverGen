"""
Module which defines custom exception and enumerations used when
instiating and re-raising those exceptions.
"""

from enum import StrEnum

from ultimate_rvc.typing_extra import StrPath


class Entity(StrEnum):
    """Enumeration of entities that can be provided."""

    DIRECTORY = "directory"
    DIRECTORIES = "directories"
    FILE = "file"
    FILES = "files"
    URL = "URL"
    MODEL_NAME = "model name"
    MODEL_NAMES = "model names"
    MODEL_FILE = "model file"
    SOURCE = "source"
    SONG_DIR = "song directory"
    AUDIO_TRACK = "audio track"
    AUDIO_TRACK_GAIN_PAIRS = "pairs of audio track and gain"
    SONG = "song"
    VOCALS_TRACK = "vocals track"
    INSTRUMENTALS_TRACK = "instrumentals track"
    BACKUP_VOCALS_TRACK = "backup vocals track"
    MAIN_VOCALS_TRACK = "main vocals track"


class Location(StrEnum):
    """Enumeration of locations where entities can be found."""

    INTERMEDIATE_AUDIO_ROOT = "the root of the intermediate audio base directory"
    OUTPUT_AUDIO_ROOT = "the root of the output audio directory"
    EXTRACTED_ZIP_FILE = "extracted zip file"


class UIMessage(StrEnum):
    """
    Enumeration of messages that can be displayed in the UI
    in place of core exception messages.
    """

    NO_AUDIO_TRACK = "No audio tracks provided."
    NO_SONG_DIR = "No song directory selected."
    NO_SONG_DIRS = (
        "No song directories selected. Please select one or more song directories"
        " containing intermediate audio files to delete."
    )
    NO_OUTPUT_AUDIO_FILES = (
        "No files selected. Please select one or more output audio files to delete."
    )
    NO_UPLOADED_FILES = "No files selected."
    NO_VOICE_MODEL = "No voice model selected."
    NO_VOICE_MODELS = "No voice models selected."
    NO_SOURCE = (
        "No source provided. Please provide a valid Youtube URL, local audio file"
        " or song directory."
    )


class NotProvidedError(ValueError):
    """Raised when an entity is not provided."""

    def __init__(self, entity: Entity, ui_msg: UIMessage | None = None) -> None:
        """
        Initialize a NotProvidedError instance.

        Exception message will be formatted as:

        "No `<entity>` provided."

        Parameters
        ----------
        entity : Entity
            The entity that was not provided.
        ui_msg : UIMessage, default=None
            Message which, if provided, is displayed in the UI
            instead of the default exception message.

        """
        super().__init__(f"No {entity} provided.")
        self.ui_msg = ui_msg


class NotFoundError(OSError):
    """Raised when an entity is not found."""

    def __init__(
        self,
        entity: Entity,
        location: StrPath | Location,
        is_path: bool = True,
    ) -> None:
        """
        Initialize a NotFoundError instance.

        Exception message will be formatted as:

        "`<entity>` not found `(`in `|` as:`)` `<location>`."

        Parameters
        ----------
        entity : Entity
            The entity that was not found.
        location : StrPath | Location
            The location where the entity was not found.
        is_path : bool, default=True
            Whether the location is a path to the entity.

        """
        proposition = "at:" if is_path else "in"
        entity_cap = entity.capitalize() if not entity.isupper() else entity
        super().__init__(
            f"{entity_cap} not found {proposition} {location}.",
        )


class VoiceModelNotFoundError(OSError):
    """Raised when a voice model is not found."""

    def __init__(self, name: str) -> None:
        r"""
        Initialize a VoiceModelNotFoundError instance.

        Exception message will be formatted as:

        'Voice model with name "`<name>`" not found.'

        Parameters
        ----------
        name : str
            The name of the voice model that was not found.

        """
        super().__init__(f'Voice model with name "{name}" not found.')


class VoiceModelExistsError(OSError):
    """Raised when a voice model already exists."""

    def __init__(self, name: str) -> None:
        r"""
        Initialize a VoiceModelExistsError instance.

        Exception message will be formatted as:

        "Voice model with name '`<name>`' already exists. Please provide
        a different name for your voice model."

        Parameters
        ----------
        name : str
            The name of the voice model that already exists.

        """
        super().__init__(
            f'Voice model with name "{name}" already exists. Please provide a different'
            " name for your voice model.",
        )


class InvalidLocationError(OSError):
    """Raised when an entity is in a wrong location."""

    def __init__(self, entity: Entity, location: Location, path: StrPath) -> None:
        r"""
        Initialize an InvalidLocationError instance.

        Exception message will be formatted as:

        "`<entity>` should be located in `<location>` but found at:
        `<path>`"

        Parameters
        ----------
        entity : Entity
            The entity that is in a wrong location.
        location : Location
            The correct location for the entity.
        path : StrPath
            The path to the entity.

        """
        entity_cap = entity.capitalize() if not entity.isupper() else entity
        super().__init__(
            f"{entity_cap} should be located in {location} but found at: {path}",
        )


class HttpUrlError(OSError):
    """Raised when a HTTP-based URL is invalid."""

    def __init__(self, url: str) -> None:
        """
        Initialize a HttpUrlError instance.

        Exception message will be formatted as:

        "Invalid HTTP-based URL: `<url>`"

        Parameters
        ----------
        url : str
            The invalid HTTP-based URL.

        """
        super().__init__(
            f"Invalid HTTP-based URL: {url}",
        )


class YoutubeUrlError(OSError):
    """
    Raised when an URL does not point to a YouTube video or
    , potentially, a Youtube playlist.
    """

    def __init__(self, url: str, playlist: bool) -> None:
        """
        Initialize a YoutubeURlError instance.

        Exception message will be formatted as:

        "URL does not point to a YouTube video `[`or playlist`]`:
         `<url>`"

        Parameters
        ----------
        url : str
            The URL that does not point to a YouTube video or playlist.
        playlist : bool
            Whether the URL might point to a YouTube playlist.

        """
        suffix = "or playlist" if playlist else ""
        super().__init__(
            f"Not able to access Youtube video {suffix} at: {url}",
        )


class UploadLimitError(ValueError):
    """Raised when the upload limit for an entity is exceeded."""

    def __init__(self, entity: Entity, limit: str | float) -> None:
        """
        Initialize an UploadLimitError instance.

        Exception message will be formatted as:

        "At most `<limit>` `<entity>` can be uploaded."

        Parameters
        ----------
        entity : Entity
            The entity for which the upload limit was exceeded.
        limit : str
            The upload limit.

        """
        super().__init__(f"At most {limit} {entity} can be uploaded.")


class UploadFormatError(ValueError):
    """
    Raised when one or more uploaded entities have an invalid format
    .
    """

    def __init__(self, entity: Entity, formats: list[str], multiple: bool) -> None:
        """
        Initialize an UploadFileFormatError instance.


        Exception message will be formatted as:

        "Only `<entity>` with the following formats can be uploaded
        `(`by themselves | together`)`: `<formats>`."

        Parameters
        ----------
        entity : Entity
            The entity that was uploaded with an invalid format.
        formats : list[str]
            Valid formats.
        multiple : bool
            Whether multiple entities are uploaded.

        """
        suffix = "by themselves" if not multiple else "together (at most one of each)"
        super().__init__(
            f"Only {entity} with the following formats can be uploaded {suffix}:"
            f" {', '.join(formats)}.",
        )
