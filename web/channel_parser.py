"""Parse YouTube channel URLs into identifiers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


@dataclass
class ChannelIdentifier:
    """Parsed YouTube channel identifier."""
    handle: Optional[str] = None      # @handle format
    channel_id: Optional[str] = None  # UC... format
    custom_name: Optional[str] = None # /c/Name format

    @property
    def is_channel_id(self) -> bool:
        return self.channel_id is not None

    @property
    def search_term(self) -> str:
        """Return the best identifier for YouTube API search."""
        if self.channel_id:
            return self.channel_id
        if self.handle:
            return self.handle
        if self.custom_name:
            return self.custom_name
        raise ValueError("No valid identifier found")


def parse_channel_url(url: str) -> ChannelIdentifier:
    """
    Parse a YouTube channel URL into a ChannelIdentifier.

    Supported formats:
    - https://youtube.com/@handle
    - https://www.youtube.com/@handle
    - https://youtube.com/channel/UCxxxxxxxx
    - https://youtube.com/c/ChannelName
    - @handle (bare handle)
    - UCxxxxxxxx (bare channel ID)
    """
    url = url.strip()

    # Bare handle
    if url.startswith("@") and "/" not in url:
        return ChannelIdentifier(handle=url.lstrip("@"))

    # Bare channel ID
    if re.match(r"^UC[\w-]{22}$", url):
        return ChannelIdentifier(channel_id=url)

    # Full URL
    parsed = urlparse(url)
    if not parsed.scheme:
        parsed = urlparse(f"https://{url}")

    path = parsed.path.rstrip("/")

    # /@handle
    handle_match = re.match(r"^/@([\w.-]+)", path)
    if handle_match:
        return ChannelIdentifier(handle=handle_match.group(1))

    # /channel/UCxxx
    channel_match = re.match(r"^/channel/(UC[\w-]+)", path)
    if channel_match:
        return ChannelIdentifier(channel_id=channel_match.group(1))

    # /c/Name
    custom_match = re.match(r"^/c/([\w.-]+)", path)
    if custom_match:
        return ChannelIdentifier(custom_name=custom_match.group(1))

    # /user/Name (legacy)
    user_match = re.match(r"^/user/([\w.-]+)", path)
    if user_match:
        return ChannelIdentifier(custom_name=user_match.group(1))

    # Last resort: treat the last path segment as a handle
    segments = [s for s in path.split("/") if s]
    if segments:
        last = segments[-1]
        if last.startswith("@"):
            return ChannelIdentifier(handle=last.lstrip("@"))
        return ChannelIdentifier(custom_name=last)

    raise ValueError(f"Could not parse channel URL: {url}")
