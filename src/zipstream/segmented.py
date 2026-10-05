#!/usr/bin/env python

"""Versioned side indexes and random-access reads for segmented DEFLATE data"""

import binascii
import collections
import hashlib
import json
import struct
import zlib
from zipfile import (
    ZIP_DEFLATED,
    stringFileHeader,
    structFileHeader,
)


SIDE_INDEX_VERSION = 1


class SideIndexError(ValueError):
    """Raised when a segmented DEFLATE side index is invalid"""


SegmentedRange = collections.namedtuple(
    "SegmentedRange",
    (
        "data",
        "compressed_start",
        "compressed_end",
        "range_crc32",
        "member_crc_verified",
        "member_identity_verified",
    ),
)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        binascii.unhexlify(value.encode("ascii"))
    except (binascii.Error, ValueError):
        return False
    return True


def load_side_index(side_index):
    """Validate and normalize a side index supplied as JSON or a dict"""
    if isinstance(side_index, (bytes, bytearray)):
        side_index = side_index.decode("utf-8")
    if isinstance(side_index, str):
        try:
            side_index = json.loads(side_index)
        except ValueError as exc:
            raise SideIndexError("Side index is not valid JSON") from exc

    if not isinstance(side_index, dict):
        raise SideIndexError("Side index must be a JSON object")
    if side_index.get("version") != SIDE_INDEX_VERSION:
        raise SideIndexError("Unsupported side index version")

    archive = side_index.get("archive")
    if not isinstance(archive, dict):
        raise SideIndexError("Side index is missing archive metadata")
    archive_size = archive.get("size")
    archive_sha256 = archive.get("sha256")
    if not _is_int(archive_size) or archive_size < 0:
        raise SideIndexError("Invalid archive size")
    if not _is_sha256(archive_sha256):
        raise SideIndexError("Invalid archive SHA-256")

    members = side_index.get("members")
    if not isinstance(members, list):
        raise SideIndexError("Side index is missing a members list")

    seen_indexes = set()
    for position, member in enumerate(members):
        if not isinstance(member, dict):
            raise SideIndexError("Member index entry {} is invalid".format(position))

        name = member.get("name")
        member_index = member.get("index")
        header_offset = member.get("local_header_offset")
        data_offset = member.get("compressed_data_offset")
        crc = member.get("crc32")
        file_size = member.get("file_size")
        compress_size = member.get("compress_size")
        segment_size = member.get("segment_size")
        checkpoints = member.get("checkpoints")

        if not isinstance(name, str) or not name or "\x00" in name:
            raise SideIndexError("Member {} has an invalid name".format(position))
        for label, value in (
            ("index", member_index),
            ("local_header_offset", header_offset),
            ("compressed_data_offset", data_offset),
            ("crc32", crc),
            ("file_size", file_size),
            ("compress_size", compress_size),
            ("segment_size", segment_size),
        ):
            if not _is_int(value) or value < 0:
                raise SideIndexError(
                    "Member {} has an invalid {}".format(name, label)
                )

        if member_index in seen_indexes:
            raise SideIndexError("Duplicate member index: {}".format(member_index))
        seen_indexes.add(member_index)
        if not 0 <= crc <= 0xFFFFFFFF:
            raise SideIndexError("Member {} has an invalid CRC-32".format(name))
        if segment_size < 1:
            raise SideIndexError("Member {} has an invalid segment size".format(name))
        if member.get("compress_type") != ZIP_DEFLATED:
            raise SideIndexError(
                "Member {} does not use segmented DEFLATE".format(name)
            )
        if data_offset < header_offset + 30:
            raise SideIndexError("Member {} has an invalid data offset".format(name))
        if data_offset + compress_size > archive_size:
            raise SideIndexError(
                "Member {} compressed data extends past the archive".format(name)
            )
        if not _is_sha256(member.get("local_header_sha256")):
            raise SideIndexError(
                "Member {} has an invalid local header digest".format(name)
            )
        if not _is_sha256(member.get("compressed_data_sha256")):
            raise SideIndexError(
                "Member {} has an invalid compressed data digest".format(name)
            )

        if not isinstance(checkpoints, list) or not checkpoints:
            raise SideIndexError("Member {} has no checkpoints".format(name))

        previous_compressed = -1
        previous_uncompressed = -1
        for checkpoint_position, checkpoint in enumerate(checkpoints):
            if not isinstance(checkpoint, dict):
                raise SideIndexError(
                    "Member {} has an invalid checkpoint".format(name)
                )
            compressed_offset = checkpoint.get("compressed_offset")
            uncompressed_offset = checkpoint.get("uncompressed_offset")
            if not _is_int(compressed_offset) or not _is_int(uncompressed_offset):
                raise SideIndexError(
                    "Member {} has a non-integer checkpoint offset".format(name)
                )
            if compressed_offset < 0 or uncompressed_offset < 0:
                raise SideIndexError(
                    "Member {} has a negative checkpoint offset".format(name)
                )
            if (
                checkpoint_position > 0
                and (file_size > 0 or compress_size > 0)
                and compressed_offset <= previous_compressed
            ):
                raise SideIndexError(
                    "Member {} compressed offsets are not strictly increasing".format(name)
                )
            if (
                checkpoint_position > 0
                and file_size > 0
                and uncompressed_offset <= previous_uncompressed
            ):
                raise SideIndexError(
                    "Member {} uncompressed offsets are not strictly increasing".format(name)
                )
            if compressed_offset > compress_size:
                raise SideIndexError(
                    "Member {} has a checkpoint past its compressed data".format(name)
                )
            if uncompressed_offset > file_size:
                raise SideIndexError(
                    "Member {} has a checkpoint past its uncompressed data".format(name)
                )
            if checkpoint_position == 0 and (
                compressed_offset != 0 or uncompressed_offset != 0
            ):
                raise SideIndexError(
                    "Member {} does not start with a zero checkpoint".format(name)
                )
            if checkpoint_position > 0:
                if checkpoint_position < len(checkpoints) - 1:
                    if uncompressed_offset != segment_size * checkpoint_position:
                        raise SideIndexError(
                            "Member {} has an incorrectly spaced checkpoint".format(name)
                        )
                elif uncompressed_offset != file_size:
                    raise SideIndexError(
                        "Member {} final checkpoint does not end at its size".format(name)
                    )
            previous_compressed = compressed_offset
            previous_uncompressed = uncompressed_offset

        if file_size > 0 and checkpoints[-1]["compressed_offset"] != compress_size:
            raise SideIndexError(
                "Member {} final checkpoint does not end at its compressed data".format(name)
            )

    return side_index


def get_member(side_index, member):
    """Look up a member by archive member index or unique name"""
    side_index = load_side_index(side_index)
    members = side_index["members"]

    if _is_int(member):
        for entry in members:
            if entry["index"] == member:
                return entry
        raise SideIndexError("Unknown member index: {}".format(member))

    if isinstance(member, str):
        matches = [entry for entry in members if entry["name"] == member]
        if not matches:
            raise SideIndexError("Unknown member: {}".format(member))
        if len(matches) > 1:
            raise SideIndexError(
                "Member name is not unique; use its integer index: {}".format(member)
            )
        return matches[0]

    raise TypeError("member must be an integer member index or member name")


def verify_archive(side_index, archive_data):
    """Verify the complete archive digest recorded in a validated side index"""
    side_index = load_side_index(side_index)
    archive = side_index["archive"]
    digest = hashlib.sha256(archive_data).hexdigest()
    return (
        len(archive_data) == archive["size"]
        and digest == archive["sha256"]
    )


def _checkpoint_at_or_before(member, offset):
    checkpoints = member["checkpoints"]
    found = checkpoints[0]
    for checkpoint in checkpoints:
        if checkpoint["uncompressed_offset"] <= offset:
            found = checkpoint
        else:
            break
    return found


def _checkpoint_at_or_after(member, offset):
    for checkpoint in member["checkpoints"]:
        if checkpoint["uncompressed_offset"] >= offset:
            return checkpoint
    return None


def read_member_range(side_index, member, start, end, read_range, verify_identity=True):
    """Decompress a half-open uncompressed byte range from a segmented member

    read_range(start, end) must return exactly the requested archive bytes. It
    is called with absolute ZIP archive offsets.

    A partial range cannot verify the member's ZIP CRC-32. That verification is
    reported as True only when the complete member is requested and decoded.
    """
    entry = get_member(side_index, member)

    if not _is_int(start) or not _is_int(end):
        raise TypeError("start and end must be integers")
    if start < 0 or end < start or end > entry["file_size"]:
        raise ValueError("Requested range is outside the member")

    header_offset = entry["local_header_offset"]
    data_offset = entry["compressed_data_offset"]
    start_checkpoint = _checkpoint_at_or_before(entry, start)
    compressed_start = data_offset + start_checkpoint["compressed_offset"]

    identity_verified = False
    if verify_identity:
        local_header = read_range(header_offset, data_offset)
        if not isinstance(local_header, (bytes, bytearray)):
            raise SideIndexError("Range source did not return bytes")
        local_header = bytes(local_header)
        header_matches = (
            len(local_header) >= 30
            and local_header[:4] == stringFileHeader
        )
        if header_matches:
            try:
                header = struct.unpack(structFileHeader, local_header[:30])
            except struct.error:
                header_matches = False
            else:
                header_matches = header[3] == ZIP_DEFLATED
        identity_verified = (
            header_matches
            and hashlib.sha256(local_header).hexdigest()
            == entry["local_header_sha256"]
        )
        if not identity_verified:
            raise SideIndexError("Local member header does not match the side index")

    if end == entry["file_size"]:
        compressed_end_relative = entry["compress_size"]
    elif _checkpoint_at_or_before(entry, end)["uncompressed_offset"] == end:
        compressed_end_relative = _checkpoint_at_or_before(
            entry,
            end,
        )["compressed_offset"]
    else:
        end_checkpoint = _checkpoint_at_or_after(entry, end)
        if end_checkpoint is None:
            raise SideIndexError("No checkpoint covers the requested end offset")
        compressed_end_relative = end_checkpoint["compressed_offset"]

    compressed_end = data_offset + compressed_end_relative
    compressed_data = read_range(compressed_start, compressed_end)
    if not isinstance(compressed_data, (bytes, bytearray)):
        raise SideIndexError("Range source did not return bytes")
    compressed_data = bytes(compressed_data)
    if len(compressed_data) != compressed_end - compressed_start:
        raise SideIndexError("Range source returned an unexpected number of bytes")

    decoder = zlib.decompressobj(-15)
    decoded = decoder.decompress(compressed_data)
    decoded += decoder.flush()
    if decoder.unused_data or decoder.unconsumed_tail:
        raise SideIndexError("Compressed range contained unexpected trailing data")

    decoded_start = start - start_checkpoint["uncompressed_offset"]
    decoded_end = decoded_start + (end - start)
    if len(decoded) < decoded_end:
        raise SideIndexError("Compressed range did not decode enough member data")
    result_data = decoded[decoded_start:decoded_end]

    full_member = start == 0 and end == entry["file_size"]
    member_crc_verified = False
    if full_member:
        if result_data and not decoder.eof:
            raise SideIndexError("Complete member did not end at a DEFLATE stream")
        if hashlib.sha256(compressed_data).hexdigest() != entry["compressed_data_sha256"]:
            raise SideIndexError("Compressed data does not match the side index")
        member_crc_verified = (
            zlib.crc32(result_data) & 0xFFFFFFFF == entry["crc32"]
        )

    return SegmentedRange(
        result_data,
        compressed_start,
        compressed_end,
        zlib.crc32(result_data) & 0xFFFFFFFF,
        member_crc_verified,
        identity_verified,
    )
