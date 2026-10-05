#!/usr/bin/env python

"""Segment-readable raw DEFLATE support and versioned side indexes."""

import binascii
import hashlib
import json
import zlib
from bisect import bisect_right
from collections import namedtuple
from zipfile import ZIP_DEFLATED


SIDE_INDEX_VERSION = 1
DEFAULT_SEGMENT_SIZE = 1024 * 1024
DIGEST_ALGORITHM = "sha256"
SEGMENTED_COMPRESSION = "deflate-raw-z-full-flush"


class SegmentedDeflater(object):
    """Compress data with independent raw DEFLATE restart points."""

    def __init__(self, level=None, segment_size=DEFAULT_SEGMENT_SIZE):
        if not isinstance(segment_size, int) or isinstance(segment_size, bool):
            raise TypeError("segment_size must be an integer")
        if segment_size <= 0:
            raise ValueError("segment_size must be greater than 0")

        self.segment_size = segment_size
        self._level = zlib.Z_DEFAULT_COMPRESSION if level is None else level
        self._compressor = zlib.compressobj(self._level, zlib.DEFLATED, -15)
        self.uncompressed_offset = 0
        self.compressed_offset = 0
        self.until_checkpoint = segment_size
        self.checkpoints = [(0, 0)]

    def _emit(self, data):
        self.compressed_offset += len(data)
        return data

    def compress(self, data):
        if not data:
            return b""

        out = []
        while data:
            if self.until_checkpoint <= 0:
                flushed = self._compressor.flush(zlib.Z_FULL_FLUSH)
                if flushed:
                    out.append(self._emit(flushed))
                self.checkpoints.append((
                    self.uncompressed_offset,
                    self.compressed_offset
                ))
                self.until_checkpoint = self.segment_size

            chunk = data[:self.until_checkpoint]
            data = data[self.until_checkpoint:]
            self.until_checkpoint -= len(chunk)
            self.uncompressed_offset += len(chunk)
            compressed = self._compressor.compress(chunk)
            if compressed:
                out.append(self._emit(compressed))

        return b"".join(out)

    def flush(self):
        return self._emit(self._compressor.flush())


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True
    ).encode("ascii")


def make_member_id(archive_id, entry):
    """Return the stable identity digest for an indexed archive member."""
    if not isinstance(archive_id, str) or len(archive_id) != 64:
        raise ValueError("Invalid archive_id")

    payload = dict(entry)
    payload.pop("member_id", None)
    payload["archive_id"] = archive_id
    digest = hashlib.sha256(_canonical_json(payload))
    return digest.hexdigest()


def _validate_checkpoints(entry):
    points = entry.get("checkpoints")
    if not isinstance(points, list) or not points:
        raise ValueError("Member checkpoints must be a non-empty list")

    normalized = []
    previous_uncompressed = -1
    previous_compressed = -1
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("Each checkpoint must be an object")
        uncompressed = point.get("uncompressed_offset")
        compressed = point.get("compressed_offset")
        if not _is_int(uncompressed) or not _is_int(compressed):
            raise ValueError("Checkpoint offsets must be integers")
        if uncompressed < 0 or compressed < 0:
            raise ValueError("Checkpoint offsets cannot be negative")
        if uncompressed <= previous_uncompressed:
            raise ValueError("Uncompressed checkpoint offsets must increase")
        if compressed <= previous_compressed:
            raise ValueError("Compressed checkpoint offsets must increase")
        if uncompressed > entry["file_size"]:
            raise ValueError("Checkpoint is outside the uncompressed member")
        if compressed > entry["compressed_size"]:
            raise ValueError("Checkpoint is outside the compressed member")
        normalized.append({
            "uncompressed_offset": uncompressed,
            "compressed_offset": compressed
        })
        previous_uncompressed = uncompressed
        previous_compressed = compressed

    if normalized[0] != {
        "uncompressed_offset": 0,
        "compressed_offset": 0
    }:
        raise ValueError("The first checkpoint must be at offset (0, 0)")
    return normalized


def validate_side_index(side_index):
    """Validate a side index and return it with normalized structures."""
    if not isinstance(side_index, dict):
        raise ValueError("Side index must be an object")
    if side_index.get("version") != SIDE_INDEX_VERSION:
        raise ValueError("Unsupported side index version")
    if side_index.get("compression") != SEGMENTED_COMPRESSION:
        raise ValueError("Unsupported segmented compression format")
    if side_index.get("digest_algorithm") != DIGEST_ALGORITHM:
        raise ValueError("Unsupported side index digest algorithm")

    archive_id = side_index.get("archive_id")
    if not isinstance(archive_id, str):
        raise ValueError("Invalid archive_id")
    try:
        binascii.unhexlify(archive_id)
    except (binascii.Error, TypeError):
        raise ValueError("Invalid archive_id")

    dimensions = [
        "archive_size",
        "central_directory_offset",
        "central_directory_size",
        "archive_footer_offset",
        "archive_footer_size",
    ]
    for name in dimensions:
        if not _is_int(side_index.get(name)) or side_index[name] < 0:
            raise ValueError("{} must be a non-negative integer".format(name))

    if side_index["central_directory_offset"] > side_index["archive_size"]:
        raise ValueError("Central directory is outside the archive")
    if side_index["central_directory_offset"] != side_index["archive_footer_offset"]:
        raise ValueError("Invalid archive footer offset")
    if side_index["central_directory_size"] > side_index["archive_footer_size"]:
        raise ValueError("Central directory exceeds its archive footer")
    if (
        side_index["archive_footer_offset"] +
        side_index["archive_footer_size"]
    ) != side_index["archive_size"]:
        raise ValueError("Archive footer does not end at archive_size")

    members = side_index.get("members")
    if not isinstance(members, list):
        raise ValueError("Side index members must be a list")

    seen_ordinals = set()
    normalized_members = []
    for entry in members:
        if not isinstance(entry, dict):
            raise ValueError("Each side index member must be an object")

        fields = [
            "ordinal", "file_size", "compressed_size", "crc32",
            "data_offset", "local_header_offset", "segment_size"
        ]
        for name in fields:
            if not _is_int(entry.get(name)) or entry[name] < 0:
                raise ValueError("{} must be a non-negative integer".format(name))

        if not isinstance(entry.get("name"), str) or not entry["name"]:
            raise ValueError("Member name is required")
        if entry["ordinal"] in seen_ordinals:
            raise ValueError("Duplicate member ordinal")
        seen_ordinals.add(entry["ordinal"])
        if entry["compress_type"] != ZIP_DEFLATED:
            raise ValueError("Only DEFLATE members can be segmented")
        if entry["crc32"] > 0xFFFFFFFF:
            raise ValueError("Invalid member CRC32")
        if entry["segment_size"] <= 0:
            raise ValueError("Invalid segment size")
        if entry["local_header_offset"] > entry["data_offset"]:
            raise ValueError("Member data cannot start before its header")
        if (
            entry["data_offset"] + entry["compressed_size"] >
            side_index["central_directory_offset"]
        ):
            raise ValueError("Member data is outside its archive")

        points = _validate_checkpoints(entry)
        normalized = dict(entry)
        normalized["checkpoints"] = points
        if normalized.get("member_id") != make_member_id(archive_id, normalized):
            raise ValueError("Member identity does not match the side index")
        normalized_members.append(normalized)

    result = dict(side_index)
    result["members"] = normalized_members
    return result


def _find_member(side_index, member):
    if _is_int(member):
        matches = [x for x in side_index["members"] if x["ordinal"] == member]
    elif isinstance(member, str):
        matches = [x for x in side_index["members"] if x["name"] == member]
    else:
        raise TypeError("member must be a member name or ordinal")

    if len(matches) != 1:
        raise LookupError("Side index does not uniquely identify the member")
    return matches[0]


SegmentedRangeReadResult = namedtuple(
    "SegmentedRangeReadResult",
    [
        "data",
        "compressed_offset",
        "compressed_length",
        "compressed_range",
        "range_crc32",
        "crc32_status",
        "crc32_scope",
        "member_crc_verified",
    ]
)


def read_member_range(side_index, member, start, end, read_compressed):
    """Read and decompress one uncompressed range of an indexed member.

    ``read_compressed(offset, length)`` must return exactly ``length`` archive
    bytes. The returned result intentionally verifies only the requested range
    bytes; it never claims that the complete member CRC was verified.
    """
    index = validate_side_index(side_index)
    entry = _find_member(index, member)

    if not _is_int(start) or not _is_int(end):
        raise ValueError("Range offsets must be integers")
    if start < 0 or end < start or end > entry["file_size"]:
        raise ValueError("Requested range is outside the member")

    footer = read_compressed(
        index["archive_footer_offset"],
        index["archive_footer_size"]
    )
    footer = bytes(footer)
    if len(footer) != index["archive_footer_size"]:
        raise IOError("Archive byte source returned a short footer read")
    archive_id = hashlib.sha256(footer).hexdigest()
    if archive_id != index["archive_id"]:
        raise ValueError("Side index is not bound to this archive")

    if start == end:
        compressed_offset = entry["data_offset"]
        return SegmentedRangeReadResult(
            b"", compressed_offset, 0,
            (compressed_offset, compressed_offset),
            0, "range-computed", "requested-range", False
        )

    uncompressed = [point["uncompressed_offset"] for point in entry["checkpoints"]]
    compressed = [point["compressed_offset"] for point in entry["checkpoints"]]
    first_segment = bisect_right(uncompressed, start) - 1
    last_segment = bisect_right(uncompressed, end - 1) - 1

    compressed_start = compressed[first_segment]
    if last_segment + 1 < len(compressed):
        compressed_end = compressed[last_segment + 1]
    else:
        compressed_end = entry["compressed_size"]

    absolute_start = entry["data_offset"] + compressed_start
    absolute_end = entry["data_offset"] + compressed_end
    compressed_length = absolute_end - absolute_start
    compressed_data = read_compressed(absolute_start, compressed_length)
    compressed_data = bytes(compressed_data)
    if len(compressed_data) != compressed_length:
        raise IOError("Archive byte source returned a short compressed read")

    decompressor = zlib.decompressobj(-15)
    data = decompressor.decompress(compressed_data)
    data += decompressor.flush()
    if decompressor.unused_data or decompressor.unconsumed_tail:
        raise IOError("Compressed member segment contained unexpected bytes")

    relative_start = start - uncompressed[first_segment]
    relative_end = end - uncompressed[first_segment]
    if len(data) < relative_end:
        raise IOError("Compressed member segment did not decode completely")
    data = data[relative_start:relative_end]
    if len(data) != end - start:
        raise IOError("Decoded data did not match the requested range")

    return SegmentedRangeReadResult(
        data,
        absolute_start,
        compressed_length,
        (absolute_start, absolute_end),
        binascii.crc32(data) & 0xFFFFFFFF,
        "range-computed",
        "requested-range",
        False
    )
