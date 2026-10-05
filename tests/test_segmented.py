import copy
import hashlib
import io
import json
import zipfile
import zlib

import pytest

import zipstream
from zipstream import (
    SIDE_INDEX_VERSION,
    SEGMENTED_COMPRESSION,
    ZipStream,
    make_member_id,
    read_member_range,
)


SEGMENT_SIZE = 97
DATA = (b"highly-repetitive-" * 200)[:5000]


def _archive_and_index():
    zs = ZipStream(
        compress_type=zipfile.ZIP_DEFLATED,
        segmented=True,
        segment_size=SEGMENT_SIZE,
    )
    zs.add(DATA, "large.bin")
    archive = bytes(zs)
    return archive, zs.side_index()


class RangeSource(object):
    def __init__(self, archive):
        self.archive = archive
        self.reads = []

    def __call__(self, offset, length):
        self.reads.append((offset, length))
        return self.archive[offset:offset + length]


def test_segmented_member_is_standard_deflate_zip():
    archive, side_index = _archive_and_index()

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        info = zf.getinfo("large.bin")
        assert info.compress_type == zipfile.ZIP_DEFLATED
        assert zf.read(info) == DATA

    entry = side_index["members"][0]
    assert side_index["version"] == SIDE_INDEX_VERSION
    assert side_index["compression"] == SEGMENTED_COMPRESSION
    assert entry["file_size"] == len(DATA)
    assert entry["compressed_size"] < len(DATA)
    assert len(entry["checkpoints"]) > 5


def test_segmented_ranges_work_with_zip64(monkeypatch):
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 100)
    monkeypatch.setattr(zipstream.ng, "ZIP64_LIMIT", 100)

    zs = ZipStream(
        compress_type=zipfile.ZIP_DEFLATED,
        segmented=True,
        segment_size=37,
    )
    zs.add(DATA, "zip64.bin")
    archive = bytes(zs)
    side_index = zs.side_index()

    assert side_index["central_directory_size"] > 0
    assert (
        side_index["central_directory_offset"] +
        side_index["central_directory_size"] <=
        side_index["archive_size"]
    )

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        assert zf.read("zip64.bin") == DATA

    result = read_member_range(
        side_index, "zip64.bin", 40, 120, RangeSource(archive)
    )
    assert result.data == DATA[40:120]


def test_checkpoints_are_independent_restart_points():
    archive, side_index = _archive_and_index()
    entry = side_index["members"][0]

    previous_uncompressed = 0
    previous_compressed = 0
    checkpoints = entry["checkpoints"] + [{
        "uncompressed_offset": entry["file_size"],
        "compressed_offset": entry["compressed_size"],
    }]

    for current, following in zip(checkpoints, checkpoints[1:]):
        assert current["uncompressed_offset"] >= previous_uncompressed
        assert current["compressed_offset"] >= previous_compressed
        previous_uncompressed = current["uncompressed_offset"]
        previous_compressed = current["compressed_offset"]

        start = entry["data_offset"] + current["compressed_offset"]
        end = entry["data_offset"] + following["compressed_offset"]
        compressed_segment = archive[start:end]
        decompressor = zlib.decompressobj(-15)
        uncompressed = decompressor.decompress(compressed_segment)
        uncompressed += decompressor.flush()

        assert not decompressor.unused_data
        assert not decompressor.unconsumed_tail
        assert uncompressed == DATA[
            current["uncompressed_offset"]:following["uncompressed_offset"]
        ]


@pytest.mark.parametrize("start,end", [
    (0, 1),
    (1, SEGMENT_SIZE),
    (SEGMENT_SIZE - 1, SEGMENT_SIZE + 1),
    (SEGMENT_SIZE, 2 * SEGMENT_SIZE),
    (3 * SEGMENT_SIZE + 7, 17 * SEGMENT_SIZE - 13),
    (len(DATA) - 1, len(DATA)),
    (0, len(DATA)),
])
def test_member_ranges_match_full_decompression(start, end):
    archive, side_index = _archive_and_index()
    source = RangeSource(archive)

    result = read_member_range(side_index, "large.bin", start, end, source)

    assert result.data == DATA[start:end]
    assert result.compressed_range == (
        result.compressed_offset,
        result.compressed_offset + result.compressed_length
    )
    assert result.range_crc32 == zlib.crc32(DATA[start:end]) & 0xFFFFFFFF
    assert result.crc32_status == "range-computed"
    assert result.crc32_scope == "requested-range"
    assert result.member_crc_verified is False

    assert source.reads[0] == (
        side_index["archive_footer_offset"],
        side_index["archive_footer_size"]
    )
    actual_compressed_read = source.reads[1]
    assert actual_compressed_read == (
        result.compressed_offset,
        result.compressed_length
    )
    assert result.compressed_offset >= side_index["members"][0]["data_offset"]
    assert (
        result.compressed_offset + result.compressed_length <=
        side_index["members"][0]["data_offset"] +
        side_index["members"][0]["compressed_size"]
    )


def test_empty_member_range_records_no_compressed_data():
    archive, side_index = _archive_and_index()
    source = RangeSource(archive)

    result = read_member_range(side_index, "large.bin", 0, 0, source)

    assert result.data == b""
    assert result.compressed_length == 0
    assert result.range_crc32 == 0
    assert result.member_crc_verified is False
    assert source.reads == [
        (side_index["archive_footer_offset"], side_index["archive_footer_size"]),
    ]


def test_side_index_is_bound_to_archive_and_member():
    archive, side_index = _archive_and_index()
    entry = side_index["members"][0]
    footer = archive[side_index["archive_footer_offset"]:]

    assert side_index["archive_id"] == hashlib.sha256(footer).hexdigest()
    assert entry["member_id"] == make_member_id(side_index["archive_id"], entry)

    wrong_archive = bytearray(archive[:-1] + bytes([(archive[-1] + 1) % 256]))
    with pytest.raises(ValueError, match="not bound"):
        read_member_range(
            side_index, "large.bin", 0, 10,
            lambda offset, length: bytes(wrong_archive[offset:offset + length])
        )

    wrong_member = copy.deepcopy(side_index)
    wrong_member["members"][0]["crc32"] ^= 1
    with pytest.raises(ValueError, match="Member identity"):
        read_member_range(
            wrong_member, "large.bin", 0, 10,
            RangeSource(archive)
        )


def test_range_requests_are_bounds_checked():
    archive, side_index = _archive_and_index()
    source = RangeSource(archive)

    for start, end in [(-1, 1), (1, 0), (0, len(DATA) + 1)]:
        with pytest.raises(ValueError, match="outside"):
            read_member_range(side_index, "large.bin", start, end, source)


def test_side_index_checkpoints_are_validated():
    _, side_index = _archive_and_index()

    bad = copy.deepcopy(side_index)
    bad["members"][0]["checkpoints"][1]["compressed_offset"] = 0
    with pytest.raises(ValueError, match="Compressed checkpoint offsets"):
        read_member_range(bad, "large.bin", 0, 10, RangeSource(b""))

    bad = copy.deepcopy(side_index)
    bad["members"][0]["checkpoints"][1]["uncompressed_offset"] = (
        bad["members"][0]["file_size"] + 1
    )
    bad["members"][0]["member_id"] = make_member_id(bad["archive_id"], bad["members"][0])
    with pytest.raises(ValueError, match="outside the uncompressed"):
        read_member_range(bad, "large.bin", 0, 10, RangeSource(b""))

    bad = copy.deepcopy(side_index)
    bad["archive_size"] += 1
    with pytest.raises(ValueError, match="Archive footer does not end"):
        read_member_range(bad, "large.bin", 0, 10, RangeSource(b""))


def test_side_index_json_is_versioned_and_self_describing():
    _, side_index = _archive_and_index()
    encoded = json.dumps(side_index)
    decoded = json.loads(encoded)

    assert decoded["version"] == 1
    assert decoded["digest_algorithm"] == "sha256"
    assert decoded["members"][0]["checkpoints"][0] == {
        "uncompressed_offset": 0,
        "compressed_offset": 0,
    }


def test_empty_segmented_member_has_initial_checkpoint():
    zs = ZipStream(compress_type=zipfile.ZIP_DEFLATED, segmented=True)
    zs.add(b"", "empty.bin")
    archive = bytes(zs)
    side_index = zs.side_index()
    entry = side_index["members"][0]

    assert entry["file_size"] == 0
    assert entry["compressed_size"] == 2
    assert entry["checkpoints"] == [{
        "uncompressed_offset": 0,
        "compressed_offset": 0,
    }]
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        assert zf.read("empty.bin") == b""


def test_segmented_can_be_disabled_for_one_member():
    zs = ZipStream(
        compress_type=zipfile.ZIP_DEFLATED,
        segmented=True,
        segment_size=2,
    )
    zs.add(b"segmented", "segmented.bin")
    zs.add(b"ordinary", "ordinary.bin", segmented=False)
    archive = bytes(zs)
    side_index = zs.side_index()

    assert [x["name"] for x in side_index["members"]] == ["segmented.bin"]
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        assert zf.read("segmented.bin") == b"segmented"
        assert zf.read("ordinary.bin") == b"ordinary"
