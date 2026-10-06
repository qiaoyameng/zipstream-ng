#!/usr/bin/env python

"""Tests for real multi-volume (spanned) ZIP generation.

These tests deliberately do NOT rely on `zipfile` to read the produced
archives: the standard library has no multi-volume support. Instead they use
an independent reader (`read_spanned`) that locates the (possibly Zip64) end
records, walks the central directory across volumes using the recorded
disk/offset fields, and reassembles and decompresses every member.
"""

import bz2
import io
import lzma
import struct
import zlib

import pytest
import zipfile

import zipstream
from zipstream import SpannedZipStream, VolumeSizeError, SpannedZipError


SPANNING_MARKER = b"PK\x07\x08"


################################
# Independent multi-volume reader
################################

class _Cursor:
    """Linear byte cursor over a list of fixed-size volumes."""

    def __init__(self, vols, volume_size, disk, off):
        self.vols = vols
        self.vs = volume_size
        self.disk = disk
        self.off = off

    def take(self, n):
        out = bytearray()
        while n:
            avail = self.vs - self.off
            take = min(avail, n)
            out += self.vols[self.disk][self.off:self.off + take]
            n -= take
            self.off += take
            if self.off == self.vs:
                self.disk += 1
                self.off = 0
        return bytes(out)

    def skip_zero_padding(self):
        # Indivisible records never span volumes, so any zeros between the
        # cursor and the next signature are volume-tail padding.
        while self.disk < len(self.vols):
            vol = self.vols[self.disk]
            end = min(self.vs, len(vol))
            while self.off < end and vol[self.off] == 0:
                self.off += 1
            if self.off >= self.vs:
                self.disk += 1
                self.off = 0
                continue
            break


def _decompress(comp, body):
    if comp == zipfile.ZIP_STORED:
        return body
    if comp == zipfile.ZIP_DEFLATED:
        return zlib.decompress(body, -15)
    if comp == zipfile.ZIP_BZIP2:
        return bz2.decompress(body)
    if comp == zipfile.ZIP_LZMA:
        filt = lzma._decode_filter_properties(
            lzma.FILTER_LZMA1, body[4:9]
        )
        return lzma.decompress(
            body[9:], format=lzma.FORMAT_RAW, filters=[filt]
        )
    raise AssertionError("unsupported compression method {}".format(comp))


def read_spanned(vols, volume_size):
    """Independently parse a spanned archive.

    Returns (data, meta) where data maps member name -> decompressed bytes and
    meta contains the decoded EOCD/Zip64 location fields and per-member info.
    """
    last = vols[-1]
    j = last.rfind(b"PK\x05\x06")
    assert j >= 0
    ecd = struct.unpack(zipfile.structEndArchive, last[j:j + 22])
    this_disk = ecd[zipfile._ECD_DISK_NUMBER]
    cd_disk = ecd[zipfile._ECD_DISK_START]
    total = ecd[zipfile._ECD_ENTRIES_TOTAL]
    cdsize = ecd[zipfile._ECD_SIZE]
    cd_off = ecd[zipfile._ECD_OFFSET]

    is_zip64 = (
        this_disk == 0xFFFF or cd_disk == 0xFFFF
        or total == 0xFFFF or cdsize == 0xFFFFFFFF or cd_off == 0xFFFFFFFF
    )
    num_disks = this_disk + 1
    if is_zip64:
        lj = last.rfind(b"PK\x06\x07")
        assert lj >= 0, "Zip64 archive without end-of-central-directory locator"
        loc = struct.unpack(
            zipfile.structEndArchive64Locator, last[lj:lj + 20]
        )
        z64_disk, z64_off, num_disks = loc[1], loc[2], loc[3]
        buf = vols[z64_disk][z64_off:z64_off + 56]
        assert buf[:4] == b"PK\x06\x06"
        rec = struct.unpack(zipfile.structEndArchive64, buf)
        cd_disk = rec[5]
        total = rec[7]
        cdsize = rec[8]
        cd_off = rec[9]

    assert num_disks == len(vols)

    # Walk the central directory (it may span volumes)
    cur = _Cursor(vols, volume_size, cd_disk, cd_off)
    members = {}
    order = []
    for _ in range(total):
        cur.skip_zero_padding()
        head = cur.take(46)
        assert head[:4] == b"PK\x01\x02", head[:4]
        cd = struct.unpack(zipfile.structCentralDir, head)
        flags = cd[zipfile._CD_FLAG_BITS]
        comp = cd[zipfile._CD_COMPRESS_TYPE]
        crc = cd[zipfile._CD_CRC]
        csize = cd[zipfile._CD_COMPRESSED_SIZE]
        usize = cd[zipfile._CD_UNCOMPRESSED_SIZE]
        nl = cd[zipfile._CD_FILENAME_LENGTH]
        el = cd[zipfile._CD_EXTRA_FIELD_LENGTH]
        cl = cd[zipfile._CD_COMMENT_LENGTH]
        diskstart = cd[zipfile._CD_DISK_NUMBER_START]
        reloff = cd[zipfile._CD_LOCAL_HEADER_OFFSET]
        name = cur.take(nl).decode("utf-8")
        extra = cur.take(el)
        cur.take(cl)

        pos = 0
        while pos + 4 <= len(extra):
            hid, dlen = struct.unpack("<HH", extra[pos:pos + 4])
            body = extra[pos + 4:pos + 4 + dlen]
            if hid == 0x0001:
                bp = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", body[bp:bp + 8])[0]
                    bp += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", body[bp:bp + 8])[0]
                    bp += 8
                if reloff == 0xFFFFFFFF:
                    reloff = struct.unpack("<Q", body[bp:bp + 8])[0]
                    bp += 8
            pos += 4 + dlen

        members[name] = dict(
            disk=diskstart, off=reloff, comp=comp, flags=flags,
            crc=crc, csize=csize, usize=usize, is_dir=name.endswith("/"),
        )
        order.append(name)

    # Extract and verify every member via its recorded start disk/offset
    data = {}
    for name in order:
        m = members[name]
        if m["is_dir"]:
            data[name] = b""
            continue
        c = _Cursor(vols, volume_size, m["disk"], m["off"])
        c.skip_zero_padding()
        lh = c.take(30)
        assert lh[:4] == b"PK\x03\x04", lh[:4]
        lf = struct.unpack(zipfile.structFileHeader, lh)
        c.take(lf[zipfile._FH_FILENAME_LENGTH])
        c.take(lf[zipfile._FH_EXTRA_FIELD_LENGTH])
        body = c.take(m["csize"])
        raw = _decompress(m["comp"], body)
        assert zlib.crc32(raw) & 0xFFFFFFFF == m["crc"], name
        assert len(raw) == m["usize"], name
        data[name] = raw

    meta = dict(
        total=total,
        num_disks=num_disks,
        is_zip64=is_zip64,
        cd_disk=cd_disk,
        cd_off=cd_off,
        cdsize=cdsize,
    )
    return data, meta, members


def generate(zss):
    """Fully generate a SpannedZipStream, returning (list_of_bytes, zss)"""
    vols = []
    for v in zss.volumes():
        vols.append(b"".join(v))
    return vols, zss


def incompressible(n, seed=0):
    """Deterministic data that doesn't shrink when compressed"""
    out = bytearray(n)
    x = (seed * 2654435761 + 12345) & 0xFFFFFFFF
    for i in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out[i] = (x >> 13) & 0xFF
    return bytes(out)


################################
# Basic structure / interface
################################

def test_volume_size_validation():
    with pytest.raises(ValueError):
        SpannedZipStream(0)
    with pytest.raises(ValueError):
        SpannedZipStream(-10)
    with pytest.raises((ValueError, TypeError)):
        SpannedZipStream(1024.5)
    with pytest.raises(ValueError):
        SpannedZipStream(True)


def test_empty_archive_single_volume():
    zss = SpannedZipStream(1000)
    vols, zss = generate(zss)
    assert zss.is_complete()
    assert zss.num_volumes() == 1
    assert len(vols) == 1
    # marker + EOCD(22)
    assert vols[0][:4] == SPANNING_MARKER
    assert vols[0][-22:][:4] == b"PK\x05\x06"
    data, meta, _ = read_spanned(vols, 1000)
    assert data == {}
    assert meta["total"] == 0
    assert meta["num_disks"] == 1


def test_nonfinal_volumes_exact_size_and_final_flags():
    zss = SpannedZipStream(300)
    for i in range(8):
        zss.add(incompressible(200, i), "f{:02d}.bin".format(i))
    flags = []
    vols = []
    for v in zss.volumes():
        flags.append(v.is_final)
        vols.append(b"".join(v))
    assert zss.is_complete()
    assert zss.num_volumes() == len(vols)
    assert flags == [False] * (len(vols) - 1) + [True]
    # every non-final volume is exactly the limit; the final is <= limit
    assert all(len(x) == 300 for x in vols[:-1])
    assert len(vols[-1]) <= 300


def test_volume_part_names():
    zss = SpannedZipStream(128)
    zss.add(incompressible(400, 3), "a.bin")
    names = [v.part_name("out.zip") for v in zss.volumes()]
    assert names[0] == "out.z01"
    assert names[-1] == "out.zip"
    assert len(names) >= 3


################################
# Member spanning across multiple volumes
################################

def test_stored_member_spans_multiple_volumes():
    vs = 256
    payload = incompressible(1000, 1)
    zss = SpannedZipStream(vs)
    zss.add(payload, "big.bin")
    zss.add(b"tiny", "small.txt")
    vols, zss = generate(zss)

    assert len(vols) > 4
    data, meta, members = read_spanned(vols, vs)
    assert data["big.bin"] == payload
    assert data["small.txt"] == b"tiny"

    # prove the big member's data really crosses volume boundaries
    start = members["big.bin"]["disk"] * vs + members["big.bin"]["off"]
    lf = struct.unpack(
        zipfile.structFileHeader,
        vols[members["big.bin"]["disk"]][
            members["big.bin"]["off"]:members["big.bin"]["off"] + 30
        ],
    )
    body_start = start + 30 + lf[10] + lf[11]
    body_end = body_start + members["big.bin"]["csize"]
    assert body_end // vs > body_start // vs


@pytest.mark.parametrize("comp", [
    zipfile.ZIP_STORED,
    zipfile.ZIP_DEFLATED,
    zipfile.ZIP_BZIP2,
    zipfile.ZIP_LZMA,
])
def test_compressed_payloads_span_volumes(comp):
    vs = 400
    expected = {}
    zss = SpannedZipStream(vs, compress_type=comp)
    for i in range(7):
        # mix incompressible (forces near-1:1 size) and compressible data
        payload = incompressible(600, i) if i % 2 == 0 else (
            bytes([i]) * 1500
        )
        name = "m{:02d}.dat".format(i)
        zss.add(payload, name)
        expected[name] = payload
    zss.mkdir("sub/")
    zss.add(b"", "empty.dat")

    vols, zss = generate(zss)
    assert len(vols) >= 5
    assert zss.is_complete()
    data, meta, members = read_spanned(vols, vs)
    assert {k: data[k] for k in expected} == expected
    assert data["sub/"] == b""
    assert data["empty.dat"] == b""
    # at least one compressed/stored body crosses a boundary
    crosses = 0
    for name, m in members.items():
        if m["is_dir"]:
            continue
        d, o = m["disk"], m["off"]
        c = _Cursor(vols, vs, d, o)
        c.skip_zero_padding()
        lf = struct.unpack(zipfile.structFileHeader, c.take(30))
        body_disk, body_off = c.disk, c.off
        c.take(lf[zipfile._FH_FILENAME_LENGTH])
        c.take(lf[zipfile._FH_EXTRA_FIELD_LENGTH])
        bs = c.disk * vs + c.off
        be = bs + m["csize"]
        if be // vs > bs // vs:
            crosses += 1
    assert crosses >= 1


def test_unsized_iterable_forces_zip64_local_header():
    vs = 350
    def gen():
        for i in range(15):
            yield bytes([i % 256]) * 250
    zss = SpannedZipStream(vs)
    zss.add(gen(), "streamed.bin")  # no size -> Zip64 LFH
    vols, zss = generate(zss)
    data, meta, members = read_spanned(vols, vs)
    assert data["streamed.bin"] == bytes(
        [i % 256 for i in range(15) for _ in range(250)]
    )
    # the local header uses a Zip64 extra field and sentinel sizes
    d, o = 0, 4
    lf = struct.unpack(zipfile.structFileHeader, vols[d][o:o + 30])
    assert lf[9] == 0xFFFFFFFF  # uncompressed size sentinel


################################
# Indivisible structure rules
################################

def test_structure_too_large_raises_volume_size_error():
    zss = SpannedZipStream(64)
    zss.add(b"hi", "n" * 200 + ".txt")  # LFH > 64 bytes
    with pytest.raises(VolumeSizeError) as ei:
        generate(zss)
    assert "larger than the configured volume size" in str(ei.value)


def test_end_record_too_large_raises_and_not_complete():
    # 22-byte EOCD can never fit in a 4-byte volume
    zss = SpannedZipStream(4)
    delivered = []
    with pytest.raises(VolumeSizeError):
        for v in zss.volumes():
            delivered.append(v.is_final)
    assert all(not is_final for is_final in delivered)
    assert zss.is_complete() is False
    assert zss.num_volumes() is None


def test_records_never_split_across_volumes():
    vs = 180
    zss = SpannedZipStream(vs)
    # pick a payload that ends 8 bytes before a boundary so the 16-byte data
    # descriptor cannot fit on that volume
    hdr = 4 + 30 + len("a.b")
    zss.add(b"A" * (vs - hdr - 8), "a.b")
    zss.add(b"B" * 40, "c.d")
    vols, zss = generate(zss)

    # every signature-bearing record must live wholly within one volume.
    # Check data descriptors and file headers never straddle a boundary.
    blob_offsets = []  # (global_offset, length, kind)
    global_off = 0
    for vi, vol in enumerate(vols):
        base = vi * vs
        # scan for LFH / DD / CDFH signatures
        for sig, length, kind in (
            (b"PK\x03\x04", 30, "LFH"),
            (b"PK\x07\x08", 16, "DD"),
            (b"PK\x01\x02", 46, "CDFH"),
        ):
            start = 0
            # skip marker (global offset 0) handled by kind check below
            while True:
                k = vol.find(sig, start)
                if k < 0:
                    break
                if not (vi == 0 and k == 0 and kind == "DD"):
                    g = base + k
                    assert (g + length - 1) // vs == g // vs, (
                        kind, g, "straddles a volume boundary"
                    )
                start = k + 1

    data, meta, _ = read_spanned(vols, vs)
    assert data["c.d"] == b"B" * 40


################################
# Zip64 handling
################################

def test_zip64_triggered_by_many_members():
    vs = 128 * 1024
    n = zipfile.ZIP_FILECOUNT_LIMIT + 1  # 65536
    zss = SpannedZipStream(vs)
    for i in range(n):
        zss.add(b"x", "f{:05d}".format(i))
    vols, zss = generate(zss)
    assert zss.is_complete()
    data, meta, _ = read_spanned(vols, vs)
    assert meta["is_zip64"]
    assert meta["total"] == n
    assert meta["num_disks"] == len(vols)
    assert len(data) == n
    assert all(data["f{:05d}".format(i)] == b"x" for i in range(n))
    # classic EOCD carries sentinel values
    last = vols[-1]
    ecd = struct.unpack(
        zipfile.structEndArchive,
        last[last.rfind(b"PK\x05\x06"):][:22],
    )
    assert ecd[zipfile._ECD_ENTRIES_TOTAL] == 0xFFFF
    assert ecd[zipfile._ECD_OFFSET] == 0xFFFFFFFF


def test_zip64_marker_when_member_offset_large_disk_count():
    # With a tiny volume, ensure classic disk-number sentinels are used and
    # the Zip64 locator records a correct total disk count.
    vs = 96
    zss = SpannedZipStream(vs)
    for i in range(30):
        zss.add(incompressible(300, i), "f{:02d}".format(i))
    vols, zss = generate(zss)
    assert len(vols) > 2
    data, meta, _ = read_spanned(vols, vs)
    assert meta["num_disks"] == len(vols)
    assert len(data) == 30


################################
# Completion / failure semantics
################################

def test_completion_state_not_available_until_done():
    zss = SpannedZipStream(200)
    assert zss.num_volumes() is None
    assert zss.is_complete() is False
    zss.add(incompressible(500, 1), "a")
    gen = zss.volumes()
    v1 = next(gen)
    # midway through generation the count is not final yet
    assert zss.num_volumes() is None
    # drain the rest
    for _ in gen:
        pass
    assert zss.is_complete()
    assert isinstance(zss.num_volumes(), int)
    assert v1.is_final is False


def test_failed_generation_does_not_mark_volumes_complete():
    vs = 256

    def bad_source():
        for _ in range(20):
            yield b"Y" * 200
        raise RuntimeError("boom")

    zss = SpannedZipStream(vs)
    zss.add(b"0" * 700, "good.bin")
    zss.add(bad_source(), "bad.bin", size=20 * 200)
    zss.add(b"z", "after.bin")

    delivered = []
    with pytest.raises(RuntimeError, match="boom"):
        for v in zss.volumes():
            delivered.append((b"".join(v), v.is_final))

    # nothing delivered is flagged as a final/complete volume
    assert delivered
    assert all(not is_final for _, is_final in delivered)
    assert zss.is_complete() is False
    assert zss.num_volumes() is None
    # the failed archive cannot be iterated again
    with pytest.raises(RuntimeError):
        list(zss.volumes())


def test_cannot_generate_twice():
    zss = SpannedZipStream(500)
    zss.add(b"x", "a")
    generate(zss)
    assert zss.is_complete()
    with pytest.raises(RuntimeError):
        generate(zss)


################################
# Single-volume archives remain ordinary zip-compatible
################################

def test_single_volume_readable_by_stdlib_zipfile():
    zss = SpannedZipStream(1_000_000)
    zss.add(b"hello", "a.txt")
    zss.mkdir("d/")
    zss.add(b"12345", "b.bin")
    blob = b"".join(b"".join(v) for v in zss.volumes())
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert sorted(zf.namelist()) == ["a.txt", "b.bin", "d/"]
        assert zf.read("a.txt") == b"hello"
        assert zf.read("b.bin") == b"12345"


def test_spanning_marker_on_first_volume():
    zss = SpannedZipStream(128)
    zss.add(incompressible(500, 9), "a.bin")
    vols, _ = generate(zss)
    assert vols[0][:4] == SPANNING_MARKER
    # the first member records an in-disk offset that skips the marker
    data, meta, members = read_spanned(vols, 128)
    first_disk_members = [
        m for m in members.values() if m["disk"] == 0
    ]
    assert first_disk_members
    assert min(m["off"] for m in first_disk_members) == 4


################################
# Extra edge cases
################################

def test_sized_stored_spanned():
    zss = SpannedZipStream(300, sized=True)  # stored by default
    assert len(zss) > 0  # single-stream size is known even though spanned
    zss.add(b"A" * 1000, "a.bin")
    zss.add(b"BB", "b.txt")
    vols, zss = generate(zss)
    data, meta, _ = read_spanned(vols, 300)
    assert data["a.bin"] == b"A" * 1000
    assert data["b.txt"] == b"BB"


def test_archive_comment_on_final_volume():
    zss = SpannedZipStream(300)
    zss.comment = b"a spanned comment"
    zss.add(incompressible(700, 4), "a.bin")
    vols, zss = generate(zss)
    # comment sits immediately after the EOCD on the final volume
    assert vols[-1][-len(b"a spanned comment"):] == b"a spanned comment"
    data, meta, _ = read_spanned(vols, 300)
    assert data["a.bin"] == incompressible(700, 4)


def test_delivered_volumes_are_immutable():
    zss = SpannedZipStream(150)
    zss.add(incompressible(600, 2), "a.bin")
    snapshots = []
    gen = zss.volumes()
    first = next(gen)
    first_bytes = b"".join(first)
    snapshots.append(first_bytes)
    # exhaust the rest; the already-delivered volume must not change
    rest = list(gen)
    assert b"".join(first) == first_bytes
    # iterating a delivered volume again yields identical bytes
    assert b"".join(first) == first_bytes
    assert first.is_final is False
    assert rest[-1].is_final is True


def test_add_path_file(tmp_path):
    p = tmp_path / "on_disk.bin"
    payload = incompressible(900, 11)
    p.write_bytes(payload)
    zss = SpannedZipStream(220)
    zss.add_path(str(p))
    vols, zss = generate(zss)
    data, meta, _ = read_spanned(vols, 220)
    assert data["on_disk.bin"] == payload
