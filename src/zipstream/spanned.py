#!/usr/bin/env python

"""Real multi-volume (spanned/split) ZIP archive generation.

Unlike splitting a single-stream ZIP's bytes into fixed-size chunks, the
archives produced by this module follow the ZIP "multi-disk" rules from the
ZIP application note (APPNOTE.TXT):

- Every member records the disk (volume) its local header begins on, its
  byte offset within that disk, and (via the central directory) the location
  of the central directory and end-of-central-directory records.
- The local file header, data descriptor, central directory file header,
  Zip64 end-of-central-directory record, Zip64 locator and end-of-central-
  directory record are indivisible structures - they are never split across
  a volume boundary.
- A member's compressed (or stored) data stream may cross any number of
  volume boundaries.
- Each completed non-final volume is exactly `volume_size` bytes (unused
  space is zero-padded). The final volume may be shorter.
- Once a volume has been produced it is never modified. This is essential
  for streaming scenarios where volumes are delivered as they are generated.
- Zip64 extensions are emitted whenever a field would otherwise overflow
  (more than 65535 disks/members, offsets over 4 GiB, central directory over
  4 GiB, or per-member offsets requiring 64-bit fields).
"""

import collections
import struct

from zipfile import (
    ZIP_STORED,
    ZIP64_LIMIT, ZIP_FILECOUNT_LIMIT, ZIP64_VERSION,
    structEndArchive64, structEndArchive, structEndArchive64Locator,
    stringEndArchive64, stringEndArchive, stringEndArchive64Locator,
)

from zipstream.ng import ZipStream


__all__ = ["SpannedZipStream", "ZipVolume", "SpannedZipError", "VolumeSizeError"]

# Padding used to fill unused space at the end of a completed volume
_PADDING_BLOCK = b"\x00" * (1024 * 1024)

# The 4-byte marker placed at the start of the first disk of a multi-volume
# archive (APPNOTE.TXT 8.5.1 / X 0.008). It carries the data-descriptor
# signature to tell readers this is a spanned archive.
_SPANNING_MARKER = b"PK\x07\x08"

# Highest disk number that fits in the 16-bit classic disk fields (65534);
# 65535 (0xFFFF) is the sentinel that points readers at the Zip64 records.
_MAX_16BIT_DISK = 0xFFFE


class SpannedZipError(RuntimeError):
    """Base class for errors while generating a spanned archive"""


class VolumeSizeError(SpannedZipError):
    """An indivisible structure is larger than the configured volume size.

    Indivisible ZIP structures (local file headers, data descriptors, central
    directory headers and end-of-central-directory records) cannot be split
    across volumes, so they can never fit in an archive using this volume
    size.
    """


class ZipVolume:
    """A single volume (disk) of a spanned ZIP archive.

    Iterating over the object yields its data as bytes chunks. Volumes are
    immutable once sealed: already-delivered volumes are never rewritten.
    """

    def __init__(self, number, volume_size):
        self.number = number
        self.volume_size = volume_size
        self._chunks = collections.deque()
        self._length = 0
        self.is_final = False

    def __iter__(self):
        for chunk in tuple(self._chunks):
            yield bytes(chunk)

    def __len__(self):
        return self._length

    @property
    def free(self):
        """Number of bytes still available in this volume"""
        return self.volume_size - self._length

    def part_name(self, base_name=None):
        """Conventional spanned-archive name for this volume.

        Without a `base_name` this is just the suffix (".z01", ".z02", ...,
        ".zip" for the final volume). Given "out.zip" the full names are
        "out.z01", ..., "out.zip".
        """
        if self.is_final:
            suffix = ".zip"
        else:
            suffix = ".z{:02d}".format(self.number + 1)

        if base_name is None:
            return suffix

        root = base_name[:-4] if base_name.lower().endswith(".zip") else base_name
        return root + suffix

    def _append(self, data):
        self._chunks.append(data)
        self._length += len(data)

    def _seal(self, is_final):
        self.is_final = is_final
        self._chunks = tuple(self._chunks)


class _VolumePacker:
    """Packs streamed bytes into fixed-size volumes.

    Two kinds of writes are supported:

    - `stream(data)`: member payload bytes, which MAY cross volume boundaries
      (each write is split as needed).
    - `atomic(data)`: an indivisible ZIP structure, which never crosses a
      volume boundary. If it doesn't fit in the remaining space of the
      current volume, the volume is padded/sealed and a new one started.
      Raises VolumeSizeError if it can't fit in a whole empty volume.

    Every sealed volume is passed to the `on_seal` callback.
    """

    def __init__(self, volume_size, on_seal, initial_data=b""):
        if volume_size <= 0:
            raise ValueError("volume_size must be a positive integer")

        self.volume_size = volume_size
        self._on_seal = on_seal
        self._number = 0
        self._pos = 0
        self._current = ZipVolume(0, volume_size)
        self.finished = False
        # Data that must appear at the very start of the first volume (the
        # 4-byte "archive spans multiple volumes" marker). Written through the
        # normal accounting so disk_offset/abs_offset include it.
        if initial_data:
            self.atomic(initial_data, "spanning marker")

    @property
    def disk_number(self):
        """Zero-based number of the volume currently being written"""
        return self._number

    @property
    def disk_offset(self):
        """Byte offset within the volume currently being written"""
        return len(self._current)

    @property
    def abs_offset(self):
        """Absolute byte offset of the write position in the concatenation
        of all volumes (includes padding from sealed volumes)"""
        return self._pos

    def _advance(self, is_final):
        """Seal the current volume (zero-padding it unless it's the final
        one) and start the next volume."""
        volume = self._current
        free = volume.free
        if not is_final and free:
            full_blocks, remainder = divmod(free, len(_PADDING_BLOCK))
            for _ in range(full_blocks):
                volume._append(_PADDING_BLOCK)
            if remainder:
                volume._append(b"\x00" * remainder)

        # Account for the padding in the absolute position
        self._pos += free if not is_final else 0
        volume._seal(is_final)
        self._on_seal(volume)

        if not is_final:
            self._number += 1
            self._current = ZipVolume(self._number, self.volume_size)

    def atomic(self, data, name="structure"):
        """Write an indivisible structure (never split across volumes)."""
        length = len(data)
        if length > self.volume_size:
            raise VolumeSizeError(
                "The {} ({} bytes) is larger than the configured volume size "
                "of {} bytes and cannot be split across volumes".format(
                    name, length, self.volume_size
                )
            )

        # If there is no room at all (volume left exactly full by a previous
        # payload write) or the record doesn't fit, seal and start a volume.
        if self._current.free == 0 or length > self._current.free:
            self._advance(is_final=False)
        self._current._append(data)
        self._pos += length

    def reserve(self, length):
        """Ensure at least `length` bytes are free in the current volume,
        advancing to a new one if needed. Used to guarantee the final
        end-of-archive records share a volume. Does not itself write data."""
        if length > self.volume_size:
            raise VolumeSizeError(
                "The end-of-central-directory records ({} bytes) are larger "
                "than the configured volume size of {} bytes".format(
                    length, self.volume_size
                )
            )
        if self._current.free == 0 or length > self._current.free:
            self._advance(is_final=False)

    def fits_on_current_disk(self, length):
        """Return True if `length` bytes fit without advancing volumes"""
        return length <= self._current.free

    def stream(self, data):
        """Write member payload bytes (may cross volume boundaries)."""
        view = memoryview(data)
        pos = 0
        size = len(view)
        while pos < size:
            free = self._current.free
            if free == 0:
                self._advance(is_final=False)
                continue
            end = min(pos + free, size)
            self._current._append(view[pos:end])
            written = end - pos
            self._pos += written
            pos = end

    def finish(self):
        """Seal the final (possibly short) volume."""
        if self.finished:
            return
        self.finished = True
        self._advance(is_final=True)


class SpannedZipStream(ZipStream):
    """A ZipStream that produces a real multi-volume (spanned) ZIP archive.

    Volumes are generated independently, one at a time, by iterating
    `.volumes()`. Each yielded `ZipVolume` is immutable once received -
    already-delivered volumes are never rewritten. Every non-final volume is
    exactly `volume_size` bytes; the final volume holds the central directory
    and end-of-central-directory records.

    The total number of volumes is only known once generation completes (see
    `.num_volumes()`). `.is_complete()` reports whether a full, valid archive
    was produced. If generation fails, already-produced volumes all have
    `is_final == False` and are NOT a complete archive.
    """

    def __init__(self, volume_size, *, compress_type=ZIP_STORED, compress_level=None, sized=False):
        """Create a SpannedZipStream.

        volume_size:
            Exact size in bytes of every non-final volume. Must be large
            enough for the archive's biggest indivisible structure (local
            file headers, data descriptors, central directory headers and the
            end-of-central-directory records). VolumeSizeError is raised while
            generating if a structure cannot fit.

        The remaining parameters are identical to `ZipStream.__init__`.
        """
        if not isinstance(volume_size, int) or isinstance(volume_size, bool) or volume_size <= 0:
            raise ValueError("volume_size must be a positive integer")

        super().__init__(
            compress_type=compress_type,
            compress_level=compress_level,
            sized=sized,
        )
        self._volume_size = volume_size
        self._spanning = False
        self._num_volumes = None

    def num_volumes(self):
        """Number of volumes in the completed archive, or None if not done"""
        return self._num_volumes

    def is_complete(self):
        """True if the whole archive (including its final volume) was
        successfully generated"""
        return self._final and self._num_volumes is not None

    def volumes(self):
        """Generate the archive one volume at a time.

        Yields immutable `ZipVolume` objects in order. All queued members are
        encoded and the central directory / end-of-central-directory records
        are written to the final volume. Each volume (except the last) is
        delivered as soon as it fills and is never touched afterwards.

        Raises RuntimeError if the archive was already generated/generating.
        Raises VolumeSizeError if an indivisible structure exceeds
        `volume_size`.
        """
        if self._final:
            raise RuntimeError("SpannedZipStream has already been finalized")
        if self._spanning:
            raise RuntimeError("SpannedZipStream is already being generated")

        self._spanning = True
        pending = collections.deque()
        packer = _VolumePacker(
            self._volume_size, pending.append,
            initial_data=_SPANNING_MARKER,
        )

        try:
            yield from self._generate_spanned(packer, pending)
            packer.finish()
            total = packer.disk_number + 1

            # Mark the archive complete before handing over the final volume.
            self._num_volumes = total
            self._final = True
            while pending:
                yield pending.popleft()
        except GeneratorExit:
            raise
        except BaseException:
            # Generation failed: make sure nothing produced so far can be
            # mistaken for a complete archive. Seal the in-progress volume as
            # a non-final partial volume and hand back pending volumes, all
            # with is_final == False.
            self._num_volumes = None
            self._final = False
            try:
                if not packer.finished:
                    packer._advance(is_final=False)
            except Exception:
                pass
            while pending:
                yield pending.popleft()
            raise

    def _drain(self, pending):
        """Yield every volume sealed since the last drain"""
        while pending:
            yield pending.popleft()

    def _generate_spanned(self, packer, pending):
        """Encode all queued members and the archive footer into volumes"""
        while self._queue:
            kwargs = self._queue.popleft()
            yield from self._gen_spanned_file_entry(packer, pending, **kwargs)
        self._gen_spanned_footer(packer)

    def _gen_spanned_file_entry(self, packer, pending, *, path=None, iterable=None, data=None, size=None, arcname, compress_type=None, compress_level=None):
        """Encode one member; its payload may span multiple volumes"""
        zinfo, data_iter, force_zip64, size = self._prepare_file_entry(
            path=path,
            iterable=iterable,
            data=data,
            size=size,
            arcname=arcname,
            compress_type=compress_type,
            compress_level=compress_level,
        )

        # Record the member's start location per the ZIP multi-disk rules:
        # the disk it begins on and its byte offset on that disk.
        zinfo.volume = packer.disk_number
        zinfo.header_offset = packer.disk_offset

        records = zinfo._iter_file_data_records(
            data_iter, force_zip64=force_zip64
        )
        for tag, chunk in records:
            if tag == zinfo._RECORD_HEADER:
                packer.atomic(
                    chunk, "local file header for '{}'".format(zinfo.filename)
                )
            elif tag == zinfo._RECORD_DESCRIPTOR:
                packer.atomic(
                    chunk, "data descriptor for '{}'".format(zinfo.filename)
                )
            else:
                # Member payload - the only thing allowed to cross volumes
                packer.stream(chunk)
                # A volume may have filled/sealed while streaming; deliver it
                yield from self._drain(pending)

        self._check_file_entry_size(zinfo, size, arcname)
        self._filelist.append(zinfo)

    def _gen_spanned_footer(self, packer):
        """Write the central directory and end-of-central-directory records"""
        # No more members can be added once the footer starts
        self._final = True

        # Central directory file headers. Each is written atomically; the
        # central directory as a whole may span several volumes.
        cent_dir_start_disk = packer.disk_number
        cent_dir_start_offset = packer.disk_offset
        cent_dir_start_abs = packer.abs_offset

        for zinfo in self._filelist:
            cdfh = zinfo._central_directory_header_data(zinfo.volume)
            packer.atomic(
                cdfh, "central directory header for '{}'".format(zinfo.filename)
            )

        cent_dir_end_abs = packer.abs_offset
        cent_dir_size = cent_dir_end_abs - cent_dir_start_abs

        count = len(self._filelist)

        # Determine which disk the end records will land on *without* writing
        # anything yet, since the total disk count must be final before the
        # end records are encoded. Try the classic (small) footer first; if it
        # doesn't fit the current disk the footer starts on the next one.
        eocd_len = 22 + len(self._comment)

        def footer_disk(footer_len):
            if packer.fits_on_current_disk(footer_len):
                return packer.disk_number
            return packer.disk_number + 1

        classic_last_disk = footer_disk(eocd_len)

        # Decide whether Zip64 end records are required. Member start disks
        # only have a 16-bit field in the CDFH (there is no per-member Zip64
        # disk-number field), so a member on disk 0xFFFF+ is unreachable.
        needs_zip64 = (
            count >= ZIP_FILECOUNT_LIMIT
            or cent_dir_size > ZIP64_LIMIT
            or cent_dir_start_offset > ZIP64_LIMIT
            or classic_last_disk > _MAX_16BIT_DISK
            or any(
                z.volume > _MAX_16BIT_DISK or z.header_offset > ZIP64_LIMIT
                for z in self._filelist
            )
        )

        if needs_zip64:
            last_disk = footer_disk(56 + 20 + eocd_len)
        else:
            last_disk = classic_last_disk

        # There is no Zip64 extension for a member's start-disk field, so a
        # member beginning on disk 0xFFFF or later can't be addressed.
        too_far = next(
            (z for z in self._filelist if z.volume > _MAX_16BIT_DISK), None
        )
        if too_far is not None:
            raise SpannedZipError(
                "Member '{}' begins on disk {} but the central directory's "
                "16-bit disk-number field can only address disks 0-{}".format(
                    too_far.filename, too_far.volume, _MAX_16BIT_DISK
                )
            )
        if last_disk > 0xFFFFFFFF:
            raise SpannedZipError(
                "This archive requires {} disks, which exceeds the Zip64 "
                "maximum of 2^32".format(last_disk + 1)
            )

        # Advance volumes if needed so the end records share one final disk
        # and the disk count encoded in them is now final.
        reserve_len = (56 + 20 + eocd_len) if needs_zip64 else eocd_len
        packer.reserve(reserve_len)

        if needs_zip64:
            # Both the Zip64 EOCD and classic EOCD "central directory offset"
            # fields are relative to the start of the disk named in the
            # preceding "disk with start of central directory" field.
            zip64_end = struct.pack(
                structEndArchive64,
                stringEndArchive64,
                44,  # size of this record after this point
                ZIP64_VERSION,  # version made by
                ZIP64_VERSION,  # version needed to extract
                last_disk,  # disk number this record is on
                cent_dir_start_disk,  # disk with the start of the central dir
                count,  # total entries in the central dir on this disk
                count,  # total entries in the central dir
                cent_dir_size,
                cent_dir_start_offset,
            )
            zip64_end_disk = packer.disk_number
            zip64_end_offset = packer.disk_offset
            packer.atomic(zip64_end, "Zip64 end of central directory record")

            locator = struct.pack(
                structEndArchive64Locator,
                stringEndArchive64Locator,
                zip64_end_disk,  # disk where the Zip64 EOCD starts
                zip64_end_offset,  # offset of the Zip64 EOCD on that disk
                last_disk + 1,  # total number of disks
            )
            packer.atomic(locator, "Zip64 end of central directory locator")

            # Classic fields are limited to 16/32 bits; sentinel values point
            # readers at the Zip64 records.
            classic_count = 0xFFFF
            classic_cd_size = 0xFFFFFFFF
            classic_cd_offset = 0xFFFFFFFF
            classic_disk = 0xFFFF
            classic_cd_disk = 0xFFFF
        else:
            classic_count = count
            classic_cd_size = cent_dir_size
            classic_cd_offset = cent_dir_start_offset
            classic_disk = last_disk
            classic_cd_disk = cent_dir_start_disk

        eocd = struct.pack(
            structEndArchive,
            stringEndArchive,
            classic_disk,  # disk number this record is on
            classic_cd_disk,  # disk with the start of the central directory
            classic_count,  # entries on this disk / total entries
            classic_count,  # total entries
            classic_cd_size,
            classic_cd_offset,
            len(self._comment)
        )
        packer.atomic(eocd + self._comment, "end of central directory record")
