/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

import java.util.Optional;

final class CudfColumn implements AutoCloseable {
  // This must be kept in sync with the native code
  public static final long UNKNOWN_NULL_COUNT = -1;
  private long columnHandle;
  private long viewHandle = 0;
  private final BaseDeviceMemoryBuffer data;
  private final BaseDeviceMemoryBuffer valid;
  private final BaseDeviceMemoryBuffer offsets;

  /**
   * Make a column form an existing cudf::column *.
   */
  public CudfColumn(long columnHandle) {
    this.columnHandle = columnHandle;
    data = getNativeDataPointer();
    valid = getNativeValidPointer();
    offsets = getNativeOffsetsPointer();
  }

  /**
   * Make a column form an existing cudf::column_view * and a buffer that holds the data.
   */
  public CudfColumn(long viewHandle, DeviceMemoryBuffer contiguousBuffer) {
    assert viewHandle != 0;
    this.viewHandle = viewHandle;

    data = contiguousBuffer.sliceFrom(getNativeDataPointer());
    valid = contiguousBuffer.sliceFrom(getNativeValidPointer());
    offsets = contiguousBuffer.sliceFrom(getNativeOffsetsPointer());
  }

  /**
   * Create an empty column
   */
  public CudfColumn(DType dtype) {
    this.columnHandle = makeEmptyCudfColumn(dtype.nativeId);
    data = null;
    valid = null;
    offsets = null;
  }

  /**
   * Create a column from host side data.
   */
  public CudfColumn(DType type, int rows, Optional<Long> nullCount,
                    HostMemoryBuffer data, HostMemoryBuffer valid, HostMemoryBuffer offset) {
    assert (nullCount.isPresent() && nullCount.get() <= Integer.MAX_VALUE)
        || !nullCount.isPresent();
    int nc = nullCount.orElse(UNKNOWN_NULL_COUNT).intValue();
    long vd = valid == null ? 0 : valid.address;
    if (rows == 0) {
      this.columnHandle = makeEmptyCudfColumn(type.nativeId);
    } else if (type == DType.STRING) {
      long cd = data == null ? 0 : data.address;
      long od = offset == null ? 0 : offset.address;
      this.columnHandle = makeStringCudfColumnHostSide(cd, od, vd, nc, rows);
    } else {
      MaskState maskState;
      if (nc == 0 || (nc < 0 && vd == 0)) {
        maskState = MaskState.UNALLOCATED;
      } else {
        maskState = MaskState.UNINITIALIZED;
      }

      if (type.isTimestamp()) {
        this.columnHandle = makeTimestampCudfColumn(type.nativeId, rows, maskState.nativeId);
      } else {
        this.columnHandle = makeNumericCudfColumn(type.nativeId, rows, maskState.nativeId);
      }
      if (nc >= 0) {
        setNativeNullCount(nc);
      }

      if (maskState == MaskState.UNINITIALIZED) {
        DeviceMemoryBufferView validityView = getNativeValidPointer();
        validityView.copyFromHostBuffer(valid, 0, validityView.length);
      }

      DeviceMemoryBufferView dataView = getNativeDataPointer();
      // The host side data may be larger than the device side because we allocated more rows
      // Than needed
      dataView.copyFromHostBuffer(data, 0, dataView.length);

      // Offsets only applies to strings and was already copied in that path...
    }
    this.data = getNativeDataPointer();
    this.valid = getNativeValidPointer();
    this.offsets = getNativeOffsetsPointer();
  }

  /**
   * Create a cudf::column_view from device side data.
   */
  public CudfColumn(DType type, int rows, Optional<Long> nullCount,
                    DeviceMemoryBuffer data, DeviceMemoryBuffer valid, DeviceMemoryBuffer offsets) {
    assert (nullCount.isPresent() && nullCount.get() <= Integer.MAX_VALUE)
        || !nullCount.isPresent();
    int nc = nullCount.orElse(UNKNOWN_NULL_COUNT).intValue();
    this.data = data;
    this.valid = valid;
    this.offsets = offsets;

    if (rows == 0) {
      this.columnHandle = makeEmptyCudfColumn(type.nativeId);
    } else {
      long cd = data == null ? 0 : data.address;
      long cdSize = data == null ? 0 : data.length;
      long od = offsets == null ? 0 : offsets.address;
      long vd = valid == null ? 0 : valid.address;
      this.viewHandle = makeCudfColumnView(type.nativeId, cd, cdSize, od, vd, nc, rows);
    }
  }

  public long getViewHandle() {
    if (viewHandle == 0) {
      viewHandle = getNativeColumnView(columnHandle);
    }
    return viewHandle;
  }

  public long getNativeRowCount() {
    return getNativeRowCount(getViewHandle());
  }

  public long getNativeNullCount() {
    if (viewHandle != 0) {
      return getNativeNullCount(getViewHandle());
    }
    return getNativeNullCountColumn(columnHandle);
  }

  private void setNativeNullCount(int nullCount) throws CudfException {
    assert viewHandle == 0 : "Cannot set the null count if a view has already been created";
    assert columnHandle != 0;
    setNativeNullCountColumn(columnHandle, nullCount);
  }

  private DeviceMemoryBufferView getNativeValidPointer() {
    long[] values = getNativeValidPointer(getViewHandle());
    if (values[0] == 0) {
      return null;
    }
    return new DeviceMemoryBufferView(values[0], values[1]);
  }

  private DeviceMemoryBufferView getNativeDataPointer() {
    long[] values = getNativeDataPointer(getViewHandle());
    if (values[0] == 0) {
      return null;
    }
    return new DeviceMemoryBufferView(values[0], values[1]);
  }

  private DeviceMemoryBufferView getNativeOffsetsPointer() {
    long[] values = getNativeOffsetsPointer(getViewHandle());
    if (values[0] == 0) {
      return null;
    }
    return new DeviceMemoryBufferView(values[0], values[1]);
  }

  public DType getNativeType() {
    return DType.fromNative(getNativeTypeId(getViewHandle()));
  }

  public BaseDeviceMemoryBuffer getData() {
    return data;
  }

  public BaseDeviceMemoryBuffer getValid() {
    return valid;
  }

  public BaseDeviceMemoryBuffer getOffsets() {
    return offsets;
  }

  public long getDeviceMemorySize() {
    long size = valid != null ? valid.getLength() : 0;
    size += offsets != null ? offsets.getLength() : 0;
    size += data != null ? data.getLength() : 0;
    return size;
  }

  public void noWarnLeakExpected() {
    if (valid != null) {
      valid.noWarnLeakExpected();
    }
    if (data != null) {
      data.noWarnLeakExpected();
    }
    if(offsets != null) {
      offsets.noWarnLeakExpected();
    }
  }

  @Override
  public void close() {
    if (viewHandle != 0) {
      deleteColumnView(viewHandle);
      viewHandle = 0;
    }
    if (columnHandle != 0) {
      deleteCudfColumn(columnHandle);
      columnHandle = 0;
    }
    if (data != null) {
      data.close();
    }
    if (valid != null) {
      valid.close();
    }
    if (offsets != null) {
      offsets.close();
    }
  }

  //TODO remove this, or make it a part of ColumnVector
  public long transform(String udf, boolean isPtx) {
    return transform(getViewHandle(), udf, isPtx);
  }

  @Override
  public String toString() {
    return String.valueOf(columnHandle == 0 ? viewHandle : columnHandle);
  }

  //////////////////////////////////////////////////////////////////////////////
  // NATIVE METHODS
  /////////////////////////////////////////////////////////////////////////////
  private native void deleteCudfColumn(long cudfColumnHandle) throws CudfException;

  private static native int getNativeTypeId(long viewHandle) throws CudfException;

  private native int getNativeRowCount(long viewHandle) throws CudfException;

  private native void setNativeNullCountColumn(long cudfColumnHandle, int nullCount) throws CudfException;

  private native int getNativeNullCount(long viewHandle) throws CudfException;

  private native int getNativeNullCountColumn(long cudfColumnHandle) throws CudfException;

  private native void deleteColumnView(long viewHandle) throws CudfException;

  private native long getNativeColumnView(long cudfColumnHandle) throws CudfException;

  private native long[] getNativeDataPointer(long viewHandle) throws CudfException;

  private native long[] getNativeOffsetsPointer(long viewHandle) throws CudfException;

  private native long[] getNativeValidPointer(long viewHandle) throws CudfException;

  private native long makeNumericCudfColumn(int type, int rows, int maskState);

  private native long makeEmptyCudfColumn(int type);

  private native long makeTimestampCudfColumn(int type, int rows, int maskState);

  static native long getNativeValidPointerSize(int size);

  private native static long transform(long viewHandle, String udf, boolean isPtx);

  private static native long makeStringCudfColumnHostSide(long charData, long offsetData, long validData, int nullCount, int size);

  private static native long makeCudfColumnView(int type, long data, long dataSize, long offsets, long valid, int nullCount, int size);

}
