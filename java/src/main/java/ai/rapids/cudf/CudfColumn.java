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

class CudfColumn {
  // This must be kept in sync with the native code
  public static final int UNKNOWN_NULL_COUNT = -1;
  public final long nativeHandle;

  public CudfColumn(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  /**
   * Create an empty column
   */
  public CudfColumn(DType dtype) {
    this.nativeHandle = makeEmptyCudfColumn(dtype.nativeId);
  }

  /**
   * Allocate a non-string column. Data must be copied into it after it is allocated, and if
   * null count is known it must be set after it is allocated.
   */
  public CudfColumn(DType dtype, int rows, MaskState maskState) {
    if (rows == 0) {
      this.nativeHandle = makeEmptyCudfColumn(dtype.nativeId);
    } else if (dtype.isTimestamp()) {
      this.nativeHandle = makeTimestampCudfColumn(dtype.nativeId, rows, maskState.nativeId);
    } else {
      this.nativeHandle = makeNumericCudfColumn(dtype.nativeId, rows, maskState.nativeId);
    }
  }

  /**
   * Create a string column from host side data.
   */
  public CudfColumn(long charData, long offsetData, long validData, int nullCount, int rows) {
    if (rows == 0) {
      this.nativeHandle = makeEmptyCudfColumn(DType.STRING.nativeId);
    } else {
      this.nativeHandle = makeStringCudfColumn(charData, offsetData, validData, nullCount, rows);
    }
  }

  public long getNativeValidPointer() {
    return getNativeValidPointer(nativeHandle);
  }

  public long getNativeRowCount() {
    return getNativeRowCount(nativeHandle);
  }

  public long getNativeNullCount() {
    return getNativeNullCount(nativeHandle);
  }

  public void setNullCount(int nullCount) throws CudfException {
    setNativeNullCount(nativeHandle, nullCount);
  }

  public long getNativeDataPointer() {
    return getNativeDataPointer(nativeHandle);
  }

  public long getNativeOffsetsPointer() {
    return getNativeOffsetsPointer(nativeHandle);
  }

  public void deleteCudfColumn() {
    deleteCudfColumn(nativeHandle);
  }

  public long transform(String udf, boolean isPtx) {
    return transform(this.nativeHandle, udf, isPtx);
  }

  //////////////////////////////////////////////////////////////////////////////
  // NATIVE METHODS
  /////////////////////////////////////////////////////////////////////////////
  private native void deleteCudfColumn(long cudfColumnHandle) throws CudfException;

  static native int getNativeTypeId(long cudfColumnHandle) throws CudfException;

  private native int getNativeRowCount(long cudfColumnHandle) throws CudfException;

  private native void setNativeNullCount(long cudfColumnHandle, int nullCount) throws CudfException;

  private native int getNativeNullCount(long cudfColumnHandle) throws CudfException;

  private native long getNativeDataPointer(long cudfColumnHandle) throws CudfException;

  private native long getNativeOffsetsPointer(long cudfColumnHandle) throws CudfException;

  private native long getNativeValidPointer(long cudfColumnHandle) throws CudfException;

  private native long makeNumericCudfColumn(int type, int rows, int maskState);

  private native long makeEmptyCudfColumn(int type);

  private native long makeTimestampCudfColumn(int type, int rows, int maskState);

  static native long getNativeValidPointerSize(int size);

  private native static long transform(long handle, String udf, boolean isPtx);

  private static native long makeStringCudfColumn(long charData, long offsetData, long validData, int nullCount, int size);

  public static native long[] getStringDataAndOffsets(long nativeHandle);

  @Override
  public String toString() {
    return String.valueOf(nativeHandle);
  }
}
