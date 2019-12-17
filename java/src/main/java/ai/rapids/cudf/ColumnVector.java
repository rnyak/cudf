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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * A Column Vector. This class represents the immutable vector of data.  This class holds
 * references to off heap memory and is reference counted to know when to release it.  Call
 * close to decrement the reference count when you are done with the column, and call inRefCount
 * to increment the reference count.
 */
public final class ColumnVector implements AutoCloseable, BinaryOperable {
  static final String STANDARD_CUDF_PORTING_MSG = "This is a legacy ColumnVector " +
      "operation that hasn't ported over yet";
  /**
   * The size in bytes of an offset entry
   */
  static final int OFFSET_SIZE = DType.INT32.sizeInBytes;
  private static final Logger log = LoggerFactory.getLogger(ColumnVector.class);
  private static final AtomicLong idGen = new AtomicLong(0);

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private final DType type;
  private final OffHeapState offHeap = new OffHeapState();
  private long rows;
  private Optional<Long> nullCount = Optional.empty();
  private int refCount;
  private final long internalId = idGen.incrementAndGet();

  /**
   * Wrap an existing on device cudf::column with the corresponding ColumnVector.
   */
  ColumnVector(long nativePointer) {
    assert nativePointer != 0;
    MemoryCleaner.register(this, offHeap);
    offHeap.internalId = internalId;
    offHeap.cudfColumnHandle = new CudfColumn(nativePointer);
    this.type = offHeap.cudfColumnHandle.getNativeType();
    offHeap.setHostData(null);
    this.rows = offHeap.cudfColumnHandle.getNativeRowCount();
    this.refCount = 0;
    incRefCountInternal(true);
    MemoryListener.deviceAllocation(getDeviceMemorySize(), internalId);
  }

  /**
   * Create a new column vector with data populated on the host.
   */
  ColumnVector(DType type, long rows, Optional<Long> nullCount,
               HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer) {
    this(type, rows, nullCount, hostDataBuffer, hostValidityBuffer, null);
  }

  /**
   * Create a new column vector with data populated on the host.
   * @param type               the type of the vector
   * @param rows               the number of rows in the vector.
   * @param nullCount          the number of nulls in the vector.
   * @param hostDataBuffer     The host side data for the vector. In the case of STRING
   *                           this is the string data stored as bytes.
   * @param hostValidityBuffer arrow like validity buffer 1 bit per row, with padding for
   *                           64-bit alignment.
   * @param offsetBuffer       only valid for STRING this is the offsets into
   *                           the hostDataBuffer indicating the start and end of a string
   *                           entry. It should be (rows + 1) ints.
   */
  ColumnVector(DType type, long rows, Optional<Long> nullCount,
               HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer,
               HostMemoryBuffer offsetBuffer) {
    if (nullCount.isPresent() && nullCount.get() > 0 && hostValidityBuffer == null) {
      throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
    }
    if (type == DType.STRING) {
      assert offsetBuffer != null : "offsets must be provided for STRING";
    } else {
      assert offsetBuffer == null : "offsets are only supported for STRING";
    }
    MemoryCleaner.register(this, offHeap);
    offHeap.internalId = internalId;
    offHeap.setHostData(new BufferEncapsulator(hostDataBuffer, hostValidityBuffer, offsetBuffer));
    this.rows = rows;
    this.nullCount = nullCount;
    this.type = type;
    refCount = 0;
    incRefCountInternal(true);
  }

  /**
   * Create a new column vector based off of data already on the device.
   * @param type the type of the vector
   * @param rows the number of rows in this vector.
   * @param nullCount the number of nulls in the dataset.
   * @param dataBuffer the data stored on the device.  The column vector takes ownership of the
   *                   buffer.  Do not use the buffer after calling this.
   * @param validityBuffer an optional validity buffer. Must be provided if nullCount != 0. The
   *                      column vector takes ownership of the buffer. Do not use the buffer
   *                      after calling this.
   * @param offsetBuffer a host buffer required for strings and string categories. The column
   *                    vector takes ownership of the buffer. Do not use the buffer after calling
   *                    this.
   */
  ColumnVector(DType type, long rows,
               Optional<Long> nullCount, DeviceMemoryBuffer dataBuffer, DeviceMemoryBuffer validityBuffer,
               DeviceMemoryBuffer offsetBuffer) {
    if (type == DType.STRING) {
      assert offsetBuffer != null : "offsets must be provided for STRING";
    } else {
      assert offsetBuffer == null : "offsets are only supported for STRING";
    }

    MemoryCleaner.register(this, offHeap);
    offHeap.internalId = internalId;
    offHeap.setHostData(null);
    offHeap.cudfColumnHandle = new CudfColumn(type, (int) rows, nullCount, dataBuffer, validityBuffer, offsetBuffer);
    this.rows = rows;
    this.nullCount = nullCount;
    this.type = type;
    this.refCount = 0;
    incRefCountInternal(true);
  }

  /**
   * This is a really ugly API, but it is possible that the lifecycle of a column of
   * data may not have a clear lifecycle thanks to java and GC. This API informs the leak
   * tracking code that this is expected for this column, and big scary warnings should
   * not be printed when this happens.
   */
  public final void noWarnLeakExpected() {
    offHeap.noWarnLeakExpected();
    if (offHeap.getHostData() != null) {
      offHeap.getHostData().noWarnLeakExpected();
    }
    if (offHeap.cudfColumnHandle != null) {
      offHeap.cudfColumnHandle.noWarnLeakExpected();
    }
  }

  /**
   * Close this Vector and free memory allocated for HostMemoryBuffer and DeviceMemoryBuffer
   */
  @Override
  public final void close() {
    refCount--;
    offHeap.delRef();
    if (refCount == 0) {
      offHeap.clean(false);
    } else if (refCount < 0) {
      log.error("Close called too many times on {}", this);
      offHeap.logRefCountDebug("double free " + this);
      throw new IllegalStateException("Close called too many times");
    }
  }

  @Override
  public String toString() {
    return "CudfColumn{" +
        "rows=" + rows +
        ", type=" + type +
        ", hostData=" + offHeap.getHostData() +
        ", nullCount=" + nullCount +
        ", cudfColumn=" + offHeap.cudfColumnHandle +
        '}';
  }

  /////////////////////////////////////////////////////////////////////////////
  // METADATA ACCESS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Increment the reference count for this column.  You need to call close on this
   * to decrement the reference count again.
   */
  public ColumnVector incRefCount() {
    return incRefCountInternal(false);
  }

  private ColumnVector incRefCountInternal(boolean isFirstTime) {
    offHeap.addRef();
    if (refCount <= 0 && !isFirstTime) {
      offHeap.logRefCountDebug("INC AFTER CLOSE " + this);
      throw new IllegalStateException("Column is already closed");
    }
    refCount++;
    return this;
  }

  /**
   * Returns the number of rows in this vector.
   */
  public long getRowCount() {
    return rows;
  }

  /**
   * Returns the amount of device memory used.
   */
  public long getDeviceMemorySize() {
    return offHeap != null ? offHeap.getDeviceMemorySize() : 0;
  }

  /**
   * Returns the amount of host memory used to store column/validity data (not metadata).
   */
  public long getHostMemorySize() {
    return offHeap != null ? offHeap.getHostMemorySize() : 0;
  }

  /**
   * Retrieve the number of characters in each string. Null strings will have value of null.
   *
   * @return ColumnVector holding length of string at index 'i' in the original vector
   */
  public ColumnVector getLengths() {
    assert DType.STRING == type : "length only available for String type";
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT32), "getLengths")) {
      return new ColumnVector(lengths(offHeap.cudfColumnHandle.getViewHandle()));
    }
  }

  /**
   * Returns the type of this vector.
   */
  @Override
  public DType getType() {
    return type;
  }

  /**
   * Returns the number of nulls in the data. Note that this might end up
   * being a very expensive operation because if the null count is not
   * known it will be calculated.
   */
  public long getNullCount() {
    if (!nullCount.isPresent()) {
      if (offHeap.cudfColumnHandle != null) {
        nullCount = Optional.of(offHeap.cudfColumnHandle.getNativeNullCount());
      } else {
        throw new IllegalStateException("Calculating an unknown null count on the host is not currently supported");
      }
    }
    return nullCount.get();
  }

  /**
   * Returns this column's current refcount
   */
  int getRefCount() {
    return refCount;
  }

  /**
   * Retrieve the number of bytes for each string. Null strings will have value of null.
   *
   * @return ColumnVector, where each element at i = byte count of string at index 'i' in the original vector
   */
  public ColumnVector getByteCount() {
    assert type == DType.STRING : "type has to be a String";
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT32), "byteCount")) {
      return new ColumnVector(byteCount(offHeap.cudfColumnHandle.getViewHandle()));
    }
  }

  /**
   * Returns if the vector has a validity vector allocated or not.
   */
  boolean hasValidityVector() {
    boolean ret;
    if (offHeap.hostData != null) {
      ret = (offHeap.hostData.valid != null);
    } else {
      ret = (offHeap.cudfColumnHandle.getValid() != null);
    }
    return ret;
  }

  /**
   * Returns if the vector has nulls.  Note that this might end up
   *    * being a very expensive operation because if the null count is not
   *    * known it will be calculated.
   */
  public boolean hasNulls() {
    return getNullCount() > 0;
  }

  /////////////////////////////////////////////////////////////////////////////
  // DATA MOVEMENT
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Return true if the data is on the device, or false if it is not. Note that
   * if there are no rows there is no data to be on the device, but this will
   * still return true.
   */
  public boolean hasDeviceData() {
    return offHeap.cudfColumnHandle != null;
  }

  /**
   * Return true if the data is on the host, or false if it is not. Note that
   * if there are no rows there is no data to be on the host, but this will
   * still return true.
   */
  public boolean hasHostData() {
    return offHeap.getHostData() != null || rows == 0;
  }

  private void checkHasDeviceData() {
    if (!hasDeviceData()) {
      if (refCount <= 0) {
        throw new IllegalStateException("Vector was already closed.");
      }
      throw new IllegalStateException("Vector not on device");
    }
  }

  private void checkHasHostData() {
    if (!hasHostData()) {
      if (refCount <= 0) {
        throw new IllegalStateException("Vector was already closed.");
      }
      throw new IllegalStateException("Vector not on host");
    }
  }

  /**
   * Drop any data stored on the host, but move it to the device first if necessary.
   */
  public final void dropHostData() {
    ensureOnDevice();
    if (offHeap.hostData != null) {
      offHeap.hostData.close();
      offHeap.hostData = null;
      // Host data tracking happens on a per buffer basis.
    }
  }

  /**
   * Drop any data stored on the device, but move it to the host first if necessary.
   */
  public final void dropDeviceData() {
    ensureOnHost();
    if (offHeap.cudfColumnHandle != null) {
      long amount = getDeviceMemorySize();
      offHeap.cudfColumnHandle.close();
      offHeap.cudfColumnHandle = null;
      MemoryListener.deviceDeallocation(amount, internalId);
    }
  }

  /**
   * Be sure the data is on the device.
   */
  public final void ensureOnDevice() {
    if (offHeap.cudfColumnHandle == null) {
      if (rows == 0) {
        offHeap.cudfColumnHandle = new CudfColumn(type);
        return;
      }
      checkHasHostData();

      assert type != DType.STRING || offHeap.getHostData().offsets != null;

      try (DevicePrediction prediction =
               new DevicePrediction(getHostMemorySize(), "ensureOnDevice");
           NvtxRange toDev = new NvtxRange("ensureOnDevice", NvtxColor.BLUE)) {

        offHeap.cudfColumnHandle = new CudfColumn(type, (int) rows, nullCount, offHeap.hostData.data,
            offHeap.hostData.valid, offHeap.hostData.offsets);

        if (offHeap.isLeakExpected()) {
          offHeap.cudfColumnHandle.noWarnLeakExpected();
        }
      }
    }
  }

  /**
   * Be sure the data is on the host.
   */
  public final void ensureOnHost() {
    if (offHeap.getHostData() == null && rows != 0) {
      checkHasDeviceData();

      try (HostPrediction prediction =
               new HostPrediction(getDeviceMemorySize(), "ensureOnHost");
           NvtxRange toHost = new NvtxRange("ensureOnHost", NvtxColor.BLUE)) {
        HostMemoryBuffer hostDataBuffer = null;
        HostMemoryBuffer hostValidityBuffer = null;
        HostMemoryBuffer hostOffsetsBuffer = null;
        BaseDeviceMemoryBuffer valid = offHeap.cudfColumnHandle.getValid();
        BaseDeviceMemoryBuffer offsets = offHeap.cudfColumnHandle.getOffsets();
        BaseDeviceMemoryBuffer data = offHeap.cudfColumnHandle.getData();
        boolean needsCleanup = true;
        try {
          if (valid != null) {
            hostValidityBuffer = HostMemoryBuffer.allocate(valid.getLength());
          }
          if (offsets != null) {
            hostOffsetsBuffer = HostMemoryBuffer.allocate(offsets.length);
          }
          // If a strings column is all null values there is no data buffer allocated
          if (data != null) {
            hostDataBuffer = HostMemoryBuffer.allocate(data.length);
          }

          offHeap.setHostData(new BufferEncapsulator(hostDataBuffer, hostValidityBuffer,
              hostOffsetsBuffer));
          needsCleanup = false;
        } finally {
          if (needsCleanup) {
            if (hostOffsetsBuffer != null) {
              hostOffsetsBuffer.close();
            }
            if (hostDataBuffer != null) {
              hostDataBuffer.close();
            }
            if (hostValidityBuffer != null) {
              hostValidityBuffer.close();
            }
          }
        }
        if (offHeap.hostData.data != null) {
          offHeap.hostData.data.copyFromDeviceBuffer(data);
        }
        if (offHeap.hostData.valid != null) {
          offHeap.hostData.valid.copyFromDeviceBuffer(valid);
        }
        if (offHeap.hostData.offsets != null) {
          offHeap.hostData.offsets.copyFromDeviceBuffer(offsets);
        }
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // DATA ACCESS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Check if the value at index is null or not.
   */
  public boolean isNull(long index) {
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    if (hasValidityVector()) {
      if (nullCount.isPresent() && !hasNulls()) {
        return false;
      }
      checkHasHostData();
      return BitVectorHelper.isNull(offHeap.getHostData().valid, index);
    }
    return false;
  }

  /**
   * For testing only.  Allows null checks to go past the number of rows, but not past the end
   * of the buffer.  NOTE: If the validity vector was allocated by cudf itself it is not
   * guaranteed to have the same padding, but for all practical purposes it does.  This is
   * just to verify that the buffer was allocated and initialized properly.
   */
  boolean isNullExtendedRange(long index) {
    long maxNullRow = BitVectorHelper.getValidityAllocationSizeInBytes(rows) * 8;
    assert (index >= 0 && index < maxNullRow) : "TEST: index is out of range 0 <= " + index + " <" +
        " " + maxNullRow;
    if (hasValidityVector()) {
      if (nullCount.isPresent() && !hasNulls()) {
        return false;
      }
      checkHasHostData();
      return BitVectorHelper.isNull(offHeap.getHostData().valid, index);
    }
    return false;
  }

  public enum BufferType {
    VALIDITY,
    OFFSET,
    DATA
  }

  /**
   * Get access to the raw host buffer for this column.  This is intended to be used with a lot
   * of caution.  The lifetime of the buffer is tied to the lifetime of the column (Do not close
   * the buffer, as the column will take care of it).  Do not modify the contents of the buffer or
   * it might negatively impact what happens on the column.  The data must be on the host for this
   * to work.
   * @param type the type of buffer to get access to.
   * @return the underlying buffer or null if no buffer is associated with it for this column.
   * Please note that if the column is empty there may be no buffers at all associated with the
   * column.
   */
  public HostMemoryBuffer getHostBufferFor(BufferType type) {
    checkHasHostData();
    HostMemoryBuffer srcBuffer = null;
    BufferEncapsulator<HostMemoryBuffer> host = offHeap.getHostData();
    switch(type) {
      case VALIDITY:
        if (host != null) {
          srcBuffer = host.valid;
        }
        break;
      case OFFSET:
        if (host != null) {
          srcBuffer = host.offsets;
        }
        break;
      case DATA:
        if (host != null) {
          srcBuffer = host.data;
        }
        break;
      default:
        throw new IllegalArgumentException(type + " is not a supported buffer type.");
    }
    return srcBuffer;
  }

  /**
   * Get access to the raw device buffer for this column.  This is intended to be used with a lot
   * of caution.  The lifetime of the buffer is tied to the lifetime of the column (Do not close
   * the buffer, as the column will take care of it).  Do not modify the contents of the buffer or
   * it might negatively impact what happens on the column.  The data must be on the device for
   * this to work. Strings and string categories do not currently work because their underlying
   * device layout is currently hidden.
   * @param type the type of buffer to get access to.
   * @return the underlying buffer or null if no buffer is associated with it for this column.
   * Please note that if the column is empty there may be no buffers at all associated with the
   * column.
   */
  public DeviceMemoryBuffer getDeviceBufferFor(BufferType type) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    assert this.type != TypeId.STRING && this.type != TypeId.STRING_CATEGORY;
    checkHasDeviceData();
    DeviceMemoryBuffer srcBuffer = null;
    BufferEncapsulator<DeviceMemoryBuffer> dev = offHeap.getDeviceData();
    switch(type) {
      case VALIDITY:
        if (dev != null) {
          srcBuffer = dev.valid;
        }
        break;
      case DATA:
        if (dev != null) {
          srcBuffer = dev.data;
        }
        break;
      default:
        throw new IllegalArgumentException(type + " is not a supported buffer type.");
    }
    return srcBuffer;
*/
  }

  void copyHostBufferBytes(byte[] dst, int dstOffset, BufferType src, long srcOffset,
                           int length) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    assert dstOffset >= 0;
    assert srcOffset >= 0;
    assert length >= 0;
    assert dstOffset + length <= dst.length;

    HostMemoryBuffer srcBuffer = getHostBufferFor(src);

    assert srcOffset + length <= srcBuffer.length : "would copy off end of buffer "
        + srcOffset + " + " + length + " > " + srcBuffer.length;
    UnsafeMemoryAccessor.getBytes(dst, dstOffset,
        srcBuffer.getAddress() + srcOffset, length);
*/
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is not null, and FALSE for any null entry (as per the validity mask)
   *
   * @return - Boolean vector
   */
  public ColumnVector isNotNull() {
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.BOOL8), "isNotNull")) {
      return new ColumnVector(isNotNullNative(offHeap.cudfColumnHandle.getViewHandle()));
    }
  }

  /**
   * Returns a vector with all values "oldValues[i]" replaced with "newValues[i]".
   * Warning:
   *    Currently this function doesn't work for Strings or StringCategories.
   *    NaNs can't be replaced in the original vector but regular values can be replaced with NaNs
   *    Nulls can't be replaced in the original vector but regular values can be replaced with Nulls
   *    Mixing of types isn't allowed, the resulting vector will be the same type as the original.
   *      e.g. You can't replace an integer vector with values from a long vector
   *
   * Usage:
   *    this = {1, 4, 5, 1, 5}
   *    oldValues = {1, 5, 7}
   *    newValues = {2, 6, 9}
   *
   *    result = this.findAndReplaceAll(oldValues, newValues);
   *    result = {2, 4, 6, 2, 6}  (1 and 5 replaced with 2 and 6 but 7 wasn't found so no change)
   *
   * @param oldValues - A vector containing values that should be replaced
   * @param newValues - A vector containing new values
   * @return - A new vector containing the old values replaced with new values
   */
  public ColumnVector findAndReplaceAll(ColumnVector oldValues, ColumnVector newValues) {
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "findAndReplace")) {
      return new ColumnVector(findAndReplaceAll(oldValues.getNativeView(), newValues.getNativeView(), this.getNativeView()));
    }
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * FALSE for any entry that is not null, and TRUE for any null entry (as per the validity mask)
   *
   * @return - Boolean vector
   */
  public ColumnVector isNull() {
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.BOOL8),"isNull")) {
      return new ColumnVector(isNullNative(offHeap.cudfColumnHandle.getViewHandle()));
    }
  }

  /**
   * Returns a ColumnVector with any null values replaced with a scalar.
   * The types of the input ColumnVector and Scalar must match, else an error is thrown.
   *
   * @param scalar - Scalar value to use as replacement
   * @return - ColumnVector with nulls replaced by scalar
   */
  public ColumnVector replaceNulls(Scalar scalar) {
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "replaceNulls")) {
      return new ColumnVector(replaceNulls(getNativeView(), scalar.getScalarHandle()));
    }
  }

  /**
   * Generic type independent asserts when getting a value from a single index.
   * @param index where to get the data from.
   */
  private void assertsForGet(long index) {
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    assert offHeap.getHostData() != null : "data is not on the host";
    assert !isNull(index) : " value at " + index + " is null";
  }

  /**
   * Get the value at index.
   */
  public byte getByte(long index) {
    assert type == DType.INT8 || type == DType.BOOL8;
    assertsForGet(index);
    return offHeap.getHostData().data.getByte(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final short getShort(long index) {
    assert type == DType.INT16;
    assertsForGet(index);
    return offHeap.getHostData().data.getShort(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final int getInt(long index) {
    assert type == DType.INT32 || type == DType.TIMESTAMP_DAYS;
    assertsForGet(index);
    return offHeap.getHostData().data.getInt(index * type.sizeInBytes);
  }

  /**
   * Get the starting byte offset for the string at index
   */
  long getStartStringOffset(long index) {
    assert type == DType.STRING;
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    assert offHeap.getHostData() != null : "data is not on the host";
    return offHeap.getHostData().offsets.getInt(index * 4);
  }

  /**
   * Get the ending byte offset for the string at index.
   */
  long getEndStringOffset(long index) {
    assert type == DType.STRING;
    assert (index >= 0 && index < rows) : "index is out of range 0 <= " + index + " < " + rows;
    assert offHeap.getHostData() != null : "data is not on the host";
    // The offsets has one more entry than there are rows.
    return offHeap.getHostData().offsets.getInt((index + 1) * 4);
  }

  /**
   * Get the value at index.
   */
  public final long getLong(long index) {
    // Timestamps with time values are stored as longs
    assert type == DType.INT64 || type.hasTimeResolution();
    assertsForGet(index);
    return offHeap.getHostData().data.getLong(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final float getFloat(long index) {
    assert type == DType.FLOAT32;
    assertsForGet(index);
    return offHeap.getHostData().data.getFloat(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.
   */
  public final double getDouble(long index) {
    assert type == DType.FLOAT64;
    assertsForGet(index);
    return offHeap.getHostData().data.getDouble(index * type.sizeInBytes);
  }

  /**
   * Get the boolean value at index
   */
  public final boolean getBoolean(long index) {
    assert type == DType.BOOL8;
    assertsForGet(index);
    return offHeap.getHostData().data.getBoolean(index * type.sizeInBytes);
  }

  /**
   * Get the value at index.  This API is slow as it has to translate the
   * string representation.  Please use it with caution.
   */
  public String getJavaString(long index) {
    assert type == DType.STRING;
    assertsForGet(index);
    int start = offHeap.getHostData().offsets.getInt(index * OFFSET_SIZE);
    int size = offHeap.getHostData().offsets.getInt((index + 1) * OFFSET_SIZE) - start;
    byte[] rawData = new byte[size];
    if (size > 0) {
      offHeap.getHostData().data.getBytes(rawData, 0, start, size);
    }
    return new String(rawData, StandardCharsets.UTF_8);
  }

  /////////////////////////////////////////////////////////////////////////////
  // DATE/TIME
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Get year from DATE32, DATE64, or TIMESTAMP
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector year() {
    assert type.isTimestamp();
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT16), "year")) {
      return new ColumnVector(year(offHeap.cudfColumnHandle.getViewHandle()));
    }
  }

  /**
   * Get month from DATE32, DATE64, or TIMESTAMP
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector month() {
    assert type.isTimestamp();
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT16), "month")) {
      return new ColumnVector(month(offHeap.cudfColumnHandle.getViewHandle()));
    }
  }

  /**
   * Get day from DATE32, DATE64, or TIMESTAMP
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector day() {
    assert type.isTimestamp();
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT16), "day")) {
      return new ColumnVector(day(offHeap.cudfColumnHandle.getViewHandle()));
    }
  }

  /**
   * Get hour from DATE64 or TIMESTAMP
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector hour() {
    assert type.hasTimeResolution();
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT16), "hour")) {
      return new ColumnVector(hour(offHeap.cudfColumnHandle.getViewHandle()));
    }
  }

  /**
   * Get minute from DATE64 or TIMESTAMP
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector minute() {
    assert type.hasTimeResolution();
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT16), "minute")) {
      return new ColumnVector(minute(offHeap.cudfColumnHandle.getViewHandle()));
    }
  }

  /**
   * Get second from DATE64 or TIMESTAMP
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector second() {
    assert type.hasTimeResolution();
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(DType.INT16), "second")) {
      return new ColumnVector(second(offHeap.cudfColumnHandle.getViewHandle()));
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // ARITHMETIC
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Multiple different unary operations. The output is the same type as input.
   * @param op      the operation to perform
   * @return the result
   */
  public ColumnVector unaryOp(UnaryOp op) {
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "unaryOp")) {
      return new ColumnVector(unaryOperation(offHeap.cudfColumnHandle.getViewHandle(), op.nativeId));
    }
  }

  /**
   * Calculate the sin, output is the same type as input.
   */
  public ColumnVector sin() {
    return unaryOp(UnaryOp.SIN);
  }

  /**
   * Calculate the cos, output is the same type as input.
   */
  public ColumnVector cos() {
    return unaryOp(UnaryOp.COS);
  }

  /**
   * Calculate the tan, output is the same type as input.
   */
  public ColumnVector tan() {
    return unaryOp(UnaryOp.TAN);
  }

  /**
   * Calculate the arcsin, output is the same type as input.
   */
  public ColumnVector arcsin() {
    return unaryOp(UnaryOp.ARCSIN);
  }

  /**
   * Calculate the arccos, output is the same type as input.
   */
  public ColumnVector arccos() {
    return unaryOp(UnaryOp.ARCCOS);
  }

  /**
   * Calculate the arctan, output is the same type as input.
   */
  public ColumnVector arctan() {
    return unaryOp(UnaryOp.ARCTAN);
  }

  /**
   * Calculate the exp, output is the same type as input.
   */
  public ColumnVector exp() {
    return unaryOp(UnaryOp.EXP);
  }

  /**
   * Calculate the log, output is the same type as input.
   */
  public ColumnVector log() {
    return unaryOp(UnaryOp.LOG);
  }

  /**
   * Calculate the sqrt, output is the same type as input.
   */
  public ColumnVector sqrt() {
    return unaryOp(UnaryOp.SQRT);
  }

  /**
   * Calculate the ceil, output is the same type as input.
   */
  public ColumnVector ceil() {
    return unaryOp(UnaryOp.CEIL);
  }

  /**
   * Calculate the floor, output is the same type as input.
   */
  public ColumnVector floor() {
    return unaryOp(UnaryOp.FLOOR);
  }

  /**
   * Calculate the abs, output is the same type as input.
   */
  public ColumnVector abs() {
    return unaryOp(UnaryOp.ABS);
  }

  /**
   * invert the bits, output is the same type as input.
   */
  public ColumnVector bitInvert() {
    return unaryOp(UnaryOp.BIT_INVERT);
  }

  /**
   * Multiple different binary operations.
   * @param op      the operation to perform
   * @param rhs     the rhs of the operation
   * @param outType the type of output you want.
   * @return the result
   */
  @Override
  public ColumnVector binaryOp(BinaryOp op, BinaryOperable rhs, DType outType) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(outType), "binaryOp")) {
      if (rhs instanceof ColumnVector) {
        ColumnVector cvRhs = (ColumnVector) rhs;
        assert rows == cvRhs.getRowCount();
        return new ColumnVector(Cudf.gdfBinaryOp(this, cvRhs, op, outType));
      } else if (rhs instanceof Scalar) {
        Scalar sRhs = (Scalar) rhs;
        return new ColumnVector(Cudf.gdfBinaryOp(this, sRhs, op, outType));
      } else {
        throw new IllegalArgumentException(rhs.getClass() + " is not supported as a binary op" +
            " with ColumnVector");
      }
    }
*/
  }

  /**
   * Slices a column (including null values) into a set of columns
   * according to a set of indices. The caller owns the ColumnVectors and is responsible
   * closing them
   *
   * The "slice" function divides part of the input column into multiple intervals
   * of rows using the indices values and it stores the intervals into the output
   * columns. Regarding the interval of indices, a pair of values are taken from
   * the indices array in a consecutive manner. The pair of indices are left-closed
   * and right-open.
   *
   * The pairs of indices in the array are required to comply with the following
   * conditions:
   * a, b belongs to Range[0, input column size]
   * a <= b, where the position of a is less or equal to the position of b.
   *
   * Exceptional cases for the indices array are:
   * When the values in the pair are equal, the function returns an empty column.
   * When the values in the pair are 'strictly decreasing', the outcome is
   * undefined.
   * When any of the values in the pair don't belong to the range[0, input column
   * size), the outcome is undefined.
   * When the indices array is empty, an empty vector of columns is returned.
   *
   * The caller owns the output ColumnVectors and is responsible for closing them.
   *
   * @param indices
   * @return A new ColumnVector array with slices from the original ColumnVector
   */
  public ColumnVector[] slice(int... indices) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    try (ColumnVector cv = ColumnVector.fromInts(indices)) {
      return slice(cv);
    }
*/
  }

  /**
   * Slices a column (including null values) into a set of columns
   * according to a set of indices. The caller owns the ColumnVectors and is responsible
   * closing them
   *
   * The "slice" function divides part of the input column into multiple intervals
   * of rows using the indices values and it stores the intervals into the output
   * columns. Regarding the interval of indices, a pair of values are taken from
   * the indices array in a consecutive manner. The pair of indices are left-closed
   * and right-open.
   *
   * The pairs of indices in the array are required to comply with the following
   * conditions:
   * a, b belongs to Range[0, input column size]
   * a <= b, where the position of a is less or equal to the position of b.
   *
   * Exceptional cases for the indices array are:
   * When the values in the pair are equal, the function returns an empty column.
   * When the values in the pair are 'strictly decreasing', the outcome is
   * undefined.
   * When any of the values in the pair don't belong to the range[0, input column
   * size), the outcome is undefined.
   * When the indices array is empty, an empty vector of columns is returned.
   *
   * The caller owns the output ColumnVectors and is responsible for closing them.
   *
   * @param indices
   * @return A new ColumnVector array with slices from the original ColumnVector
   */
  public ColumnVector[] slice(ColumnVector indices) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "slice")) {
      long[] nativeHandles = cudfSlice(this.getNativeCudfColumnAddress(), indices.getNativeCudfColumnAddress());
      ColumnVector[] columnVectors = new ColumnVector[nativeHandles.length];
      IntStream.range(0, nativeHandles.length).forEach(i -> columnVectors[i] = new ColumnVector(nativeHandles[i]));
      return columnVectors;
    }
*/
  }

  /**
   * Splits a column (including null values) into a set of columns
   * according to a set of indices. The caller owns the ColumnVectors and is responsible
   * closing them.
   *
   * The "split" function divides the input column into multiple intervals
   * of rows using the splits indices values and it stores the intervals into the
   * output columns. Regarding the interval of indices, a pair of values are taken
   * from the indices array in a consecutive manner. The pair of indices are
   * left-closed and right-open.
   *
   * The indices array ('splits') is require to be a monotonic non-decreasing set.
   * The indices in the array are required to comply with the following conditions:
   * a, b belongs to Range[0, input column size]
   * a <= b, where the position of a is less or equal to the position of b.
   *
   * The split function will take a pair of indices from the indices array
   * ('splits') in a consecutive manner. For the first pair, the function will
   * take the value 0 and the first element of the indices array. For the last pair,
   * the function will take the last element of the indices array and the size of
   * the input column.
   *
   * Exceptional cases for the indices array are:
   * When the values in the pair are equal, the function return an empty column.
   * When the values in the pair are 'strictly decreasing', the outcome is
   * undefined.
   * When any of the values in the pair don't belong to the range[0, input column
   * size), the outcome is undefined.
   * When the indices array is empty, an empty vector of columns is returned.
   *
   * The input columns may have different sizes. The number of
   * columns must be equal to the number of indices in the array plus one.
   *
   * Example:
   * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
   * splits: {2, 5, 9}
   * output:  {{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}}
   *
   * Note that this is very similar to the output from a PartitionedTable.
   *
   * @param indices the indexes to split with
   * @return A new ColumnVector array with slices from the original ColumnVector
   */
  public ColumnVector[] split(int... indices) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    try (ColumnVector cv = ColumnVector.fromInts(indices)) {
      return split(cv);
    }
*/
  }

  /**
   * Splits a column (including null values) into a set of columns
   * according to a set of indices. The caller owns the ColumnVectors and is responsible
   * closing them.
   *
   * The "split" function divides the input column into multiple intervals
   * of rows using the splits indices values and it stores the intervals into the
   * output columns. Regarding the interval of indices, a pair of values are taken
   * from the indices array in a consecutive manner. The pair of indices are
   * left-closed and right-open.
   *
   * The indices array ('splits') is require to be a monotonic non-decreasing set.
   * The indices in the array are required to comply with the following conditions:
   * a, b belongs to Range[0, input column size]
   * a <= b, where the position of a is less or equal to the position of b.
   *
   * The split function will take a pair of indices from the indices array
   * ('splits') in a consecutive manner. For the first pair, the function will
   * take the value 0 and the first element of the indices array. For the last pair,
   * the function will take the last element of the indices array and the size of
   * the input column.
   *
   * Exceptional cases for the indices array are:
   * When the values in the pair are equal, the function return an empty column.
   * When the values in the pair are 'strictly decreasing', the outcome is
   * undefined.
   * When any of the values in the pair don't belong to the range[0, input column
   * size), the outcome is undefined.
   * When the indices array is empty, an empty vector of columns is returned.
   *
   * The input columns may have different sizes. The number of
   * columns must be equal to the number of indices in the array plus one.
   *
   * Example:
   * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
   * splits: {2, 5, 9}
   * output:  {{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}}
   *
   * Note that this is very similar to the output from a PartitionedTable.
   *
   * @param indices the indexes to split with
   * @return A new ColumnVector array with slices from the original ColumnVector
   */
  public ColumnVector[] split(ColumnVector indices) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "split")) {
      long[] nativeHandles = split(this.getNativeCudfColumnAddress(), indices.getNativeCudfColumnAddress());
      ColumnVector[] columnVectors = new ColumnVector[nativeHandles.length];
      for (int i = 0; i < nativeHandles.length; i++) {
        columnVectors[i] = new ColumnVector(nativeHandles[i]);
      }
      return columnVectors;
    }
*/
  }

  /**
   * Computes the sum of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar sum() {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return sum(type);
  }

  /**
   * Computes the sum of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar sum(DType outType) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return reduce(ReductionOp.SUM, outType);
  }

  /**
   * Returns the minimum of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar min() {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return min(type);
  }

  /**
   * Returns the minimum of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar min(DType outType) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return reduce(ReductionOp.MIN, outType);
  }

  /**
   * Returns the maximum of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar max() {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return max(type);
  }

  /**
   * Returns the maximum of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar max(DType outType) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return reduce(ReductionOp.MAX, outType);
  }

  /**
   * Returns the product of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar product() {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return product(type);
  }

  /**
   * Returns the product of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar product(DType outType) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return reduce(ReductionOp.PRODUCT, outType);
  }

  /**
   * Returns the sum of squares of all values in the column, returning a
   * scalar of the same type as this column.
   */
  public Scalar sumOfSquares() {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return sumOfSquares(type);
  }

  /**
   * Returns the sum of squares of all values in the column, returning a
   * scalar of the specified type.
   */
  public Scalar sumOfSquares(DType outType) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return reduce(ReductionOp.SUMOFSQUARES, outType);
  }

  /**
   * Returns the sample standard deviation of all values in the column,
   * returning a FLOAT64 scalar unless the column type is FLOAT32 then
   * a FLOAT32 scalaris returned. Null's are not counted as an element
   * of the column when calculating the standard deviation.
   */
  public Scalar standardDeviation() {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    if(type != TypeId.FLOAT32)
      standardDeviation(TypeId.FLOAT64);
    return standardDeviation(type);
*/
  }

  /**
   * Returns the sample standard deviation of all values in the column,
   * returning a scalar of the specified type. Null's are not counted as
   * an element of the column when calculating the standard deviation.
   */
  public Scalar standardDeviation(DType outType) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return reduce(ReductionOp.STD, outType);
  }

  /**
   * Computes the reduction of the values in all rows of a column.
   * Overflows in reductions are not detected. Specifying a higher precision
   * output type may prevent overflow. Only the MIN and MAX ops are
   * The null values are skipped for the operation.
   * @param op The reduction operation to perform
   * @return The scalar result of the reduction operation. If the column is
   * empty or the reduction operation fails then the
   * {@link Scalar#isValid()} method of the result will return false.
   */
  public Scalar reduce(ReductionOp op) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return reduce(op, type);
  }

  /**
   * Computes the reduction of the values in all rows of a column.
   * Overflows in reductions are not detected. Specifying a higher precision
   * output type may prevent overflow. Only the MIN and MAX ops are
   * supported for reduction of non-arithmetic types (DATE32, TIMESTAMP...)
   * The null values are skipped for the operation.
   * @param op      The reduction operation to perform
   * @param outType The type of scalar value to return
   * @return The scalar result of the reduction operation. If the column is
   * empty or the reduction operation fails then the
   * {@link Scalar#isValid()} method of the result will return false.
   */
  public Scalar reduce(ReductionOp op, DType outType) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return Cudf.reduce(this, op, outType);
  }

  /////////////////////////////////////////////////////////////////////////////
  // LOGICAL
  /////////////////////////////////////////////////////////////////////////////
 
  /**
   * Returns a vector of the logical `not` of each value in the input 
   * column (this)
   */
  public ColumnVector not() {
    return unaryOp(UnaryOp.NOT);
  }

  /////////////////////////////////////////////////////////////////////////////
  // TYPE CAST
  /////////////////////////////////////////////////////////////////////////////

  static long predictSizeFor(long baseSize, long rows, boolean includeValidity) {
    long total = baseSize * rows;
    if (includeValidity) {
      total += BitVectorHelper.getValidityAllocationSizeInBytes(rows);
    }
    return total;
  }

  long predictSizeFor(DType type) {
    return predictSizeFor(type.sizeInBytes, rows, hasValidityVector());
  }

  private long predictSizeForRowMult(long baseSize, double rowMult) {
    long rowGuess = (long)(rows * rowMult);
    return predictSizeFor(baseSize, rowGuess, hasValidityVector());
  }

  /**
   * Generic method to cast ColumnVector
   * When casting from a Date, Timestamp, or Boolean to a numerical type the underlying numerical
   * representation of the data will be used for the cast.
   * @param type type of the resulting ColumnVector
   * @return A new vector allocated on the GPU
   */
  public ColumnVector castTo(DType type) {
    if (this.type == type) {
      // Optimization
      return incRefCount();
    }
    try (DevicePrediction prediction = new DevicePrediction(predictSizeFor(type), "cast")) {
      return new ColumnVector(castTo(offHeap.cudfColumnHandle.getViewHandle(), type.nativeId));
    }
  }

  /**
   * Cast to Byte - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to byte
   * When casting from a Date, Timestamp, or Boolean to a byte type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asBytes() {
    return castTo(DType.INT8);
  }

  /**
   * Cast to Short - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to short
   * When casting from a Date, Timestamp, or Boolean to a short type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asShorts() {
    return castTo(DType.INT16);
  }

  /**
   * Cast to Int - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to int
   * When casting from a Date, Timestamp, or Boolean to a int type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asInts() {
    return castTo(DType.INT32);
  }

  /**
   * Cast to Long - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to long
   * When casting from a Date, Timestamp, or Boolean to a long type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asLongs() {
    return castTo(DType.INT64);
  }

  /**
   * Cast to Float - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to float
   * When casting from a Date, Timestamp, or Boolean to a float type the underlying numerical
   * representatio of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asFloats() {
    return castTo(DType.FLOAT32);
  }

  /**
   * Cast to Double - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to double
   * When casting from a Date, Timestamp, or Boolean to a double type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asDoubles() {
    return castTo(DType.FLOAT64);
  }

  /**
   * Cast to TIMESTAMP_DAYS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_DAYS
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampDays() {
    if (type == DType.STRING) {
      return asTimestamp(DType.TIMESTAMP_DAYS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_DAYS);
  }

  /**
   * Cast to TIMESTAMP_DAYS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_DAYS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampDays(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_DAYS, format);
  }

  /**
   * Cast to TIMESTAMP_SECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_SECONDS
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampSeconds() {
    if (type == DType.STRING) {
      return asTimestamp(DType.TIMESTAMP_SECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_SECONDS);
  }

  /**
   * Cast to TIMESTAMP_SECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_SECONDS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampSeconds(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_SECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_MICROSECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_MICROSECONDS
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMicroseconds() {
    if (type == DType.STRING) {
      return asTimestamp(DType.TIMESTAMP_MICROSECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_MICROSECONDS);
  }

  /**
   * Cast to TIMESTAMP_MICROSECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_MICROSECONDS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMicroseconds(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_MICROSECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_MILLISECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_MILLISECONDS.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMilliseconds() {
    if (type == DType.STRING) {
      return asTimestamp(DType.TIMESTAMP_MILLISECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_MILLISECONDS);
  }

  /**
   * Cast to TIMESTAMP_MILLISECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_MILLISECONDS.
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMilliseconds(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_MILLISECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_NANOSECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_NANOSECONDS.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampNanoseconds() {
    if (type == DType.STRING) {
      return asTimestamp(DType.TIMESTAMP_NANOSECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_NANOSECONDS);
  }

  /**
   * Cast to TIMESTAMP_NANOSECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_NANOSECONDS.
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampNanoseconds(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_NANOSECONDS, format);
  }

  /**
   * Native method to parse and convert a NVString column vector to unix timestamp. A unix
   * timestamp is a long value representing how many units since 1970-01-01 00:00:00.000 in either
   * positive or negative direction. This mirrors the functionality spark sql's to_unix_timestamp.
   * Strings that fail to parse will default to 0. Supported time units are second, millisecond,
   * microsecond, and nanosecond. Larger time units for column vectors are not supported yet in cudf.
   * @param unit integer native ID of the time unit to parse the timestamp into.
   * @param format strptime format specifier string of the timestamp. Used to parse and convert
   *               the timestamp with. Supports %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z format specifiers.
   *               See https://github.com/rapidsai/custrings/blob/branch-0.10/docs/source/datetime.md
   *               for full parsing format specification and documentation.
   * @return native handle of the resulting cudf column, used to construct the Java column vector
   *         by the timestampToLong method.
   */
  private static native long stringTimestampToTimestamp(long viewHandle, int unit, String format);

  /**
   * Wrap static native string timestamp to long conversion method. Retrieves the column vector's
   * cudf native handle and uses it to invoke the native function that calls NVStrings'
   * timestamp2long method. Strings that fail to parse will default to 0, corresponding
   * to 1970-01-01 00:00:00.000.
   * @param timestampType timestamp DType that includes the time unit to parse the timestamp into.
   * @param format strptime format specifier string of the timestamp. Used to parse and convert
   *               the timestamp with. Supports %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z format specifiers.
   *               See https://github.com/rapidsai/custrings/blob/branch-0.10/docs/source/datetime.md
   *               for full parsing format specification and documentation.
   * @return A new ColumnVector containing the long representations of the timestamps in the
   *         original column vector.
   */
  public ColumnVector asTimestamp(DType timestampType, String format) {
    assert type == DType.STRING : "A column of type string " +
                                  "is required for .to_timestamp() operation";
    assert format != null : "Format string may not be NULL";
    assert timestampType.isTimestamp() : "unsupported conversion to non-timestamp DType";

    // Prediction could be better, but probably okay for now
    try (DevicePrediction prediction = new DevicePrediction(predictSizeForRowMult(format.length(), 2), "asTimestamp")) {
      return new ColumnVector(stringTimestampToTimestamp(getNativeView(),
          timestampType.nativeId, format));
    }
  }

  /**
   * Cast to Strings.
   * @return A new vector allocated on the GPU.
   */
  public ColumnVector asStrings() {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return castTo(TypeId.STRING, TimeUnit.NONE);
  }

  /////////////////////////////////////////////////////////////////////////////
  // INTERNAL/NATIVE ACCESS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * USE WITH CAUTION: This method exposes the address of the native cudf::column_view.  This allows
   * writing custom kernels or other cuda operations on the data.  DO NOT close this column
   * vector until you are completely done using the native column_view.  DO NOT modify the column in
   * any way.  This should be treated as a read only data structure. This API is unstable as
   * the underlying C/C++ API is still not stabilized.  If the underlying data structure
   * is renamed this API may be replaced.  The underlying data structure can change from release
   * to release (it is not stable yet) so be sure that your native code is complied against the
   * exact same version of libcudf as this is released for.
   */
  final long getNativeView() {
    if (offHeap.cudfColumnHandle == null) {
      //Data must have been dropped from device recreate it on device
      assert rows <= Integer.MAX_VALUE;
      ensureOnDevice();
    }
    return offHeap.cudfColumnHandle.getViewHandle();
  }

//  private static native long allocateCudfColumn() throws CudfException;

  private native static long byteCount(long viewHandle) throws CudfException;

  private static native long castTo(long nativeHandle, int type);

//  /**
//   * Set a CuDF column given data and validity bitmask pointers, size, and datatype, and
//   * count of null (non-valid) elements
//   * @param cudfColumnHandle native handle of cudf::column.
//   * @param dataPtr          Pointer to data.
//   * @param valid            Pointer to validity bitmask for the data.
//   * @param size             Number of rows in the column.
//   * @param TypeId            Data type of the column.
//   * @param null_count       The number of non-valid elements in the validity bitmask.
//   * @param timeUnit         {@link TimeUnit}
//   */
//  private static native void cudfColumnViewAugmented(long cudfColumnHandle, long dataPtr,
//                                                     long valid,
//                                                     int size, int TypeId, int null_count,
//                                                     int timeUnit) throws CudfException;

//  private native long[] cudfSlice(long nativeHandle, long indices) throws CudfException;

//  private native long[] split(long nativeHandle, long indices) throws CudfException;

  private native long findAndReplaceAll(long valuesHandle, long replaceHandle, long myself) throws CudfException;

  /**
   * Translate the host side string representation of strings into the device side representation
   * and populate the cudfColumn with it.
   * @param cudfColumnHandle native handle of cudf::column.
   * @param dataPtr          Pointer to string data either on the host or the device.
   * @param dataPtrOnHost    true if dataPtr is on the host. false if it is on the device.
   * @param hostOffsetsPtr   Pointer to offsets data on the host.
   * @param resetOffsetsToZero true if the offsets should be reset to start at 0.
   * @param deviceValidPtr   Pointer to validity bitmask on the device.
   * @param deviceOutputDataPtr Pointer to where the int category data will be stored for
   *                            STRING_CATEGORY. Should be 0 for STRING.
   * @param numRows          Number of rows in the column.
   * @param TypeId            Data type of the column. In this case must be STRING or
   *                         STRING_CATEGORY
   * @param nullCount        The number of non-valid elements in the validity bitmask.
   */
//  private static native void cudfColumnViewStrings(long cudfColumnHandle, long dataPtr,
//                                                   boolean dataPtrOnHost, long hostOffsetsPtr,
//                                                   boolean resetOffsetsToZero, long deviceValidPtr,
//                                                   long deviceOutputDataPtr,
//                                                   int numRows, int TypeId, int nullCount);

  /**
   * Native method to switch all characters in a column of strings to lowercase characters.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @return native handle of the resulting cudf column, used to construct the Java column
   *         by the lower method.
   */
  private static native long lowerStrings(long cudfViewHandle);

  /**
   * Native method to switch all characters in a column of strings to uppercase characters.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @return native handle of the resulting cudf column, used to construct the Java column
   *         by the upper method.
   */
  private static native long upperStrings(long cudfViewHandle);

  /**
   * Wrap static native string capitilization methods, retrieves the column vectors cudf native
   * handle and uses it to invoke the native function that calls NVStrings' upper method. Does not
   * support String categories yet.
   * @return A new ColumnVector containing the upper case versions of the strings in the original
   *         column vector.
   */
  public ColumnVector upper() {
    assert type == DType.STRING : "A column of type string is required for .upper() operation";
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "upper")) {
      return new ColumnVector(upperStrings(getNativeView()));
    }
  }

  /**
   * Wrap static native string capitilization methods, retrieves the column vectors cudf native
   * handle and uses it to invoke the native function that calls NVStrings' lower method. Does not
   * support String categories yet.
   * @return A new ColumnVector containing the lower case versions of the strings in the original
   *         column vector.
   */
  public ColumnVector lower() {
    assert type == DType.STRING : "A column of type string is required for .lower() operation";
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "lower")) {
      return new ColumnVector(lowerStrings(getNativeView()));
    }
  }

//  private native Scalar exactQuantile(long cudfColumnHandle, int quantileMethod, double quantile) throws CudfException;

//  private native Scalar approxQuantile(long cudfColumnHandle, double quantile) throws CudfException;

//  private static native long rollingWindow(long cudfColumnHandle, int window, int min_periods,
//                                           int forward_window, int agg_type, long window_col,
//                                           long min_periods_col, long forward_window_col);

  private static native long lengths(long viewHandle) throws CudfException;

//  private static native int getTimeUnitInternal(long cudfColumnHandle) throws CudfException;

//  private static native int getNullCount(long cudfColumnHandle) throws CudfException;

//  private static native int getDeviceMemoryStringSize(long cudfColumnHandle) throws CudfException;

  private static native long concatenate(long[] columnHandles) throws CudfException;

  private static native long fromScalar(long scalarHandle, int rowCount) throws CudfException;

  private static native long replaceNulls(long columnHandle, long scalarHandle) throws CudfException;

  private static native long isNullNative(long viewHandle);

  private static native long isNotNullNative(long viewHandle);

  private static native long unaryOperation(long viewHandle, int op);
  
  private static native long year(long viewHandle) throws CudfException;

  private static native long month(long viewHandle) throws CudfException;

  private static native long day(long viewHandle) throws CudfException;

  private static native long hour(long viewHandle) throws CudfException;

  private static native long minute(long viewHandle) throws CudfException;

  private static native long second(long viewHandle) throws CudfException;

  /////////////////////////////////////////////////////////////////////////////
  // HELPER CLASSES
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Encapsulator class to hold the two buffers as a cohesive object
   */
  private static final class BufferEncapsulator<T extends MemoryBuffer> implements AutoCloseable {
    public final T data;
    public final T valid;
    public final T offsets;

    BufferEncapsulator(T data, T valid, T offsets) {
      this.data = data;
      this.valid = valid;
      this.offsets = offsets;
    }

    @Override
    public String toString() {
      T type = data == null ? valid : data;
      type = type == null ? offsets : type;
      String t = "UNKNOWN";
      if (type != null) {
        t = type.getClass().getSimpleName();
      }
      return "BufferEncapsulator{type= " + t
          + ", data= " + data
          + ", valid= " + valid
          + ", offsets= " + offsets + "}";
    }

    @Override
    public void close() {
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

    /**
     * This is a really ugly API, but it is possible that the lifecycle of a column of
     * data may not have a clear lifecycle thanks to java and GC. This API informs the leak
     * tracking code that this is expected for this column, and big scary warnings should
     * not be printed when this happens.
     */
    public void noWarnLeakExpected() {
      if (data != null) {
        data.noWarnLeakExpected();
      }
      if (valid != null) {
        valid.noWarnLeakExpected();
      }
      if (offsets != null) {
        offsets.noWarnLeakExpected();
      }
    }
  }

  /**
   * Holds the off heap state of the column vector so we can clean it up, even if it is leaked.
   */
  protected static final class OffHeapState extends MemoryCleaner.Cleaner {
    private long internalId;
    private BufferEncapsulator<HostMemoryBuffer> hostData;
    private CudfColumn cudfColumnHandle = null;

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      if (getHostData() != null) {
        getHostData().close();
        setHostData(null);
        neededCleanup = true;
      }
      if (cudfColumnHandle != null) {
        cudfColumnHandle.close();
        cudfColumnHandle = null;
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("YOU LEAKED A COLUMN VECTOR!!!!");
        logRefCountDebug("Leaked vector");
      }
      return neededCleanup;
    }

    public BufferEncapsulator<HostMemoryBuffer> getHostData() {
      return hostData;
    }

    public void setHostData(BufferEncapsulator<HostMemoryBuffer> hostData) {
      if (isLeakExpected() && hostData != null) {
        hostData.noWarnLeakExpected();
      }
      this.hostData = hostData;
    }

    /**
     * This returns total memory allocated in device for the ColumnVector.
     * @return number of device bytes allocated for this column
     */
    public long getDeviceMemorySize() {
      long size = 0;
      if (cudfColumnHandle != null) {
        size = cudfColumnHandle.getDeviceMemorySize();
      }
      return size;
    }

    /**
     * This returns total memory allocated on the host for the ColumnVector.
     */
    public long getHostMemorySize() {
      long total = 0;
      if (hostData != null) {
        if (hostData.valid != null) {
          total += hostData.valid.length;
        }
        if (hostData.data != null) {
          total += hostData.data.length;
        }
        if (hostData.offsets != null) {
          total += hostData.offsets.length;
        }
      }
      return total;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // BUILDER
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Create a new Builder to hold the specified number of rows.  Be sure to close the builder when
   * done with it. Please try to use {@see #build(int, Consumer)} instead to avoid needing to
   * close the builder.
   * @param type the type of vector to build.
   * @param rows the number of rows this builder can hold
   * @return the builder to use.
   */
  public static Builder builder(DType type, int rows) {
    return new Builder(type, rows, 0);
  }

  /**
   * Create a new Builder to hold the specified number of rows and with enough space to hold the
   * given amount of string data. Be sure to close the builder when done with it. Please try to
   * use {@see #build(int, int, Consumer)} instead to avoid needing to close the builder.
   * @param type the type of vector to build.
   * @param rows the number of rows this builder can hold
   * @param stringBufferSize the size of the string buffer to allocate.
   * @return the builder to use.
   */
  public static Builder builder(DType type, int rows, long stringBufferSize) {
    assert type == DType.STRING;
    return new ColumnVector.Builder(type, rows, stringBufferSize);
  }

  /**
   * Create a new vector.
   * @param type       the type of vector to build.
   * @param rows       maximum number of rows that the vector can hold.
   * @param init       what will initialize the vector.
   * @return the created vector.
   */
  public static ColumnVector build(DType type, int rows, Consumer<Builder> init) {
    try (ColumnVector.Builder builder = builder(type, rows)) {
      init.accept(builder);
      return builder.build();
    }
  }

  public static ColumnVector build(DType type, int rows, long stringBufferSize, Consumer<Builder> init) {
    try (ColumnVector.Builder builder = builder(type, rows, stringBufferSize)) {
      init.accept(builder);
      return builder.build();
    }
  }

  /**
   * Create a new vector without sending data to the device.
   * @param type       the type of vector to build.
   * @param rows       maximum number of rows that the vector can hold.
   * @param init       what will initialize the vector.
   * @return the created vector.
   */
  public static ColumnVector buildOnHost(DType type, int rows,
                                         Consumer<Builder> init) {
    try (Builder builder = builder(type, rows)) {
      init.accept(builder);
      return builder.buildOnHost();
    }
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector boolFromBytes(byte... values) {
    return build(DType.BOOL8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromBytes(byte... values) {
    return build(DType.INT8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromShorts(short... values) {
    return build(DType.INT16, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromInts(int... values) {
    return build(DType.INT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromLongs(long... values) {
    return build(DType.INT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromFloats(float... values) {
    return build(DType.FLOAT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromDoubles(double... values) {
    return build(DType.FLOAT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector daysFromInts(int... values) {
    return build(DType.TIMESTAMP_DAYS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_SECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampMilliSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_MILLISECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampMicroSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_MICROSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampNanoSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_NANOSECONDS, values.length, (b) -> b.appendArray(values));
  }

  private static ColumnVector fromStrings(DType type, String... values) {
    assert type == DType.STRING;
    int rows = values.length;
    long nullCount = 0;
    // How many bytes do we need to hold the data.  Sorry this is really expensive
    long bufferSize = 0;
    for (String s: values) {
      if (s == null) {
        nullCount++;
      } else {
        bufferSize += s.getBytes(StandardCharsets.UTF_8).length;
      }
    }
    if (nullCount > 0) {
      return build(type, rows, bufferSize, (b) -> b.appendBoxed(values));
    }
    return build(type, rows, bufferSize, (b) -> {
      for (String s: values) {
        b.append(s);
      }
    });
  }

  /**
   * Create a new string vector from the given values.  This API
   * supports inline nulls. This is really intended to be used only for testing as
   * it is slow and memory intensive to translate between java strings and UTF8 strings.
   */
  public static ColumnVector fromStrings(String... values) {
    return fromStrings(DType.STRING, values);
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedBooleans(Boolean... values) {
    return build(DType.BOOL8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedBytes(Byte... values) {
    return build(DType.INT8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedShorts(Short... values) {
    return build(DType.INT16, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedInts(Integer... values) {
    return build(DType.INT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedLongs(Long... values) {
    return build(DType.INT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedFloats(Float... values) {
    return build(DType.FLOAT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedDoubles(Double... values) {
    return build(DType.FLOAT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampDaysFromBoxedInts(Integer... values) {
    return build(DType.TIMESTAMP_DAYS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_SECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampMilliSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_MILLISECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampMicroSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_MICROSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampNanoSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_NANOSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector of length rows, where each row is filled with the Scalar's
   * value
   * @param scalar - Scalar to use to fill rows
   * @param rows - Number of rows in the new ColumnVector
   * @return - new ColumnVector
   */
  public static ColumnVector fromScalar(Scalar scalar, int rows) {
    long amount = predictSizeFor(scalar.getType().sizeInBytes, rows, !scalar.isValid());
    try (DevicePrediction ignored = new DevicePrediction(amount, "fromScalar")) {
      long columnHandle = fromScalar(scalar.getScalarHandle(), rows);
      return new ColumnVector(columnHandle);
    }
  }

  /**
   * Create a new vector by concatenating multiple columns together.
   * Note that all columns must have the same type.
   */
  public static ColumnVector concatenate(ColumnVector... columns) {
    if (columns.length < 2) {
      throw new IllegalArgumentException("Concatenate requires 2 or more columns");
    }
    long total = 0;
    for (ColumnVector cv: columns) {
      total += cv.getDeviceMemorySize();
    }
    try (DevicePrediction prediction = new DevicePrediction(total, "concatenate")) {
      long[] columnHandles = new long[columns.length];
      for (int i = 0; i < columns.length; ++i) {
        columnHandles[i] = columns[i].getNativeView();
      }
      return new ColumnVector(concatenate(columnHandles));
    }
  }

  /**
   * Calculate the quantile of this ColumnVector
   * @param method   the method used to calculate the quantile
   * @param quantile the quantile value [0,1]
   * @return the quantile as double. The type can be changed in future
   */
  public Scalar exactQuantile(QuantileMethod method, double quantile) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return exactQuantile(this.getNativeCudfColumnAddress(), method.nativeId, quantile);
  }

  /**
   * Calculate the approximate quantile of this ColumnVector
   * @param quantile the quantile value [0,1]
   * @return the quantile, with the same type as this object
   */
  public Scalar approxQuantile(double quantile) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return approxQuantile(this.getNativeCudfColumnAddress(), quantile);
  }

  /**
   * This function aggregates values in a window around each element i of the input
   * column. Please refer to WindowsOptions for various options that can be passed.
   * @param opts various window function arguments.
   * @return Column containing aggregate function result.
   */
  public ColumnVector rollingWindow(WindowOptions opts) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    return new ColumnVector(
        rollingWindow(this.getNativeCudfColumnAddress(),
            opts.getWindow() >= 0 ? opts.getWindow() : 0,
            opts.getMinPeriods() >= 0 ? opts.getMinPeriods() : 0,
            opts.getForwardWindow() >=0 ? opts.getForwardWindow() : 0,
            opts.getAggType().nativeId,
            opts.getWindowCol() == null ? 0 : opts.getWindowCol().getNativeCudfColumnAddress(),
            opts.getMinPeriodsCol() == null ? 0 : opts.getMinPeriodsCol().getNativeCudfColumnAddress(),
            opts.getForwardWindowCol() == null ? 0 :
            opts.getForwardWindowCol().getNativeCudfColumnAddress()));
*/
  }

  /**
   * Build
   */
  public static final class Builder implements AutoCloseable {
    private final long rows;
    private final DType type;
    private HostMemoryBuffer data;
    private HostMemoryBuffer valid;
    private HostMemoryBuffer offsets;
    private long currentIndex = 0;
    private long nullCount;
    private long stringBufferSize = 0;
    private int currentStringByteIndex = 0;
    private boolean built;

    /**
     * Create a builder with a buffer of size rows
     * @param type       datatype
     * @param rows       number of rows to allocate.
     * @param stringBufferSize the size of the string data buffer if we are
     *                         working with Strings.  It is ignored otherwise.
     */
    Builder(DType type, long rows, long stringBufferSize) {
      this.type = type;
      this.rows = rows;
      if (type == DType.STRING) {
        if (stringBufferSize <= 0) {
          // We need at least one byte or we will get NULL back for data
          stringBufferSize = 1;
        }
        this.data = HostMemoryBuffer.allocate(stringBufferSize);
        // The offsets are ints and there is 1 more than the number of rows.
        this.offsets = HostMemoryBuffer.allocate((rows + 1) * OFFSET_SIZE);
        // The first offset is always 0
        this.offsets.setInt(0, 0);
        this.stringBufferSize = stringBufferSize;
      } else {
        this.data = HostMemoryBuffer.allocate(rows * type.sizeInBytes);
      }
    }

    /**
     * Create a builder with a buffer of size rows (for testing ONLY).
     * @param type       datatype
     * @param rows       number of rows to allocate.
     * @param testData   a buffer to hold the data (should be large enough to hold rows entries).
     * @param testValid  a buffer to hold the validity vector (should be large enough to hold
     *                   rows entries or is null).
     * @param testOffsets a buffer to hold the offsets for strings and string categories.
     */
    Builder(DType type, long rows, HostMemoryBuffer testData,
            HostMemoryBuffer testValid, HostMemoryBuffer testOffsets) {
      this.type = type;
      this.rows = rows;
      this.data = testData;
      this.valid = testValid;
    }

    public final Builder append(boolean value) {
      assert type == DType.BOOL8;
      assert currentIndex < rows;
      data.setByte(currentIndex * type.sizeInBytes, value ? (byte)1 : (byte)0);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value) {
      assert type == DType.INT8 || type == DType.BOOL8;
      assert currentIndex < rows;
      data.setByte(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value, long count) {
      assert (count + currentIndex) <= rows;
      assert type == DType.INT8 || type == DType.BOOL8;
      data.setMemory(currentIndex * type.sizeInBytes, count, value);
      currentIndex += count;
      return this;
    }

    public final Builder append(short value) {
      assert type == DType.INT16;
      assert currentIndex < rows;
      data.setShort(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(int value) {
      assert (type == DType.INT32 || type == DType.TIMESTAMP_DAYS);
      assert currentIndex < rows;
      data.setInt(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(long value) {
      assert type == DType.INT64 || type == DType.TIMESTAMP_MILLISECONDS ||
          type == DType.TIMESTAMP_MICROSECONDS || type == DType.TIMESTAMP_NANOSECONDS ||
          type == DType.TIMESTAMP_SECONDS;
      assert currentIndex < rows;
      data.setLong(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(float value) {
      assert type == DType.FLOAT32;
      assert currentIndex < rows;
      data.setFloat(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public final Builder append(double value) {
      assert type == DType.FLOAT64;
      assert currentIndex < rows;
      data.setDouble(currentIndex * type.sizeInBytes, value);
      currentIndex++;
      return this;
    }

    public Builder append(String value) {
      assert value != null : "appendNull must be used to append null strings";
      return appendUTF8String(value.getBytes(StandardCharsets.UTF_8));
    }

    public Builder appendUTF8String(byte[] value) {
      return appendUTF8String(value, 0, value.length);
    }

    public Builder appendUTF8String(byte[] value, int offset, int length) {
      assert value != null : "appendNull must be used to append null strings";
      assert offset >= 0;
      assert length >= 0;
      assert value.length + offset <= length;
      assert type == DType.STRING;
      assert currentIndex < rows;
      // just for strings we want to throw a real exception if we would overrun the buffer
      long oldLen = data.getLength();
      long newLen = oldLen;
      while (currentStringByteIndex + length > newLen) {
        newLen *= 2;
      }
      if (newLen > Integer.MAX_VALUE) {
        throw new IllegalStateException("A string buffer is not supported over 2GB in size");
      }
      if (newLen != oldLen) {
        // need to grow the size of the buffer.
        HostMemoryBuffer newData = HostMemoryBuffer.allocate(newLen);
        try {
          newData.copyFromHostBuffer(0, data, 0, currentStringByteIndex);
          data.close();
          data = newData;
          newData = null;
        } finally {
          if (newData != null) {
            newData.close();
          }
        }
      }
      if (length > 0) {
        data.setBytes(currentStringByteIndex, value, offset, length);
      }
      currentStringByteIndex += length;
      currentIndex++;
      offsets.setInt(currentIndex * OFFSET_SIZE, currentStringByteIndex);
      return this;
    }

    public Builder appendArray(byte... values) {
      assert (values.length + currentIndex) <= rows;
      assert type == DType.INT8 || type == DType.BOOL8;
      data.setBytes(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(short... values) {
      assert type == DType.INT16;
      assert (values.length + currentIndex) <= rows;
      data.setShorts(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(int... values) {
      assert (type == DType.INT32 || type == DType.TIMESTAMP_DAYS);
      assert (values.length + currentIndex) <= rows;
      data.setInts(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(long... values) {
      assert type == DType.INT64 || type == DType.TIMESTAMP_MILLISECONDS ||
          type == DType.TIMESTAMP_MICROSECONDS || type == DType.TIMESTAMP_NANOSECONDS ||
          type == DType.TIMESTAMP_SECONDS;
      assert (values.length + currentIndex) <= rows;
      data.setLongs(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(float... values) {
      assert type == DType.FLOAT32;
      assert (values.length + currentIndex) <= rows;
      data.setFloats(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(double... values) {
      assert type == DType.FLOAT64;
      assert (values.length + currentIndex) <= rows;
      data.setDoubles(currentIndex * type.sizeInBytes, values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Byte... values) throws IndexOutOfBoundsException {
      for (Byte b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Boolean... values) throws IndexOutOfBoundsException {
      for (Boolean b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b ? (byte) 1 : (byte) 0);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Short... values) throws IndexOutOfBoundsException {
      for (Short b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Integer... values) throws IndexOutOfBoundsException {
      for (Integer b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Long... values) throws IndexOutOfBoundsException {
      for (Long b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Float... values) throws IndexOutOfBoundsException {
      for (Float b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Double... values) throws IndexOutOfBoundsException {
      for (Double b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(String... values) throws IndexOutOfBoundsException {
      for (String b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append this vector to the end of this vector
     * @param columnVector - Vector to be added
     * @return - The CudfColumn based on this builder values
     */
    public final Builder append(ColumnVector columnVector) {
      assert columnVector.rows <= (rows - currentIndex);
      assert columnVector.type == type;
      assert columnVector.offHeap.getHostData() != null;

      if (type == DType.STRING) {
        throw new UnsupportedOperationException(
            "Appending a string column vector client side is not currently supported");
      } else {
        data.copyFromHostBuffer(currentIndex * type.sizeInBytes, columnVector.offHeap.getHostData().data,
            0L,
            columnVector.getRowCount() * type.sizeInBytes);
      }

      //As this is doing the append on the host assume that a null count is available
      long otherNc = columnVector.getNullCount();
      if (otherNc != 0) {
        if (valid == null) {
          allocateBitmaskAndSetDefaultValues();
        }
        //copy values from intCudfColumn to this
        BitVectorHelper.append(columnVector.offHeap.getHostData().valid, valid, currentIndex,
            columnVector.rows);
        nullCount += otherNc;
      }
      currentIndex += columnVector.rows;
      return this;
    }

    private void allocateBitmaskAndSetDefaultValues() {
      long bitmaskSize = CudfColumn.getNativeValidPointerSize((int) rows);
      valid = HostMemoryBuffer.allocate(bitmaskSize);
      valid.setMemory(0, bitmaskSize, (byte) 0xFF);
    }

    /**
     * Append null value.
     */
    public final Builder appendNull() {
      setNullAt(currentIndex);
      currentIndex++;
      if (type == DType.STRING) {
        offsets.setInt(currentIndex * OFFSET_SIZE, currentStringByteIndex);
      }
      return this;
    }

    /**
     * Set a specific index to null.
     * @param index
     */
    public final Builder setNullAt(long index) {
      assert index < rows;

      // add null
      if (this.valid == null) {
        allocateBitmaskAndSetDefaultValues();
      }
      nullCount += BitVectorHelper.setNullAt(valid, index);
      return this;
    }

    /**
     * Finish and create the immutable CudfColumn.
     */
    public final ColumnVector build() {
      if (built) {
        throw new IllegalStateException("Cannot reuse a builder.");
      }
      ColumnVector cv = new ColumnVector(type,
          currentIndex, Optional.of(nullCount), data, valid, offsets);
      try {
        cv.ensureOnDevice();
        built = true;
      } finally {
        if (!built) {
          cv.close();
        }
      }
      return cv;
    }

    /**
     * Finish and create the immutable CudfColumn.
     */
    public final ColumnVector buildOnHost() {
      if (built) {
        throw new IllegalStateException("Cannot reuse a builder.");
      }
      ColumnVector cv = new ColumnVector(type,
          currentIndex, Optional.of(nullCount), data, valid, offsets);
      built = true;
      return cv;
    }

    /**
     * Close this builder and free memory if the CudfColumn wasn't generated. Verifies that
     * the data was released even in the case of an error.
     */
    @Override
    public final void close() {
      if (!built) {
        data.close();
        data = null;
        if (valid != null) {
          valid.close();
          valid = null;
        }
        if (offsets != null) {
          offsets.close();
          offsets = null;
        }
        built = true;
      }
    }

    @Override
    public String toString() {
      return "Builder{" +
          "data=" + data +
          "type=" + type +
          ", valid=" + valid +
          ", currentIndex=" + currentIndex +
          ", nullCount=" + nullCount +
          ", rows=" + rows +
          ", built=" + built +
          '}';
    }
  }

  /**
   * Transform a vector
   * @param udf This function will be applied to every element in the vector
   * @param isPtx Is this function a PTX
   * @return
   */
  public ColumnVector transform(String udf, boolean isPtx) {
    return new ColumnVector(offHeap.cudfColumnHandle.transform(udf, isPtx));
  }
}
