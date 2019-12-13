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

import java.io.File;
import java.util.*;

import static ai.rapids.cudf.ColumnVector.STANDARD_CUDF_PORTING_MSG;

/**
 * Class to represent a collection of ColumnVectors and operations that can be performed on them
 * collectively.
 * The refcount on the columns will be increased once they are passed in
 */
public final class Table implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private final long rows;
  private long nativeHandle;
  private ColumnVector[] columns;
  // This is an estimate of how compressed data is when guessing the output size of data
  // For ORC and Parquet this is relatively conservative. We might want a different
  // one for text based formats.
  private static final long COMPRESSION_RATIO_ESTIMATE = 10;

  /**
   * Table class makes a copy of the array of {@link ColumnVector}s passed to it. The class
   * will decrease the refcount
   * on itself and all its contents when closed and free resources if refcount is zero
   * @param columns - Array of ColumnVectors
   */
  public Table(ColumnVector... columns) {
    assert columns != null : "ColumnVectors can't be null";
    rows = columns.length > 0 ? columns[0].getRowCount() : 0;

    for (ColumnVector columnVector : columns) {
      assert (null != columnVector) : "ColumnVectors can't be null";
      assert (rows == columnVector.getRowCount()) : "All columns should have the same number of " +
          "rows " + columnVector.getType();
    }

    // Since Arrays are mutable objects make a copy
    this.columns = new ColumnVector[columns.length];
    long[] cudfColumnPointers = new long[columns.length];
    for (int i = 0; i < columns.length; i++) {
      this.columns[i] = columns[i];
      columns[i].incRefCount();
      cudfColumnPointers[i] = columns[i].getNativeCudfColumnAddress();
    }

    nativeHandle = createCudfTableView(cudfColumnPointers);
  }

  private Table(long[] cudfColumns) {
    assert cudfColumns != null : "CudfColumns can't be null";
    this.columns = new ColumnVector[cudfColumns.length];
    try {
      for (int i = 0; i < cudfColumns.length; i++) {
        this.columns[i] = new ColumnVector(cudfColumns[i]);
      }
      nativeHandle = createCudfTableView(cudfColumns);
      this.rows = columns[0].getRowCount();
    } catch (Throwable t) {
      for (int i = 0; i < cudfColumns.length; i++) {
        if (this.columns[i] != null) {
          this.columns[i].close();
        } else {
          new CudfColumn(cudfColumns[i]).deleteCudfColumn();
        }
      }
      throw t;
    }
  }

  /**
   * Provides a faster way to get access to the columns. Only to be used internally, and it should
   * never be modified in anyway.
   */
  ColumnVector[] getColumns() {
    return columns;
  }

  /**
   * Return the {@link ColumnVector} at the specified index. If you want to keep a reference to
   * the column around past the life time of the table, you will need to increment the reference
   * count on the column yourself.
   */
  public ColumnVector getColumn(int index) {
    assert index < columns.length;
    return columns[index];
  }

  public final long getRowCount() {
    return rows;
  }

  public final int getNumberOfColumns() {
    return columns.length;
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      deleteCudfTable(nativeHandle);
      nativeHandle = 0;
    }
    if (columns != null) {
      for (int i = 0; i < columns.length; i++) {
        columns[i].close();
        columns[i] = null;
      }
      columns = null;
    }
  }

  @Override
  public String toString() {
    return "Table{" +
        "columns=" + Arrays.toString(columns) +
        ", cudfTable=" + nativeHandle +
        ", rows=" + rows +
        '}';
  }

  /**
   * Returns the Device memory buffer size.
   */
  public long getDeviceMemorySize() {
    long total = 0;
    for (ColumnVector cv: columns) {
      total += cv.getDeviceMemorySize();
    }
    return total;
  }

  /////////////////////////////////////////////////////////////////////////////
  // NATIVE APIs
  /////////////////////////////////////////////////////////////////////////////

  private static native long[] gdfPartition(long inputTable,
                                            int[] columnsToHash,
                                            int cudfHashFunction,
                                            int numberOfPartitions,
                                            int[] outputOffsets) throws CudfException;

  private static native void deleteCudfTable(long handle) throws CudfException;

  private static native long[] gdfReadJSON(String filePath, long bufferAddress, long bufferLength, long startRange, long rangeLength, String[] filterColumnNames, String[] columnNames, String[] typesAsStrings) throws CudfException;

  private static native long bound(long inputTable, long valueTable,
                                   boolean[] descFlags, boolean[] areNullsSmallest, boolean isUpperBound) throws CudfException;

  private static native void gdfWriteORC(int compressionType, String outputFileName, long buffer, long bufferLength, long tableToWrite) throws CudfException;

  /**
   * Ugly long function to read CSV.  This is a long function to avoid the overhead of reaching
   * into a java
   * object to try and pull out all of the options.  If this becomes unwieldy we can change it.
   * @param columnNames       names of all of the columns, even the ones filtered out
   * @param dTypes            types of all of the columns as strings.  Why strings? who knows.
   * @param filterColumnNames name of the columns to read, or an empty array if we want to read
   *                          all of them
   * @param filePath          the path of the file to read, or null if no path should be read.
   * @param address           the address of the buffer to read from or 0 if we should not.
   * @param length            the length of the buffer to read from.
   * @param headerRow         the 0 based index row of the header can be -1
   * @param delim             character deliminator (must be ASCII).
   * @param quote             character quote (must be ASCII).
   * @param comment           character that starts a comment line (must be ASCII) use '\0'
   * @param nullValues        values that should be treated as nulls
   * @param trueValues        values that should be treated as boolean true
   * @param falseValues       values that should be treated as boolean false
   */
  private static native long[] readCSV(String[] columnNames, String[] dTypes,
                                       String[] filterColumnNames,
                                       String filePath, long address, long length,
                                       int headerRow, byte delim, byte quote,
                                       byte comment, String[] nullValues,
                                       String[] trueValues, String[] falseValues) throws CudfException;

  /**
   * Read in Parquet formatted data.
   * @param filterColumnNames name of the columns to read, or an empty array if we want to read
   *                          all of them
   * @param filePath          the path of the file to read, or null if no path should be read.
   * @param address           the address of the buffer to read from or 0 if we should not.
   * @param length            the length of the buffer to read from.
   * @param timeUnit          return type of TimeStamp in units
   */
  private static native long[] readParquet(String[] filterColumnNames, String filePath,
                                           long address, long length, int timeUnit) throws CudfException;

  /**
   * Read in ORC formatted data.
   * @param filterColumnNames name of the columns to read, or an empty array if we want to read
   *                          all of them
   * @param filePath          the path of the file to read, or null if no path should be read.
   * @param address           the address of the buffer to read from or 0 for no buffer.
   * @param length            the length of the buffer to read from.
   * @param usingNumPyTypes   whether the parser should implicitly promote DATE32 and TIMESTAMP
   *                          columns to DATE64 for compatibility with NumPy.
   * @param timeUnit          return type of TimeStamp in units
   */
  private static native long[] readORC(String[] filterColumnNames,
                                       String filePath, long address, long length,
                                       boolean usingNumPyTypes, int timeUnit) throws CudfException;

  private static native long[] gdfGroupByAggregate(long inputTable, int[] keyIndices, int[] aggColumnsIndices,
                                                   int[] aggTypes, boolean ignoreNullKeys) throws CudfException;

  private static native long[] orderBy(long inputTable, long[] sortKeys, boolean[] isDescending,
                                       boolean[] areNullsSmallest) throws CudfException;

  private static native long[] gdfLeftJoin(long leftTable, int[] leftJoinCols, long rightTable,
                                           int[] rightJoinCols) throws CudfException;

  private static native long[] gdfInnerJoin(long leftTable, int[] leftJoinCols, long rightTable,
                                            int[] rightJoinCols) throws CudfException;

  private static native long[] concatenate(long[] cudfTablePointers) throws CudfException;

  private static native long[] filter(long input, long mask);

  //XXX until we have split a ColumnVector into a host column and a device column
  // caching the table_view is a bug, as we could drop the device data which would
  // invalidate everything that the table_view is pointing at on the device.
  private native long createCudfTableView(long[] nativeColumnHandles);

  /////////////////////////////////////////////////////////////////////////////
  // TABLE CREATION APIs
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Read a JSON file
   * This method interprets each line as a complete json object and accepts json in two formats
   * If the following format is passed, the column names will be inferred as 0, 1, 2,...
   *  "[1, 1.2, \"hello\"]\n
   *   [3, 2.3, \"string\"]\n
   *   [2, 23.2, \"str\"]\n"
   *
   * If the following format is passed, the column names will be assigned as col1, col2, col3,...
   *  "{\"col1\": 1, \"col2\": 1.1, \"col3\": \"a\"}\n
   *   {\"col1\": 3, \"col2\": 4.2, \"col3\": \"hello\"}\n
   *   {\"col1\": 7, \"col2\": 1.3, \"col3\": \"seven\u24E1\u25B6\"}"
   *
   * @param path local path to the json file
   * @return the file parsed as a table on the GPU.
   */
  public static Table readJSON(File path) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return readJSON(JSONOptions.DEFAULT, path);
  }

  /**
   * Read a JSON file
   * This method interprets each line as a complete json object and accepts json in two formats
   * If the following format is passed, the column names will be inferred as 0, 1, 2,...
   *  "[1, 1.2, \"hello\"]\n
   *   [3, 2.3, \"string\"]\n
   *   [2, 23.2, \"str\"]\n"
   *
   * If the following format is passed, the column names will be assigned as col1, col2, col3,...
   *  "{\"col1\": 1, \"col2\": 1.1, \"col3\": \"a\"}\n
   *   {\"col1\": 3, \"col2\": 4.2, \"col3\": \"hello\"}\n
   *   {\"col1\": 7, \"col2\": 1.3, \"col3\": \"seven\u24E1\u25B6\"}"
   *
   * @param opts JSONOptions to use to parse the file, currently only offset and size is supported
   * @param path local path to the json file
   * @return the file parsed as a table on the GPU.
   */
  public static Table readJSON(JSONOptions opts, File path) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return readJSON(Schema.INFERRED, opts, path);
  }

  /**
   * Read a JSON file
   * This method interprets each line as a complete json object and accepts json in two formats
   * If the following format is passed, the column names will be inferred as 0, 1, 2,...
   *  "[1, 1.2, \"hello\"]\n
   *   [3, 2.3, \"string\"]\n
   *   [2, 23.2, \"str\"]\n"
   *
   * If the following format is passed, the column names will be assigned as col1, col2, col3,...
   *  "{\"col1\": 1, \"col2\": 1.1, \"col3\": \"a\"}\n
   *   {\"col1\": 3, \"col2\": 4.2, \"col3\": \"hello\"}\n
   *   {\"col1\": 7, \"col2\": 1.3, \"col3\": \"seven\u24E1\u25B6\"}"
   *
   * @param schema Schema containing the data types to use for the returned table
   * @param path local path to the json file
   * @return the file parsed as a table on the GPU.
   */
  public static Table readJSON(Schema schema, File path) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return readJSON(schema, JSONOptions.DEFAULT, path);
  }

  /**
   * Read a JSON file
   * This method interprets each line as a complete json object and accepts json in two formats
   * If the following format is passed, the column names will be inferred as 0, 1, 2,...
   *  "[1, 1.2, \"hello\"]\n
   *   [3, 2.3, \"string\"]\n
   *   [2, 23.2, \"str\"]\n"
   *
   * If the following format is passed, the column names will be assigned as col1, col2, col3,...
   *  "{\"col1\": 1, \"col2\": 1.1, \"col3\": \"a\"}\n
   *   {\"col1\": 3, \"col2\": 4.2, \"col3\": \"hello\"}\n
   *   {\"col1\": 7, \"col2\": 1.3, \"col3\": \"seven\u24E1\u25B6\"}"
   *
   * @param schema Schema containing the data types to use for the returned table
   * @param opts JSONOptions to use to parse the file, currently only offset and size is supported
   * @param path local path to the json file
   * @return the file parsed as a table on the GPU.
   */
  public static Table readJSON(Schema schema, JSONOptions opts, File path) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    long amount = opts.getSizeGuessOrElse(() -> path.length());
    try (DevicePrediction prediction = new DevicePrediction(amount, "JSON FILE")) {
      return new Table(gdfReadJSON(path.getAbsolutePath(), 0, 0, 0, 0, opts.getIncludeColumnNames(), schema.getColumnNames(), schema.getTypesAsStrings()));
    }
*/
  }

  /**
   * Read a JSON file
   * This method interprets each line as a complete json object and accepts json in two formats
   * If the following format is passed, the column names will be inferred as 0, 1, 2,...
   *  "[1, 1.2, \"hello\"]\n
   *   [3, 2.3, \"string\"]\n
   *   [2, 23.2, \"str\"]\n"
   *
   * If the following format is passed, the column names will be assigned as col1, col2, col3,...
   *  "{\"col1\": 1, \"col2\": 1.1, \"col3\": \"a\"}\n
   *   {\"col1\": 3, \"col2\": 4.2, \"col3\": \"hello\"}\n
   *   {\"col1\": 7, \"col2\": 1.3, \"col3\": \"seven\u24E1\u25B6\"}"
   *
   * @param buffer json data
   * @return the file parsed as a table on the GPU.
   */
  public static Table readJSON(byte[] buffer) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return readJSON(JSONOptions.DEFAULT, buffer);
  }

  /**
   * Read a JSON file
   * This method interprets each line as a complete json object and accepts json in two formats
   * If the following format is passed, the column names will be inferred as 0, 1, 2,...
   *  "[1, 1.2, \"hello\"]\n
   *   [3, 2.3, \"string\"]\n
   *   [2, 23.2, \"str\"]\n"
   *
   * If the following format is passed, the column names will be assigned as col1, col2, col3,...
   *  "{\"col1\": 1, \"col2\": 1.1, \"col3\": \"a\"}\n
   *   {\"col1\": 3, \"col2\": 4.2, \"col3\": \"hello\"}\n
   *   {\"col1\": 7, \"col2\": 1.3, \"col3\": \"seven\u24E1\u25B6\"}"
   *
   * @param schema Schema containing the data types to use for the returned table
   * @param buffer json data
   * @return the file parsed as a table on the GPU.
   */
  public static Table readJSON(Schema schema, byte[] buffer) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return readJSON(schema, JSONOptions.DEFAULT, buffer);
  }

  /**
   * Read a JSON file
   * This method interprets each line as a complete json object and accepts json in two formats
   * If the following format is passed, the column names will be inferred as 0, 1, 2,...
   *  "[1, 1.2, \"hello\"]\n
   *   [3, 2.3, \"string\"]\n
   *   [2, 23.2, \"str\"]\n"
   *
   * If the following format is passed, the column names will be assigned as col1, col2, col3,...
   *  "{\"col1\": 1, \"col2\": 1.1, \"col3\": \"a\"}\n
   *   {\"col1\": 3, \"col2\": 4.2, \"col3\": \"hello\"}\n
   *   {\"col1\": 7, \"col2\": 1.3, \"col3\": \"seven\u24E1\u25B6\"}"
   *
   * @param opts JSONOptions to use to parse the file, currently only offset and size is supported
   * @param buffer json data
   * @return the file parsed as a table on the GPU.
   */
  public static Table readJSON(JSONOptions opts, byte[] buffer) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return readJSON(Schema.INFERRED, opts, buffer);
  }

  /**
   * Read a JSON file
   * This method interprets each line as a complete json object and accepts json in two formats
   * If the following format is passed, the column names will be inferred as 0, 1, 2,...
   *  "[1, 1.2, \"hello\"]\n
   *   [3, 2.3, \"string\"]\n
   *   [2, 23.2, \"str\"]\n"
   *
   * If the following format is passed, the column names will be assigned as col1, col2, col3,...
   *  "{\"col1\": 1, \"col2\": 1.1, \"col3\": \"a\"}\n
   *   {\"col1\": 3, \"col2\": 4.2, \"col3\": \"hello\"}\n
   *   {\"col1\": 7, \"col2\": 1.3, \"col3\": \"seven\u24E1\u25B6\"}"
   *
   * @param schema Schema containing the data types to use for the returned table
   * @param opts JSONOptions to use to parse the file, currently only offset and size is supported
   * @param buffer json data
   * @return the file parsed as a table on the GPU.
   */
  public static Table readJSON(Schema schema, JSONOptions opts, byte[] buffer) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return readJSON(schema, opts, buffer, 0, 0);
  }

  /**
   * Read a JSON file
   * This method interprets each line as a complete json object and accepts json in two formats
   * If the following format is passed, the column names will be inferred as 0, 1, 2,...
   *  "[1, 1.2, \"hello\"]\n
   *   [3, 2.3, \"string\"]\n
   *   [2, 23.2, \"str\"]\n"
   *
   * If the following format is passed, the column names will be assigned as col1, col2, col3,...
   *  "{\"col1\": 1, \"col2\": 1.1, \"col3\": \"a\"}\n
   *   {\"col1\": 3, \"col2\": 4.2, \"col3\": \"hello\"}\n
   *   {\"col1\": 7, \"col2\": 1.3, \"col3\": \"seven\u24E1\u25B6\"}"
   *
   * @param schema Schema containing the data types to use for the returned table
   * @param opts JSONOptions to use to parse the file, currently only offset and size is supported
   * @param buffer json data
   * @param offset start offset to start in buffer
   * @param len size of the buffer to read after offset
   * @return the file parsed as a table on the GPU.
   */
  public static Table readJSON(Schema schema, JSONOptions opts, byte[] buffer, long offset, long len) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    if (len == 0) {
      len = buffer.length;
    }
    assert len > 0 : "Invalid buffer range size";
    assert len <= buffer.length - offset : "Buffer range size greater than buffer";
    assert offset >= 0 && offset < len : "Buffer offset out of range";
    try (HostPrediction prediction = new HostPrediction(len, "readJSON");
        HostMemoryBuffer newBuf = HostMemoryBuffer.allocate(len)) {
      newBuf.setBytes(0, buffer, offset, len);
      // using default ranges but keeping the included column names
      return readJSON(schema, opts, newBuf, 0, 0);
    }
*/
  }

  /**
   * Read a JSON file
   * This method interprets each line as a complete json object and accepts json in two formats
   * If the following format is passed, the column names will be inferred as 0, 1, 2,...
   *  "[1, 1.2, \"hello\"]\n
   *   [3, 2.3, \"string\"]\n
   *   [2, 23.2, \"str\"]\n"
   *
   * If the following format is passed, the column names will be assigned as col1, col2, col3,...
   *  "{\"col1\": 1, \"col2\": 1.1, \"col3\": \"a\"}\n
   *   {\"col1\": 3, \"col2\": 4.2, \"col3\": \"hello\"}\n
   *   {\"col1\": 7, \"col2\": 1.3, \"col3\": \"seven\u24E1\u25B6\"}"
   *
   * @param schema Schema containing the data types to use for the returned table
   * @param buffer json data
   * @param offset start offset to start in buffer
   * @param len size of the buffer to read after offset
   * @return the file parsed as a table on the GPU.
   */
  public static Table readJSON(Schema schema, JSONOptions opts, HostMemoryBuffer buffer, long offset, long len) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    if (len == 0) {
      len = buffer.length;
    }
    assert len > 0 : "Invalid buffer range size";
    assert len <= buffer.length - offset : "Buffer range size greater than buffer";
    assert offset >= 0 && offset < buffer.length : "Buffer offset out of range";
    long amount = opts.getSizeGuessOrElse(len);
    try (DevicePrediction prediction = new DevicePrediction(amount, "JSON BUFFER")) {
      return new Table(gdfReadJSON(null, buffer.getAddress() + offset, buffer.getLength(), offset, len, opts.getIncludeColumnNames(), schema.getColumnNames(), schema.getTypesAsStrings()));
    }
*/
  }

  /**
   * Read a CSV file using the default CSVOptions.
   * @param schema the schema of the file.  You may use Schema.INFERRED to infer the schema.
   * @param path the local file to read.
   * @return the file parsed as a table on the GPU.
   */
  public static Table readCSV(Schema schema, File path) {
    return readCSV(schema, CSVOptions.DEFAULT, path);
  }

  /**
   * Read a CSV file.
   * @param schema the schema of the file.  You may use Schema.INFERRED to infer the schema.
   * @param opts various CSV parsing options.
   * @param path the local file to read.
   * @return the file parsed as a table on the GPU.
   */
  public static Table readCSV(Schema schema, CSVOptions opts, File path) {
    long amount = opts.getSizeGuessOrElse(() -> path.length());
    try (DevicePrediction prediction = new DevicePrediction(amount, "CSV FILE")) {
      return new Table(
          readCSV(schema.getColumnNames(), schema.getTypesAsStrings(),
              opts.getIncludeColumnNames(), path.getAbsolutePath(),
              0, 0,
              opts.getHeaderRow(),
              opts.getDelim(),
              opts.getQuote(),
              opts.getComment(),
              opts.getNullValues(),
              opts.getTrueValues(),
              opts.getFalseValues()));
    }
  }

  /**
   * Read CSV formatted data using the default CSVOptions.
   * @param schema the schema of the data. You may use Schema.INFERRED to infer the schema.
   * @param buffer raw UTF8 formatted bytes.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readCSV(Schema schema, byte[] buffer) {
    return readCSV(schema, CSVOptions.DEFAULT, buffer, 0, buffer.length);
  }

  /**
   * Read CSV formatted data.
   * @param schema the schema of the data. You may use Schema.INFERRED to infer the schema.
   * @param opts various CSV parsing options.
   * @param buffer raw UTF8 formatted bytes.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readCSV(Schema schema, CSVOptions opts, byte[] buffer) {
    return readCSV(schema, opts, buffer, 0, buffer.length);
  }

  /**
   * Read CSV formatted data.
   * @param schema the schema of the data. You may use Schema.INFERRED to infer the schema.
   * @param opts various CSV parsing options.
   * @param buffer raw UTF8 formatted bytes.
   * @param offset the starting offset into buffer.
   * @param len the number of bytes to parse.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readCSV(Schema schema, CSVOptions opts, byte[] buffer, long offset,
                              long len) {
    if (len <= 0) {
      len = buffer.length - offset;
    }
    assert len > 0;
    assert len <= buffer.length - offset;
    assert offset >= 0 && offset < buffer.length;
    try (HostPrediction prediction = new HostPrediction(len, "readCSV");
        HostMemoryBuffer newBuf = HostMemoryBuffer.allocate(len)) {
      newBuf.setBytes(0, buffer, offset, len);
      return readCSV(schema, opts, newBuf, 0, len);
    }
  }

  /**
   * Read CSV formatted data.
   * @param schema the schema of the data. You may use Schema.INFERRED to infer the schema.
   * @param opts various CSV parsing options.
   * @param buffer raw UTF8 formatted bytes.
   * @param offset the starting offset into buffer.
   * @param len the number of bytes to parse.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readCSV(Schema schema, CSVOptions opts, HostMemoryBuffer buffer,
                              long offset, long len) {
    if (len <= 0) {
      len = buffer.length - offset;
    }
    assert len > 0;
    assert len <= buffer.getLength() - offset;
    assert offset >= 0 && offset < buffer.length;
    long amount = opts.getSizeGuessOrElse(len);
    try (DevicePrediction prediction = new DevicePrediction(amount, "CSV BUFFER")) {
      return new Table(readCSV(schema.getColumnNames(), schema.getTypesAsStrings(),
          opts.getIncludeColumnNames(), null,
          buffer.getAddress() + offset, len,
          opts.getHeaderRow(),
          opts.getDelim(),
          opts.getQuote(),
          opts.getComment(),
          opts.getNullValues(),
          opts.getTrueValues(),
          opts.getFalseValues()));
    }
  }

  /**
   * Read a Parquet file using the default ParquetOptions.
   * @param path the local file to read.
   * @return the file parsed as a table on the GPU.
   */
  public static Table readParquet(File path) {
    return readParquet(ParquetOptions.DEFAULT, path);
  }

  /**
   * Read a Parquet file.
   * @param opts various parquet parsing options.
   * @param path the local file to read.
   * @return the file parsed as a table on the GPU.
   */
  public static Table readParquet(ParquetOptions opts, File path) {
    long amount = opts.getSizeGuessOrElse(() -> path.length() * COMPRESSION_RATIO_ESTIMATE);
    try (DevicePrediction prediction = new DevicePrediction(amount, "PARQUET FILE")) {
      return new Table(readParquet(opts.getIncludeColumnNames(),
          path.getAbsolutePath(), 0, 0, opts.timeUnit().nativeId));
    }
  }

  /**
   * Read parquet formatted data.
   * @param buffer raw parquet formatted bytes.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readParquet(byte[] buffer) {
    return readParquet(ParquetOptions.DEFAULT, buffer, 0, buffer.length);
  }

  /**
   * Read parquet formatted data.
   * @param opts various parquet parsing options.
   * @param buffer raw parquet formatted bytes.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readParquet(ParquetOptions opts, byte[] buffer) {
    return readParquet(opts, buffer, 0, buffer.length);
  }

  /**
   * Read parquet formatted data.
   * @param opts various parquet parsing options.
   * @param buffer raw parquet formatted bytes.
   * @param offset the starting offset into buffer.
   * @param len the number of bytes to parse.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readParquet(ParquetOptions opts, byte[] buffer, long offset, long len) {
    if (len <= 0) {
      len = buffer.length - offset;
    }
    assert len > 0;
    assert len <= buffer.length - offset;
    assert offset >= 0 && offset < buffer.length;
    try (HostPrediction prediction = new HostPrediction(len, "readParquet");
        HostMemoryBuffer newBuf = HostMemoryBuffer.allocate(len)) {
      newBuf.setBytes(0, buffer, offset, len);
      return readParquet(opts, newBuf, 0, len);
    }
  }

  /**
   * Read parquet formatted data.
   * @param opts various parquet parsing options.
   * @param buffer raw parquet formatted bytes.
   * @param offset the starting offset into buffer.
   * @param len the number of bytes to parse.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readParquet(ParquetOptions opts, HostMemoryBuffer buffer,
                                  long offset, long len) {
    if (len <= 0) {
      len = buffer.length - offset;
    }
    assert len > 0;
    assert len <= buffer.getLength() - offset;
    assert offset >= 0 && offset < buffer.length;
    long amount = opts.getSizeGuessOrElse(len * COMPRESSION_RATIO_ESTIMATE);
    try (DevicePrediction prediction = new DevicePrediction(amount, "PARQUET BUFFER")) {
      return new Table(readParquet(opts.getIncludeColumnNames(),
          null, buffer.getAddress() + offset, len, opts.timeUnit().nativeId));
    }
  }

  /**
   * Read a ORC file using the default ORCOptions.
   * @param path the local file to read.
   * @return the file parsed as a table on the GPU.
   */
  public static Table readORC(File path) {
    return readORC(ORCOptions.DEFAULT, path);
  }

  /**
   * Read a ORC file.
   * @param opts ORC parsing options.
   * @param path the local file to read.
   * @return the file parsed as a table on the GPU.
   */
  public static Table readORC(ORCOptions opts, File path) {
    long amount = opts.getSizeGuessOrElse(() -> path.length() * COMPRESSION_RATIO_ESTIMATE);
    try (DevicePrediction prediction = new DevicePrediction(amount, "ORC FILE")) {
      return new Table(readORC(opts.getIncludeColumnNames(),
          path.getAbsolutePath(), 0, 0, opts.usingNumPyTypes(), opts.timeUnit().nativeId));
    }
  }

  /**
   * Read ORC formatted data.
   * @param buffer raw ORC formatted bytes.
   * @param buffer raw ORC formatted bytes.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readORC(byte[] buffer) {
    return readORC(ORCOptions.DEFAULT, buffer, 0, buffer.length);
  }

  /**
   * Read ORC formatted data.
   * @param opts various ORC parsing options.
   * @param buffer raw ORC formatted bytes.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readORC(ORCOptions opts, byte[] buffer) {
    return readORC(opts, buffer, 0, buffer.length);
  }

  /**
   * Read ORC formatted data.
   * @param opts various ORC parsing options.
   * @param buffer raw ORC formatted bytes.
   * @param offset the starting offset into buffer.
   * @param len the number of bytes to parse.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readORC(ORCOptions opts, byte[] buffer, long offset, long len) {
    if (len <= 0) {
      len = buffer.length - offset;
    }
    assert len > 0;
    assert len <= buffer.length - offset;
    assert offset >= 0 && offset < buffer.length;
    try (HostPrediction prediction = new HostPrediction(len, "readORC");
        HostMemoryBuffer newBuf = HostMemoryBuffer.allocate(len)) {
      newBuf.setBytes(0, buffer, offset, len);
      return readORC(opts, newBuf, 0, len);
    }
  }

  /**
   * Read ORC formatted data.
   * @param opts various ORC parsing options.
   * @param buffer raw ORC formatted bytes.
   * @param offset the starting offset into buffer.
   * @param len the number of bytes to parse.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readORC(ORCOptions opts, HostMemoryBuffer buffer,
                              long offset, long len) {
    if (len <= 0) {
      len = buffer.length - offset;
    }
    assert len > 0;
    assert len <= buffer.getLength() - offset;
    assert offset >= 0 && offset < buffer.length;
    long amount = opts.getSizeGuessOrElse(len * COMPRESSION_RATIO_ESTIMATE);
    try (DevicePrediction prediction = new DevicePrediction(amount, "ORC BUFFER")) {
      return new Table(readORC(opts.getIncludeColumnNames(),
          null, buffer.getAddress() + offset, len, opts.usingNumPyTypes(),
          opts.timeUnit().nativeId));
    }
  }

  /**
   * Writes this table to a file on the host
   *
   * @param outputFile - File to write the table to
   */
  public void writeORC(File outputFile) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    gdfWriteORC(ORCWriterOptions.DEFAULT.getCompressionType().nativeId, outputFile.getAbsolutePath(), 0, 0, this.nativeHandle);
  }

  /**
   * Writes this table to a file on the host
   *
   * @param outputFile - File to write the table to
   */
  public void writeORC(ORCWriterOptions options, File outputFile) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    gdfWriteORC(options.getCompressionType().nativeId, outputFile.getAbsolutePath(), 0, 0, this.nativeHandle);
  }

  /**
   * Concatenate multiple tables together to form a single table.
   * The schema of each table (i.e.: number of columns and types of each column) must be equal
   * across all tables and will determine the schema of the resulting table.
   */
  public static Table concatenate(Table... tables) {
    if (tables.length < 2) {
      throw new IllegalArgumentException("concatenate requires 2 or more tables");
    }
    long amount = 0;
    int numColumns = tables[0].getNumberOfColumns();
    long[] tableHandles = new long[tables.length];
    for (int i = 0; i < tables.length; ++i) {
      amount += tables[i].getDeviceMemorySize();
      tableHandles[i] = tables[i].nativeHandle;
      assert tables[i].getNumberOfColumns() == numColumns : "all tables must have the same schema";
    }
    try (DevicePrediction prediction = new DevicePrediction(amount, "concat")) {
      return new Table(concatenate(tableHandles));
    }
  }

  /**
   * Given a sorted table return the lower bound.
   * Example:
   *
   *  Single column:
   *      idx            0   1   2   3   4
   *   inputTable  =   { 10, 20, 20, 30, 50 }
   *   valuesTable =   { 20 }
   *   result      =   { 1 }
   *
   *  Multi Column:
   *      idx                0    1    2    3    4
   *   inputTable      = {{  10,  20,  20,  20,  20 },
   *                      { 5.0,  .5,  .5,  .7,  .7 },
   *                      {  90,  77,  78,  61,  61 }}
   *   valuesTable     = {{ 20 },
   *                      { .7 },
   *                      { 61 }}
   *   result          = {  3 }
   * NaNs in column values produce incorrect results.
   * The input table and the values table need to be non-empty (row count > 0)
   * The column data types of the tables' have to match in order.
   * Strings and String categories do not work for this method. If the input table is
   * unsorted the results are wrong. Types of columns can be of mixed data types.
   * @param areNullsSmallest true if nulls are assumed smallest
   * @param valueTable the table of values that need to be inserted
   * @param descFlags indicates the ordering of the column(s), true if descending
   * @return ColumnVector with lower bound indices for all rows in valueTable
   */
  public ColumnVector lowerBound(boolean[] areNullsSmallest,
      Table valueTable, boolean[] descFlags) {
    assertForBounds(valueTable);
    return new ColumnVector(bound(this.nativeHandle, valueTable.nativeHandle,
      descFlags, areNullsSmallest, false));
  }

  /**
   * Given a sorted table return the upper bound.
   * Example:
   *
   *  Single column:
   *      idx            0   1   2   3   4
   *   inputTable  =   { 10, 20, 20, 30, 50 }
   *   valuesTable =   { 20 }
   *   result      =   { 3 }
   *
   *  Multi Column:
   *      idx                0    1    2    3    4
   *   inputTable      = {{  10,  20,  20,  20,  20 },
   *                      { 5.0,  .5,  .5,  .7,  .7 },
   *                      {  90,  77,  78,  61,  61 }}
   *   valuesTable     = {{ 20 },
   *                      { .7 },
   *                      { 61 }}
   *   result          = {  5 }
   * NaNs in column values produce incorrect results.
   * The input table and the values table need to be non-empty (row count > 0)
   * The column data types of the tables' have to match in order.
   * Strings and String categories do not work for this method. If the input table is
   * unsorted the results are wrong. Types of columns can be of mixed data types.
   * @param areNullsSmallest true if nulls are assumed smallest
   * @param valueTable the table of values that need to be inserted
   * @param descFlags indicates the ordering of the column(s), true if descending
   * @return ColumnVector with upper bound indices for all rows in valueTable
   */
  public ColumnVector upperBound(boolean[] areNullsSmallest,
      Table valueTable, boolean[] descFlags) {
    assertForBounds(valueTable);
    return new ColumnVector(bound(this.nativeHandle, valueTable.nativeHandle,
      descFlags, areNullsSmallest, true));
  }

  private void assertForBounds(Table valueTable) {
    assert this.getRowCount() != 0 : "Input table cannot be empty";
    assert valueTable.getRowCount() != 0 : "Value table cannot be empty";
    for (int i = 0; i < Math.min(columns.length, valueTable.columns.length); i++) {
      assert valueTable.columns[i].getType() == this.getColumn(i).getType() :
          "Input and values tables' data types do not match";
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // TABLE MANIPULATION APIs
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Orders the table using the sortkeys returning a new allocated table. The caller is
   * responsible for cleaning up
   * the {@link ColumnVector} returned as part of the output {@link Table}
   * <p>
   * Example usage: orderBy(true, Table.asc(0), Table.desc(3)...);
   * @param args             - Suppliers to initialize sortKeys.
   * @return Sorted Table
   */
  public Table orderBy(OrderByArg... args) {
    assert args.length <= columns.length;
    long[] sortKeys = new long[args.length];
    boolean[] isDescending = new boolean[args.length];
    boolean[] areNullsSmallest = new boolean[args.length];
    for (int i = 0; i < args.length; i++) {
      int index = args[i].index;
      assert (index >= 0 && index < columns.length) :
          "index is out of range 0 <= " + index + " < " + columns.length;
      isDescending[i] = args[i].isDescending;
      areNullsSmallest[i] = args[i].isNullSmallest;
      sortKeys[i] = columns[index].getNativeCudfColumnAddress();
    }

    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "orderBy")) {
      return new Table(orderBy(nativeHandle, sortKeys, isDescending, areNullsSmallest));
    }
  }

  public static OrderByArg asc(final int index) {
    return new OrderByArg(index, false, false);
  }

  public static OrderByArg desc(final int index) {
    return new OrderByArg(index, true, false);
  }

  public static OrderByArg asc(final int index, final boolean isNullSmallest) {
    return new OrderByArg(index, false, isNullSmallest);
  }

  public static OrderByArg desc(final int index, final boolean isNullSmallest) {
    return new OrderByArg(index, true, isNullSmallest);
  }

  public static Aggregate count(int index) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return Aggregate.count(index);
  }

  public static Aggregate max(int index) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return Aggregate.max(index);
  }

  public static Aggregate min(int index) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return Aggregate.min(index);
  }

  public static Aggregate sum(int index) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return Aggregate.sum(index);
  }

  public static Aggregate mean(int index) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return Aggregate.mean(index);
  }

  public AggregateOperation groupBy(GroupByOptions groupByOptions, int... indices) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return groupByInternal(groupByOptions, indices);
  }

  public AggregateOperation groupBy(int... indices) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    return groupByInternal(GroupByOptions.builder().withIgnoreNullKeys(false).build(),
        indices);
*/
  }

  private AggregateOperation groupByInternal(GroupByOptions groupByOptions, int[] indices) {
    int[] operationIndicesArray = copyAndValidate(indices);
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
//    return new AggregateOperation(this, groupByOptions, operationIndicesArray);
  }

  public TableOperation onColumns(int... indices) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    int[] operationIndicesArray = copyAndValidate(indices);
    return new TableOperation(this, operationIndicesArray);
*/
  }

  private int[] copyAndValidate(int[] indices) {
    throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
    int[] operationIndicesArray = new int[indices.length];
    for (int i = 0; i < indices.length; i++) {
      operationIndicesArray[i] = indices[i];
      assert operationIndicesArray[i] >= 0 && operationIndicesArray[i] < columns.length :
          "operation index is out of range 0 <= " + operationIndicesArray[i] + " < " + columns.length;
    }
    return operationIndicesArray;
*/
  }

  /**
   * Filters this table using a column of boolean values as a mask, returning a new one.
   * <p>
   * Given a mask column, each element `i` from the input columns
   * is copied to the output columns if the corresponding element `i` in the mask is
   * non-null and `true`. This operation is stable: the input order is preserved.
   * <p>
   * This table and mask columns must have the same number of rows.
   * <p>
   * The output table has size equal to the number of elements in boolean_mask
   * that are both non-null and `true`.
   * <p>
   * If the original table row count is zero, there is no error, and an empty table is returned.
   * @param mask column of type {@link DType#BOOL8} used as a mask to filter
   *             the input column
   * @return table containing copy of all elements of this table passing
   * the filter defined by the boolean mask
   */
  public Table filter(ColumnVector mask) {
    assert mask.getType() == DType.BOOL8 : "Mask column must be of type BOOL8";
    assert getRowCount() == 0 || getRowCount() == mask.getRowCount() : "Mask column has incorrect size";
    try (DevicePrediction prediction = new DevicePrediction(getDeviceMemorySize(), "filter")) {
      return new Table(filter(nativeHandle, mask.getNativeCudfColumnAddress()));
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // HELPER CLASSES
  /////////////////////////////////////////////////////////////////////////////

  public static final class OrderByArg {
    final int index;
    final boolean isDescending;
    final boolean isNullSmallest;

    OrderByArg(int index, boolean isDescending, boolean isNullSmallest) {
      this.index = index;
      this.isDescending = isDescending;
      this.isNullSmallest = isNullSmallest;
    }
  }

  /**
   * class to encapsulate indices and table
   */
  private final static class Operation {
    final int[] indices;
    final Table table;

    Operation(Table table, int... indices) {
      this.indices = indices;
      this.table = table;
    }
  }

  /**
   * Keeps track of a gdf_column* and operator ids
   */
  private static final class ColumnOp {
    private final long colNativeId;
    private final int opNativeId;

    ColumnOp(long column, int opNativeId){
      this.colNativeId = column;
      this.opNativeId = opNativeId;
    }

    @Override
    public boolean equals(Object other) {
      if (other instanceof ColumnOp){
        ColumnOp otherCop = (ColumnOp) other;
        return otherCop.colNativeId == colNativeId &&
            otherCop.opNativeId == opNativeId;
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(colNativeId, opNativeId);
    }
  }

  /**
   * Class representing aggregate operations
   */
  public static final class AggregateOperation {

    private final Operation operation;
    private final GroupByOptions groupByOptions;

    AggregateOperation(final Table table, GroupByOptions groupByOptions, final int... indices) {
      operation = new Operation(table, indices);
      this.groupByOptions = groupByOptions;
    }

    /**
     * Aggregates the group of columns represented by indices
     * Usage:
     *      aggregate(count(), max(2),...);
     *      example:
     *        input : 1, 1, 1
     *                1, 2, 1
     *                2, 4, 5
     *
     *        table.groupBy(0, 2).count()
     *
     *                col0, col1
     *        output:   1,   1
     *                  1,   2
     *                  2,   1 ==> aggregated count
     * @param aggregates
     * @return
     */
    public Table aggregate(Aggregate... aggregates) {
      throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
      assert aggregates != null && aggregates.length > 0;

      // aggregateColumnIndices: the numeric indices of the columns to aggregate
      // as provided by the user
      List<Integer> aggregateColumnIndices = new ArrayList<>();
      // ops: the corresponding cudf enum values for each aggregate
      // operation, for each column index
      List<Integer> ops = new ArrayList<>();
      // finalAggColumns: this holds the user's desired list of aggregates,
      // as specified in the variadic for this method
      List<ColumnOp> finalAggColumns = new ArrayList<>();
      // aggToCudfColumn: a map of ColumnOp (tuple of gdf_column* and enum for operation)
      // to the index of the column that we asked cudf to compute. 
      // These indices can then be used after the aggregate to find ColumnVector
      // instances, we should place (and ref count) in the order as provided by the user.
      HashMap<ColumnOp, Integer> aggToCudfColumn = new HashMap<>();

      // keep track of the unique agg indices, starting at 1 after the
      // grouping key
      int uniqueAggIndex = operation.indices.length;
      for (int aggregateIndex = 0; aggregateIndex < aggregates.length; aggregateIndex++) {
        Aggregate agg = aggregates[aggregateIndex];
        int origColumnIndex = agg.getIndex();
        int origOpNativeId = agg.getNativeId();

        // keep track of (gdf_column*, op) pairs, as that is what
        // compute_original_requests does to track duplicates
        ColumnOp cop = new ColumnOp(
            operation.table.getColumn(origColumnIndex).getNativeCudfColumnAddress(),
            origOpNativeId);

        // if we haven't seen an aggregate for this (gdf_column *, agg id) pair
        if (!aggToCudfColumn.containsKey(cop)) {
          // add to the aggregates that cudf will perform
          aggregateColumnIndices.add(agg.getIndex());
          ops.add(agg.getNativeId());

          // keep the index of the column where we can find the result later
          aggToCudfColumn.put(cop, uniqueAggIndex++);
        }
        finalAggColumns.add(cop);
      }

      Table aggregate;
      try (DevicePrediction prediction = new DevicePrediction(operation.table.getDeviceMemorySize(), "aggregate")) {
        // aggregate with deduplicated operations
        aggregate = new Table(gdfGroupByAggregate(
            operation.table.nativeHandle,
            operation.indices,
            // one way of converting List[Integer] to int[]
            aggregateColumnIndices.stream().mapToInt(i -> i).toArray(),
            ops.stream().mapToInt(i -> i).toArray(),
            groupByOptions.getIgnoreNullKeys()));
      }

      // prepare the final table
      ColumnVector[] finalCols = new ColumnVector[operation.indices.length + aggregates.length];

      int aggIndex = 0;

      // pick out the grouping columns
      for (int groupIndex : operation.indices) {
        finalCols[groupIndex] = aggregate.getColumn(groupIndex);
        aggIndex++;
      }

      // pick out the aggregate columns (copying the reference for duplicate aggs)
      for (ColumnOp cop : finalAggColumns) {
        int originalIndex = aggToCudfColumn.get(cop);
        finalCols[aggIndex] = aggregate.getColumn(originalIndex);
        aggIndex++;
      }

      // Note: Table will increase ref counts accordingly, which means
      // that duplicate columns in finalCols, will get a refCount equal
      // to the number of times their references appear on the table (good)
      Table tbl = new Table(finalCols);

      // returning a brand new table now, so we close the original
      // this brings the refCount down for each column, such that Table
      // is the only holder
      aggregate.close();

      return tbl;
*/
    }
  }

  public static final class TableOperation {

    private final Operation operation;

    TableOperation(final Table table, final int... indices) {
      operation = new Operation(table, indices);
    }

    /**
     * Joins two tables on the join columns that are passed in.
     * Usage:
     * Table t1 ...
     * Table t2 ...
     * Table result = t1.onColumns(0,1).leftJoin(t2.onColumns(2,3));
     * @param rightJoinIndices - Indices of the right table to join on
     * @return the joined table.  The order of the columns returned will be join columns,
     * left non-join columns, right non-join columns.
     */
    public Table leftJoin(TableOperation rightJoinIndices) {
      throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
      try (DevicePrediction prediction = new DevicePrediction(operation.table.getDeviceMemorySize() +
          rightJoinIndices.operation.table.getDeviceMemorySize(), "leftJoin")) {
        return new Table(gdfLeftJoin(operation.table.nativeHandle, operation.indices,
            rightJoinIndices.operation.table.nativeHandle, rightJoinIndices.operation.indices));
      }
*/
    }

    /**
     * Joins two tables on the join columns that are passed in.
     * Usage:
     * Table t1 ...
     * Table t2 ...
     * Table result = t1.onColumns(0,1).innerJoin(t2.onColumns(2,3));
     * @param rightJoinIndices - Indices of the right table to join on
     * @return the joined table.  The order of the columns returned will be join columns,
     * left non-join columns, right non-join columns.
     */
    public Table innerJoin(TableOperation rightJoinIndices) {
      throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
      try (DevicePrediction prediction = new DevicePrediction(operation.table.getDeviceMemorySize() +
          rightJoinIndices.operation.table.getDeviceMemorySize(), "innerJoin")) {
        return new Table(gdfInnerJoin(operation.table.nativeHandle, operation.indices,
            rightJoinIndices.operation.table.nativeHandle, rightJoinIndices.operation.indices));
*/
      }
    }

    /**
     * Hash partition a table into the specified number of partitions.
     * @param numberOfPartitions - number of partitions to use
     * @return - {@link PartitionedTable} - Table that exposes a limited functionality of the
     * {@link Table} class
     */
    public PartitionedTable partition(int numberOfPartitions) {
      throw new UnsupportedOperationException(STANDARD_CUDF_PORTING_MSG);
/*
      int[] partitionOffsets = new int[numberOfPartitions];
      try (DevicePrediction prediction = new DevicePrediction(operation.table.getDeviceMemorySize(), "partition")) {
        return new PartitionedTable(new Table(gdfPartition(operation.table.nativeHandle,
            operation.indices,
            hashFunction.nativeId,
            partitionOffsets.length,
            partitionOffsets)), partitionOffsets);
      }
    }
*/
  }

  /////////////////////////////////////////////////////////////////////////////
  // BUILDER
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Create a table on the GPU with data from the CPU.  This is not fast and intended mostly for
   * tests.
   */
  public static final class TestBuilder {
    private final List<DType> types = new ArrayList<>();
    private final List<Object> typeErasedData = new ArrayList<>();

    public TestBuilder column(String... values) {
      types.add(DType.STRING);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Boolean... values) {
      types.add(DType.BOOL8);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Byte... values) {
      types.add(DType.INT8);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Short... values) {
      types.add(DType.INT16);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Integer... values) {
      types.add(DType.INT32);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Long... values) {
      types.add(DType.INT64);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Float... values) {
      types.add(DType.FLOAT32);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Double... values) {
      types.add(DType.FLOAT64);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder timestampDayColumn(Long... values) {
      types.add(DType.TIMESTAMP_DAYS);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder timestampNanosecondsColumn(Long... values) {
      types.add(DType.TIMESTAMP_NANOSECONDS);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder timestampMillisecondsColumn(Long... values) {
      types.add(DType.TIMESTAMP_MILLISECONDS);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder timestampMicrosecondsColumn(Long... values) {
      types.add(DType.TIMESTAMP_MICROSECONDS);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder timestampSecondsColumn(Long... values) {
      types.add(DType.TIMESTAMP_SECONDS);
      typeErasedData.add(values);
      return this;
    }

    private static ColumnVector from(DType type, Object dataArray) {
      ColumnVector ret = null;
      switch (type) {
        case STRING:
          ret = ColumnVector.fromStrings((String[]) dataArray);
          break;
        case BOOL8:
          ret = ColumnVector.fromBoxedBooleans((Boolean[]) dataArray);
          break;
        case INT8:
          ret = ColumnVector.fromBoxedBytes((Byte[]) dataArray);
          break;
        case INT16:
          ret = ColumnVector.fromBoxedShorts((Short[]) dataArray);
          break;
        case INT32:
          ret = ColumnVector.fromBoxedInts((Integer[]) dataArray);
          break;
        case INT64:
          ret = ColumnVector.fromBoxedLongs((Long[]) dataArray);
          break;
        case TIMESTAMP_DAYS:
          ret = ColumnVector.timestampDaysFromBoxedInts((Integer[]) dataArray);
          break;
        case TIMESTAMP_SECONDS:
          ret = ColumnVector.timestampSecondsFromBoxedLongs((Long[]) dataArray);
          break;
        case TIMESTAMP_MILLISECONDS:
          ret = ColumnVector.timestampMilliSecondsFromBoxedLongs((Long[]) dataArray);
          break;
        case TIMESTAMP_MICROSECONDS:
          ret = ColumnVector.timestampMicroSecondsFromBoxedLongs((Long[]) dataArray);
          break;
        case TIMESTAMP_NANOSECONDS:
          ret = ColumnVector.timestampNanoSecondsFromBoxedLongs((Long[]) dataArray);
          break;
        case FLOAT32:
          ret = ColumnVector.fromBoxedFloats((Float[]) dataArray);
          break;
        case FLOAT64:
          ret = ColumnVector.fromBoxedDoubles((Double[]) dataArray);
          break;
        default:
          throw new IllegalArgumentException(type + " is not supported yet");
      }
      return ret;
    }

    public Table build() {
      List<ColumnVector> columns = new ArrayList<>(types.size());
      try {
        for (int i = 0; i < types.size(); i++) {
          columns.add(from(types.get(i), typeErasedData.get(i)));
        }
        for (ColumnVector cv : columns) {
          cv.ensureOnDevice();
        }
        return new Table(columns.toArray(new ColumnVector[columns.size()]));
      } finally {
        for (ColumnVector cv : columns) {
          cv.close();
        }
      }
    }
  }
}
