/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstring>
#include <map>

#include <unordered_set>

#include <cudf/table/table.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/io/functions.hpp>
#include <cudf/join.hpp>

#include "cudf/utilities/legacy/nvcategory_util.hpp"
#include "cudf/legacy/copying.hpp"
#include "cudf/legacy/groupby.hpp"
#include "cudf/legacy/io_readers.hpp"
#include "cudf/legacy/table.hpp"
#include "cudf/search.hpp"
#include "cudf/types.hpp"
#include "cudf/legacy/join.hpp"
#include "cudf/column/column.hpp"
#include "cudf/sorting.hpp"
#include "cudf/table/table_view.hpp"

#include "jni_utils.hpp"

namespace cudf {
namespace jni {

/**
 * Copy contents of a jbooleanArray into an array of int8_t pointers
 */
static jni_rmm_unique_ptr<int8_t> copy_to_device(JNIEnv *env, const native_jbooleanArray &n_arr) {
  jsize len = n_arr.size();
  size_t byte_len = len * sizeof(int8_t);
  const jboolean *tmp = n_arr.data();

  std::unique_ptr<int8_t[]> host(new int8_t[byte_len]);

  for (int i = 0; i < len; i++) {
    host[i] = static_cast<int8_t>(n_arr[i]);
  }

  auto device = jni_rmm_alloc<int8_t>(env, byte_len);
  jni_cuda_check(env, cudaMemcpy(device.get(), host.get(), byte_len, cudaMemcpyHostToDevice));
  return device;
}

/**
 * Take a table returned by some operation and turn it into an array of column* so we can track them ourselves
 * in java instead of having their life tied to the table.
 */
static jlongArray convert_table_for_return(JNIEnv * env, std::unique_ptr<cudf::experimental::table> &table_result) {
    std::vector<std::unique_ptr<cudf::column>> ret = table_result->release();
    int num_columns = ret.size();
    cudf::jni::native_jlongArray outcol_handles(env, num_columns);
    for (int i = 0; i < num_columns; i++) {
      outcol_handles[i] = reinterpret_cast<jlong>(ret[i].release());
    }
    return outcol_handles.get_jArray();
}

} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_createCudfTableView(JNIEnv *env, jclass class_object,
                                                                  jlongArray j_cudf_columns) {
  JNI_NULL_CHECK(env, j_cudf_columns, "columns are null", 0);

  try {
      cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_columns(env, j_cudf_columns);

    std::vector<cudf::column_view> column_views(n_cudf_columns.size());
    for (int i = 0 ; i < n_cudf_columns.size() ; i++) {
        column_views[i] = *n_cudf_columns[i];
    }
    cudf::table_view* tv = new cudf::table_view(column_views);
    return reinterpret_cast<jlong>(tv);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_deleteCudfTable(JNIEnv *env, jclass class_object,
                                                               jlong j_cudf_table_view) {
  JNI_NULL_CHECK(env, j_cudf_table_view, "table view handle is null", );
  delete reinterpret_cast<cudf::table_view*>(j_cudf_table_view);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_orderBy(
    JNIEnv *env, jclass j_class_object, jlong j_input_table, jlongArray j_sort_keys_columns,
    jbooleanArray j_is_descending, jbooleanArray j_are_nulls_smallest) {

  // input validations & verifications
  JNI_NULL_CHECK(env, j_input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_sort_keys_columns, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_is_descending, "sort order array is null", NULL);
  JNI_NULL_CHECK(env, j_are_nulls_smallest, "null order array is null", NULL);

  try {
    cudf::jni::native_jpointerArray<cudf::column_view> n_sort_keys_columns(env, j_sort_keys_columns);
    jsize num_columns = n_sort_keys_columns.size();
    const cudf::jni::native_jbooleanArray n_is_descending(env, j_is_descending);
    jsize num_columns_is_desc = n_is_descending.size();

    if (num_columns_is_desc != num_columns) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                    "columns and is_descending lengths don't match", NULL);
    }

    const cudf::jni::native_jbooleanArray n_are_nulls_smallest(env, j_are_nulls_smallest);
    jsize num_columns_null_smallest = n_are_nulls_smallest.size();

    if (num_columns_null_smallest != num_columns) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                    "columns and areNullsSmallest lengths don't match", NULL);
    }

    std::vector<cudf::order> order(n_is_descending.size());
    for (int i = 0; i < n_is_descending.size(); i++) {
      order[i] = n_is_descending[i] ? cudf::order::DESCENDING : cudf::order::ASCENDING;
    }
    std::vector<cudf::null_order> null_order(n_are_nulls_smallest.size());
    for (int i = 0; i < n_are_nulls_smallest.size(); i++) {
      null_order[i] = n_are_nulls_smallest[i] ? cudf::null_order::BEFORE : cudf::null_order::AFTER;
    }

    std::vector<cudf::column_view> columns;
    columns.reserve(num_columns);
    for (int i = 0; i < num_columns; i++) {
      columns.push_back(*n_sort_keys_columns[i]);
    }
    cudf::table_view keys(columns);

    auto sorted_col = cudf::experimental::sorted_order(keys, order, null_order);

    cudf::table_view *input_table = reinterpret_cast<cudf::table_view *>(j_input_table);
    std::unique_ptr<cudf::experimental::table> result = cudf::experimental::gather(*input_table,
            sorted_col->view());
    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_readCSV(
    JNIEnv *env, jclass j_class_object, jobjectArray col_names, jobjectArray data_types,
    jobjectArray filter_col_names, jstring inputfilepath, jlong buffer, jlong buffer_length,
    jint header_row, jbyte delim, jbyte quote, jbyte comment, jobjectArray null_values,
    jobjectArray true_values, jobjectArray false_values) {
  JNI_NULL_CHECK(env, null_values, "null_values must be supplied, even if it is empty", NULL);

  bool read_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
    read_buffer = false;
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "cannot pass in both a buffer and an inputfilepath", NULL);
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported",
                  NULL);
  }

  try {
    cudf::jni::native_jstringArray n_col_names(env, col_names);
    cudf::jni::native_jstringArray n_data_types(env, data_types);

    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inputfilepath can't be empty",
                    NULL);
    }

    cudf::jni::native_jstringArray n_null_values(env, null_values);
    cudf::jni::native_jstringArray n_true_values(env, true_values);
    cudf::jni::native_jstringArray n_false_values(env, false_values);
    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    std::unique_ptr<cudf::experimental::io::source_info> source;
    if (read_buffer) {
      source.reset(new cudf::experimental::io::source_info(reinterpret_cast<char *>(buffer), buffer_length));
    } else {
      source.reset(new cudf::experimental::io::source_info(filename.get()));
    }

    cudf::experimental::io::read_csv_args read_arg{*source};
    read_arg.lineterminator = '\n';
    // delimiter ideally passed in
    read_arg.delimiter = delim;
    read_arg.delim_whitespace = 0;
    read_arg.skipinitialspace = 0;
    read_arg.header = header_row;

    read_arg.names = n_col_names.as_cpp_vector();
    read_arg.dtype = n_data_types.as_cpp_vector();

    read_arg.use_cols_names = n_filter_col_names.as_cpp_vector();

    read_arg.skip_blank_lines = true;

    read_arg.true_values = n_true_values.as_cpp_vector();
    read_arg.false_values = n_false_values.as_cpp_vector();

    read_arg.na_values = n_null_values.as_cpp_vector();
    read_arg.keep_default_na = false; ///< Keep the default NA values
    read_arg.na_filter = n_null_values.size() > 0;

    read_arg.mangle_dupe_cols = true;
    read_arg.dayfirst = 0;
    read_arg.compression = cudf::experimental::io::compression_type::AUTO;
    read_arg.decimal = '.';
    read_arg.quotechar = quote;
    read_arg.quoting = cudf::experimental::io::quote_style::MINIMAL;
    read_arg.doublequote = true;
    read_arg.comment = comment;

    cudf::experimental::io::table_with_metadata result = cudf::experimental::io::read_csv(read_arg);
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_readParquet(
    JNIEnv *env, jclass j_class_object, jobjectArray filter_col_names, jstring inputfilepath,
    jlong buffer, jlong buffer_length, jint unit) {
  bool read_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
    read_buffer = false;
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "cannot pass in both a buffer and an inputfilepath", NULL);
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported",
                  NULL);
  }

  try {
    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inputfilepath can't be empty",
                    NULL);
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    std::unique_ptr<cudf::experimental::io::source_info> source;
    if (read_buffer) {
      source.reset(new cudf::experimental::io::source_info(reinterpret_cast<char *>(buffer), buffer_length));
    } else {
      source.reset(new cudf::experimental::io::source_info(filename.get()));
    }

    cudf::experimental::io::read_parquet_args read_arg(*source);

    read_arg.columns = n_filter_col_names.as_cpp_vector();

    read_arg.row_group = -1;
    read_arg.skip_rows = -1;
    read_arg.num_rows = -1;
    read_arg.strings_to_categorical = false;
    read_arg.timestamp_type = cudf::data_type(static_cast<cudf::type_id>(unit));

    cudf::experimental::io::table_with_metadata result = cudf::experimental::io::read_parquet(read_arg);
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeParquet(JNIEnv* env, jclass,
    jlong j_table,
    jobjectArray j_col_names,
    jobjectArray j_metadata_keys,
    jobjectArray j_metadata_values,
    jint j_compression,
    jint j_stats_freq,
    jstring j_output_path) {
  JNI_NULL_CHECK(env, j_table, "null table", );
  JNI_NULL_CHECK(env, j_col_names, "null columns", );
  JNI_NULL_CHECK(env, j_metadata_keys, "null metadata keys", );
  JNI_NULL_CHECK(env, j_metadata_values, "null metadata values", );
  try {
    using namespace cudf::experimental::io;
    cudf::jni::native_jstringArray col_names(env, j_col_names);
    cudf::jni::native_jstringArray meta_keys(env, j_metadata_keys);
    cudf::jni::native_jstringArray meta_values(env, j_metadata_values);
    cudf::jni::native_jstring output_path(env, j_output_path);

    table_metadata metadata{col_names.as_cpp_vector()};
    for (size_t i = 0; i < meta_keys.size(); ++i) {
      metadata.user_data[meta_keys[i].get()] = meta_values[i].get();
    }

    sink_info sink{output_path.get()};
    compression_type compression{static_cast<compression_type>(j_compression)};
    statistics_freq stats{static_cast<statistics_freq>(j_stats_freq)};

    cudf::table_view *tview = reinterpret_cast<cudf::table_view *>(j_table);
    write_parquet_args args(sink, *tview, &metadata, compression, stats);
    write_parquet(args);
  } CATCH_STD(env, )
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_readORC(
    JNIEnv *env, jclass j_class_object, jobjectArray filter_col_names, jstring inputfilepath,
    jlong buffer, jlong buffer_length, jboolean usingNumPyTypes, jint unit) {
  bool read_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
    read_buffer = false;
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "cannot pass in both a buffer and an inputfilepath", NULL);
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported",
                  NULL);
  }

  try {
    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inputfilepath can't be empty",
                    NULL);
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    std::unique_ptr<cudf::experimental::io::source_info> source;
    if (read_buffer) {
      source.reset(new cudf::experimental::io::source_info(reinterpret_cast<char *>(buffer), buffer_length));
    } else {
      source.reset(new cudf::experimental::io::source_info(filename.get()));
    }

    cudf::experimental::io::read_orc_args read_arg{*source};
    read_arg.columns = n_filter_col_names.as_cpp_vector();
    read_arg.stripe = -1;
    read_arg.skip_rows = -1;
    read_arg.num_rows = -1;
    read_arg.use_index = false;
    read_arg.use_np_dtypes = static_cast<bool>(usingNumPyTypes);
    read_arg.timestamp_type = cudf::data_type(static_cast<cudf::type_id>(unit));

    cudf::experimental::io::table_with_metadata result = cudf::experimental::io::read_orc(read_arg);
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeORC(JNIEnv *env, jclass,
                                                              jint j_compression_type,
                                                              jobjectArray j_col_names,
                                                              jobjectArray j_metadata_keys,
                                                              jobjectArray j_metadata_values,
                                                              jstring outputfilepath, jlong buffer,
                                                              jlong buffer_length, jlong j_table_view) {
  bool write_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, outputfilepath, "output file or buffer must be supplied", );
    write_buffer = false;
  } else if (outputfilepath != NULL) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "cannot pass in both a buffer and an outputfilepath", );
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported", );
  }
  JNI_NULL_CHECK(env, j_col_names, "null columns", );
  JNI_NULL_CHECK(env, j_metadata_keys, "null metadata keys", );
  JNI_NULL_CHECK(env, j_metadata_values, "null metadata values", );

  try {
    cudf::jni::native_jstring filename(env, outputfilepath);
    cudf::jni::native_jstringArray meta_keys(env, j_metadata_keys);
    cudf::jni::native_jstringArray meta_values(env, j_metadata_values);
    cudf::jni::native_jstringArray col_names(env, j_col_names);
    namespace orc = cudf::experimental::io;
    if (write_buffer) {
      JNI_THROW_NEW(env, "java/lang/UnsupportedOperationException",
                        "buffers are not supported", );
    } else {
      orc::sink_info info(filename.get());
      orc::table_metadata metadata{col_names.as_cpp_vector()};
      for (size_t i = 0; i < meta_keys.size(); ++i) {
        metadata.user_data[meta_keys[i].get()] = meta_values[i].get();
      }
      orc::write_orc_args args(info, *reinterpret_cast<cudf::table_view*>(j_table_view), &metadata,
                                                static_cast<orc::compression_type>(j_compression_type));
      orc::write_orc(args);
    }
  }
  CATCH_STD(env, );
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftJoin(
    JNIEnv *env, jclass clazz, jlong left_table, jintArray left_col_join_indices, jlong right_table,
    jintArray right_col_join_indices) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, left_col_join_indices, "left_col_join_indices is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, right_col_join_indices, "right_col_join_indices is null", NULL);

  try {
    cudf::table_view *n_left_table = reinterpret_cast<cudf::table_view *>(left_table);
    cudf::table_view *n_right_table = reinterpret_cast<cudf::table_view *>(right_table);
    cudf::jni::native_jintArray left_join_cols_arr(env, left_col_join_indices);
    std::vector<cudf::size_type> left_join_cols(left_join_cols_arr.data(), left_join_cols_arr.data() + left_join_cols_arr.size());
    cudf::jni::native_jintArray right_join_cols_arr(env, right_col_join_indices);
    std::vector<cudf::size_type> right_join_cols(right_join_cols_arr.data(), right_join_cols_arr.data() + right_join_cols_arr.size());

    int dedupe_size = left_join_cols.size();
    std::vector<std::pair<cudf::size_type, cudf::size_type>> dedupe(dedupe_size);
    for (int i = 0; i < dedupe_size; i++) {
      dedupe[i].first = left_join_cols[i];
      dedupe[i].second = right_join_cols[i];
    }

    std::unique_ptr<cudf::experimental::table> result = cudf::experimental::left_join(
            *n_left_table, *n_right_table,
            left_join_cols, right_join_cols,
            dedupe);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_innerJoin(
    JNIEnv *env, jclass clazz, jlong left_table, jintArray left_col_join_indices, jlong right_table,
    jintArray right_col_join_indices) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, left_col_join_indices, "left_col_join_indices is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, right_col_join_indices, "right_col_join_indices is null", NULL);

  try {
    cudf::table_view *n_left_table = reinterpret_cast<cudf::table_view *>(left_table);
    cudf::table_view *n_right_table = reinterpret_cast<cudf::table_view *>(right_table);
    cudf::jni::native_jintArray left_join_cols_arr(env, left_col_join_indices);
    std::vector<cudf::size_type> left_join_cols(left_join_cols_arr.data(), left_join_cols_arr.data() + left_join_cols_arr.size());
    cudf::jni::native_jintArray right_join_cols_arr(env, right_col_join_indices);
    std::vector<cudf::size_type> right_join_cols(right_join_cols_arr.data(), right_join_cols_arr.data() + right_join_cols_arr.size());

    int dedupe_size = left_join_cols.size();
    std::vector<std::pair<cudf::size_type, cudf::size_type>> dedupe(dedupe_size);
    for (int i = 0; i < dedupe_size; i++) {
      dedupe[i].first = left_join_cols[i];
      dedupe[i].second = right_join_cols[i];
    }

    std::unique_ptr<cudf::experimental::table> result = cudf::experimental::inner_join(
            *n_left_table, *n_right_table,
            left_join_cols, right_join_cols,
            dedupe);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_concatenate(JNIEnv *env, jclass clazz,
                                                                   jlongArray table_handles) {
  JNI_NULL_CHECK(env, table_handles, "input tables are null", NULL);
  try {
    cudf::jni::native_jpointerArray<cudf::table_view> tables(env, table_handles);

    long unsigned int num_tables = tables.size();
    // There are some issues with table_view and std::vector. We cannot give the
    // vector a size or it will not compile.
    std::vector<cudf::table_view> to_concat;
    to_concat.reserve(num_tables);
    for (int i = 0; i < num_tables; i++) {
      JNI_NULL_CHECK(env, tables[i], "input table included a null", NULL);
      to_concat.push_back(*tables[i]);
    }
    std::unique_ptr<cudf::experimental::table> table_result = cudf::experimental::concatenate(to_concat);
    return cudf::jni::convert_table_for_return(env, table_result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfPartition(
    JNIEnv *env, jclass clazz, jlong input_table, jintArray columns_to_hash,
    jint cudf_hash_function, jint number_of_partitions, jintArray output_offsets) {

  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, columns_to_hash, "columns_to_hash is null", NULL);
  JNI_NULL_CHECK(env, output_offsets, "output_offsets is null", NULL);
  JNI_ARG_CHECK(env, number_of_partitions > 0, "number_of_partitions is zero", NULL);

  try {
    cudf::table *n_input_table = reinterpret_cast<cudf::table *>(input_table);
    cudf::jni::native_jintArray n_columns_to_hash(env, columns_to_hash);
    gdf_hash_func n_cudf_hash_function = static_cast<gdf_hash_func>(cudf_hash_function);
    int n_number_of_partitions = static_cast<int>(number_of_partitions);
    cudf::jni::native_jintArray n_output_offsets(env, output_offsets);

    JNI_ARG_CHECK(env, n_columns_to_hash.size() > 0, "columns_to_hash is zero", NULL);

    cudf::jni::output_table output(env, n_input_table, true);
    std::vector<gdf_column *> cols = output.get_gdf_columns();

    for (int i = 0; i < cols.size(); i++) {
      gdf_column * col = cols[i];
      if (col->dtype == GDF_STRING_CATEGORY) {
        // We need to add in the category for partition to work at all...
        NVCategory * orig = static_cast<NVCategory *>(n_input_table->get_column(i)->dtype_info.category);
        col->dtype_info.category = orig;
      }
    }

    JNI_GDF_TRY(env, NULL,
                gdf_hash_partition(n_input_table->num_columns(), n_input_table->begin(),
                                   n_columns_to_hash.data(), n_columns_to_hash.size(),
                                   n_number_of_partitions, cols.data(), n_output_offsets.data(),
                                   n_cudf_hash_function));

    // Need to gather the string categories after partitioning.
    for (int i = 0; i < cols.size(); i++) {
      gdf_column * col = cols[i];
      if (col->dtype == GDF_STRING_CATEGORY) {
        // We need to fix it up...
        NVCategory * orig = static_cast<NVCategory *>(n_input_table->get_column(i)->dtype_info.category);
        nvcategory_gather(col, orig);
      }
    }

    return output.get_native_handles_and_release();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfGroupByAggregate(
    JNIEnv *env, jclass clazz, jlong input_table, jintArray keys,
    jintArray aggregate_column_indices, jintArray agg_types, jboolean ignore_null_keys) {
  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, keys, "input keys are null", NULL);
  JNI_NULL_CHECK(env, aggregate_column_indices, "input aggregate_column_indices are null", NULL);
  JNI_NULL_CHECK(env, agg_types, "agg_types are null", NULL);

  try {
    cudf::table *n_input_table = reinterpret_cast<cudf::table *>(input_table);
    cudf::jni::native_jintArray n_keys(env, keys);
    cudf::jni::native_jintArray n_values(env, aggregate_column_indices);
    cudf::jni::native_jintArray n_ops(env, agg_types);
    std::vector<gdf_column *> n_keys_cols;
    std::vector<gdf_column *> n_values_cols;

    for (int i = 0; i < n_keys.size(); i++) {
      n_keys_cols.push_back(n_input_table->get_column(n_keys[i]));
    }

    for (int i = 0; i < n_values.size(); i++) {
      n_values_cols.push_back(n_input_table->get_column(n_values[i]));
    }

    cudf::table const n_keys_table(n_keys_cols);
    cudf::table const n_values_table(n_values_cols);

    std::vector<cudf::groupby::operators> ops;
    for (int i = 0; i < n_ops.size(); i++) {
      ops.push_back(static_cast<cudf::groupby::operators>(n_ops[i]));
    }

    std::pair<cudf::table, cudf::table> result = cudf::groupby::hash::groupby(
        n_keys_table, n_values_table, ops, cudf::groupby::hash::Options(ignore_null_keys));

    try {
      std::vector<gdf_column *> output_columns;
      output_columns.reserve(result.first.num_columns() + result.second.num_columns());
      output_columns.insert(output_columns.end(), result.first.begin(), result.first.end());
      output_columns.insert(output_columns.end(), result.second.begin(), result.second.end());
      cudf::jni::native_jlongArray native_handles(
          env, reinterpret_cast<jlong *>(output_columns.data()), output_columns.size());
      return native_handles.get_jArray();
    } catch (...) {
      result.first.destroy();
      result.second.destroy();
      throw;
    }
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_filter(JNIEnv *env, jclass,
                                                              jlong input_jtable,
                                                              jlong mask_jcol) {
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_NULL_CHECK(env, mask_jcol, "mask column is null", 0);
  try {
    cudf::table_view *input = reinterpret_cast<cudf::table_view *>(input_jtable);
    cudf::column_view *mask = reinterpret_cast<cudf::column_view *>(mask_jcol);
    std::unique_ptr<cudf::experimental::table> result = cudf::experimental::apply_boolean_mask(*input, *mask);
    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_bound(JNIEnv *env, jclass,
    jlong input_jtable, jlong values_jtable, jbooleanArray desc_flags, jbooleanArray are_nulls_smallest,
    jboolean is_upper_bound) {
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_NULL_CHECK(env, values_jtable, "values table is null", 0);
  using cudf::table_view;
  using cudf::column;
  try {
    table_view *input = reinterpret_cast<table_view *>(input_jtable);
    table_view *values = reinterpret_cast<table_view *>(values_jtable);
    cudf::jni::native_jbooleanArray const n_desc_flags(env, desc_flags);
    cudf::jni::native_jbooleanArray const n_are_nulls_smallest(env, are_nulls_smallest);

    std::vector<cudf::order> column_desc_flags(n_desc_flags.size());
    std::vector<cudf::null_order> column_null_orders(n_are_nulls_smallest.size());

    JNI_ARG_CHECK(env, (column_desc_flags.size() == column_null_orders.size()), "null-order and sort-order size mismatch", 0);
    uint32_t num_columns = column_null_orders.size();
    for (int i = 0 ; i < num_columns ; i++) {
      column_desc_flags[i] = n_desc_flags[i] ? cudf::order::DESCENDING : cudf::order::ASCENDING;
      column_null_orders[i] = n_are_nulls_smallest[i] ? cudf::null_order::BEFORE: cudf::null_order::AFTER;
    }

    std::unique_ptr<column> result;
    if (is_upper_bound) {
      result = std::move(cudf::experimental::upper_bound(*input, *values, column_desc_flags, column_null_orders));
    } else {
      result = std::move(cudf::experimental::lower_bound(*input, *values, column_desc_flags, column_null_orders));
    }
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jobjectArray JNICALL Java_ai_rapids_cudf_Table_contiguousSplit(JNIEnv *env, jclass clazz,
                                                             jlong input_table,
                                                             jintArray split_indices) {
  JNI_NULL_CHECK(env, input_table, "native handle is null", 0);
  JNI_NULL_CHECK(env, split_indices, "split indices are null", 0);

  try {
    cudf::table_view *n_table = reinterpret_cast<cudf::table_view *>(input_table);
    cudf::jni::native_jintArray n_split_indices(env, split_indices);

    std::vector<cudf::size_type> indices(n_split_indices.data(), n_split_indices.data() + n_split_indices.size());

    std::vector<cudf::experimental::contiguous_split_result> result = 
        cudf::experimental::contiguous_split(*n_table, indices);
    cudf::jni::native_jobjectArray<jobject> n_result = 
        cudf::jni::contiguous_table_array(env, result.size());
    for (int i = 0; i < result.size(); i++) {
      n_result.set(i, cudf::jni::contiguous_table_from(env, result[i]));
    }
    return n_result.wrapped();
  }
  CATCH_STD(env, NULL);
}

} // extern "C"
