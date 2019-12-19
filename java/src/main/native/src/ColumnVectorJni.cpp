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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/datetime.hpp>
#include <cudf/filling.hpp>
#include <cudf/replace.hpp>
#include <cudf/search.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>

#include "cudf/legacy/copying.hpp"
#include "cudf/legacy/quantiles.hpp"
#include "cudf/legacy/replace.hpp"
#include "cudf/legacy/rolling.hpp"

#include "jni_utils.hpp"

using unique_nvcat_ptr = std::unique_ptr<NVCategory, decltype(&NVCategory::destroy)>;
using unique_nvstr_ptr = std::unique_ptr<NVStrings, decltype(&NVStrings::destroy)>;

namespace cudf {
namespace jni {
static jlongArray put_strings_on_host(JNIEnv *env, NVStrings *nvstr) {
  cudf::jni::native_jlongArray ret(env, 4);
  unsigned int numstrs = nvstr->size();
  size_t strdata_size = nvstr->memsize();
  size_t offset_size = sizeof(int) * (numstrs + 1);
  std::unique_ptr<char, decltype(free) *> strdata(
      static_cast<char *>(malloc(sizeof(char) * strdata_size)), free);
  std::unique_ptr<int, decltype(free) *> offsetdata(
      static_cast<int *>(malloc(sizeof(int) * (numstrs + 1))), free);
  nvstr->create_offsets(strdata.get(), offsetdata.get(), nullptr, false);
  ret[0] = reinterpret_cast<jlong>(strdata.get());
  ret[1] = strdata_size;
  ret[2] = reinterpret_cast<jlong>(offsetdata.get());
  ret[3] = offset_size;
  strdata.release();
  offsetdata.release();
  return ret.get_jArray();
}
} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_allocateCudfColumn(JNIEnv *env,
                                                                            jobject j_object) {
  try {
    return reinterpret_cast<jlong>(calloc(1, sizeof(gdf_column)));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnVector_freeCudfColumn(JNIEnv *env,
                                                                       jobject j_object,
                                                                       jlong handle,
                                                                       jboolean deep_clean) {
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  if (column != NULL) {
    if (deep_clean) {
      gdf_column_free(column);
    } else if (column->dtype == GDF_STRING) {
      NVStrings::destroy(static_cast<NVStrings *>(column->data));
    } else if (column->dtype == GDF_STRING_CATEGORY) {
      NVCategory::destroy(static_cast<NVCategory *>(column->dtype_info.category));
    }
    free(column->col_name);
  }
  free(column);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_getDataPtr(JNIEnv *env, jobject j_object,
                                                                    jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  return reinterpret_cast<jlong>(column->data);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_getValidPtr(JNIEnv *env, jobject j_object,
                                                                     jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  return reinterpret_cast<jlong>(column->valid);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getRowCount(JNIEnv *env, jobject j_object,
                                                                    jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  return static_cast<jint>(column->size);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getNullCount(JNIEnv *env, jobject j_object,
                                                                     jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  return column->null_count;
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getDTypeInternal(JNIEnv *env,
                                                                         jobject j_object,
                                                                         jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  return column->dtype;
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_ColumnVector_getTimeUnitInternal(JNIEnv *env,
                                                                            jobject j_object,
                                                                            jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  return column->dtype_info.time_unit;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnVector_cudfColumnViewAugmented(
    JNIEnv *env, jobject, jlong handle, jlong data_ptr, jlong j_valid, jint size, jint dtype,
    jint null_count, jint time_unit) {
  JNI_NULL_CHECK(env, handle, "column is null", );
  gdf_column *column = reinterpret_cast<gdf_column *>(handle);
  void *data = reinterpret_cast<void *>(data_ptr);
  cudf::valid_type *valid = reinterpret_cast<cudf::valid_type *>(j_valid);
  gdf_dtype c_dtype = static_cast<gdf_dtype>(dtype);
  gdf_dtype_extra_info info{};
  info.time_unit = static_cast<gdf_time_unit>(time_unit);
  JNI_GDF_TRY(env, ,
              gdf_column_view_augmented(column, data, valid, size, c_dtype, null_count, info));
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ColumnVector_cudfColumnViewStrings(
    JNIEnv *env, jobject, jlong handle, jlong data_ptr, jboolean data_ptr_on_host,
    jlong host_offsets_ptr, jboolean reset_offsets_to_zero, jlong device_valid_ptr,
    jlong device_output_data_ptr, jint size, jint jdtype, jint null_count) {
  JNI_NULL_CHECK(env, handle, "column is null", );
  JNI_NULL_CHECK(env, data_ptr, "string data is null", );
  JNI_NULL_CHECK(env, host_offsets_ptr, "host offsets is null", );

  try {
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    char *data = reinterpret_cast<char *>(data_ptr);
    uint32_t *host_offsets = reinterpret_cast<uint32_t *>(host_offsets_ptr);

    uint32_t data_size = host_offsets[size];
    if (reset_offsets_to_zero) {
      data_size -= host_offsets[0];
    }

    cudf::valid_type *valid = reinterpret_cast<cudf::valid_type *>(device_valid_ptr);
    gdf_dtype dtype = static_cast<gdf_dtype>(jdtype);
    gdf_dtype_extra_info info{};

    // NOTE: Even though the caller API is tailor-made to use
    // NVCategory::create_from_offsets or NVStrings::create_from_offsets, it's much faster to
    // use create_from_index, block-transferring the host string data to the device first.

    char *device_data = nullptr;
    cudf::jni::jni_rmm_unique_ptr<char> dev_string_data_holder;
    if (data_ptr_on_host) {
      dev_string_data_holder = cudf::jni::jni_rmm_alloc<char>(env, data_size);
      JNI_CUDA_TRY(
          env, ,
          cudaMemcpyAsync(dev_string_data_holder.get(), data, data_size, cudaMemcpyHostToDevice));
      device_data = dev_string_data_holder.get();
    } else {
      device_data = data;
    }

    uint32_t offset_amount_to_subtract = 0;
    if (reset_offsets_to_zero) {
      offset_amount_to_subtract = host_offsets[0];
    }
    std::vector<std::pair<const char *, size_t>> index{};
    index.reserve(size);
    for (int i = 0; i < size; i++) {
      index[i].first = device_data + host_offsets[i] - offset_amount_to_subtract;
      index[i].second = host_offsets[i + 1] - host_offsets[i];
    }

    if (dtype == GDF_STRING) {
      unique_nvstr_ptr strings(NVStrings::create_from_index(index.data(), size, false),
                               &NVStrings::destroy);
      JNI_GDF_TRY(
          env, ,
          gdf_column_view_augmented(column, strings.get(), valid, size, dtype, null_count, info));
      strings.release();
    } else if (dtype == GDF_STRING_CATEGORY) {
      JNI_NULL_CHECK(env, device_output_data_ptr, "device data pointer is null", );
      int *cat_data = reinterpret_cast<int *>(device_output_data_ptr);
      unique_nvcat_ptr cat(NVCategory::create_from_index(index.data(), size, false),
                           &NVCategory::destroy);
      info.category = cat.get();
      if (size != cat->get_values(cat_data, true)) {
        JNI_THROW_NEW(env, "java/lang/IllegalStateException",
                      "Internal Error copying str cat data", );
      }
      JNI_GDF_TRY(
          env, , gdf_column_view_augmented(column, cat_data, valid, size, dtype, null_count, info));
      cat.release();
    } else {
      throw std::logic_error("ONLY STRING TYPES ARE SUPPORTED...");
    }
  }
  CATCH_STD(env, );
}

// Resolve the mutated dictionary with the original index values
// gathering column metadata from the most relevant sources
cudf::jni::gdf_column_wrapper gather_mutated_category(gdf_column *dict_result, gdf_column *column) {
  std::vector<gdf_column*> vec {dict_result};
  cudf::table tmp_table(vec);

  cudf::jni::gdf_column_wrapper result(column->size, dict_result->dtype, column->null_count != 0);
  gdf_column * result_ptr = result.get();
  std::vector<gdf_column*> out_vec {result_ptr};
  cudf::table output_table(out_vec);

  gather(&tmp_table, static_cast<cudf::size_type *>(column->data), &output_table);
  if (column->null_count > 0) {
    CUDA_TRY(cudaMemcpy(result_ptr->valid, column->valid,
                        gdf_num_bitmask_elements(column->size), cudaMemcpyDeviceToDevice));
    result_ptr->null_count = column->null_count;
  }
  result_ptr->dtype_info.time_unit = dict_result->dtype_info.time_unit;

  return result;
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_upperStrings(JNIEnv *env, jobject j_object,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);

  try {
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::strings_column_view strings_column(*column);

    std::unique_ptr<cudf::column> result = cudf::strings::to_upper(strings_column);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_lowerStrings(JNIEnv *env, jobject j_object,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);

  try {
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::strings_column_view strings_column(*column);

    std::unique_ptr<cudf::column> result = cudf::strings::to_lower(strings_column);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_ColumnVector_getStringDataAndOffsetsBack(JNIEnv *env, jobject, jlong handle) {
  JNI_NULL_CHECK(env, handle, "column is null", NULL);

  try {
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    gdf_dtype dtype = column->dtype;
    // data address, data length, offsets address, offsets length
    if (dtype == GDF_STRING) {
      return cudf::jni::put_strings_on_host(env, static_cast<NVStrings *>(column->data));
    } else if (dtype == GDF_STRING_CATEGORY) {
      NVCategory *cat = static_cast<NVCategory *>(column->dtype_info.category);
      unique_nvstr_ptr nvstr(cat->to_strings(), &NVStrings::destroy);
      return cudf::jni::put_strings_on_host(env, nvstr.get());
    } else {
      throw std::logic_error("ONLY STRING TYPES ARE SUPPORTED...");
    }
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_concatenate(JNIEnv *env, jclass clazz,
                                                                     jlongArray column_handles) {
  JNI_NULL_CHECK(env, column_handles, "input columns are null", 0);
  using cudf::column;
  using cudf::column_view;
  try {
    cudf::jni::native_jpointerArray<column_view> columns(env, column_handles);
    std::vector<column_view> columns_vector(columns.size());
    for (int i = 0; i < columns.size(); ++i) {
      JNI_NULL_CHECK(env, columns[i], "column to concat is null", 0);
      columns_vector[i] = *columns[i];
    }
    std::unique_ptr<column> result = cudf::concatenate(columns_vector);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_fromScalar(JNIEnv *env, jclass,
    jlong j_scalar, jint row_count) {
  JNI_NULL_CHECK(env, j_scalar, "scalar is null", 0);
  try {
    auto scalar_val = reinterpret_cast<cudf::scalar const*>(j_scalar);
    auto dtype = scalar_val->type();
    cudf::mask_state mask_state = scalar_val->is_valid() ? cudf::mask_state::UNALLOCATED : cudf::mask_state::ALL_NULL;
    std::unique_ptr<cudf::column> col;
    if (row_count == 0) {
      col = cudf::make_empty_column(dtype);
    } else if (cudf::is_fixed_width(dtype)) {
      col = cudf::make_fixed_width_column(dtype, row_count, mask_state);
      auto mut_view = col->mutable_view();
      cudf::experimental::fill(mut_view, 0, row_count, *scalar_val);
    } else if (dtype.id() == cudf::type_id::STRING) {
      // create a string column of all empty strings to fill (cheapest string column to create)
      auto offsets = cudf::make_numeric_column(cudf::data_type{cudf::INT32}, row_count + 1, cudf::mask_state::UNALLOCATED);
      auto data = cudf::make_empty_column(cudf::data_type{cudf::INT8});
      auto mask_buffer = cudf::create_null_mask(row_count, cudf::UNALLOCATED);
      auto str_col = cudf::make_strings_column(row_count, std::move(offsets), std::move(data), 0, std::move(mask_buffer));

      col = cudf::experimental::fill(str_col->view(), 0, row_count, *scalar_val);
    } else {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "Invalid data type", 0);
    }
    return reinterpret_cast<jlong>(col.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_replaceNulls(JNIEnv *env, jclass,
    jlong j_col, jlong j_scalar) {
  JNI_NULL_CHECK(env, j_col, "column is null", 0);
  JNI_NULL_CHECK(env, j_scalar, "scalar is null", 0);
  try {
    cudf::column_view col = *reinterpret_cast<cudf::column_view*>(j_col);
    auto val = reinterpret_cast<cudf::scalar*>(j_scalar);
    std::unique_ptr<cudf::column> result = cudf::experimental::replace_nulls(col, *val);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_ColumnVector_exactQuantile(JNIEnv *env, jclass clazz,
                                                                         jlong input_column,
                                                                         jint quantile_method,
                                                                         jdouble quantile) {
  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  try {
    gdf_column *n_input_column = reinterpret_cast<gdf_column *>(input_column);
    cudf::interpolation n_quantile_method = static_cast<cudf::interpolation>(quantile_method);
    gdf_context ctxt{0, GDF_SORT, 0, 0};
    gdf_scalar result{};
    JNI_GDF_TRY(env, NULL,
                cudf::quantile_exact(n_input_column, n_quantile_method, quantile, &result, &ctxt));
    //return cudf::jni::jscalar_from_scalar(env, result, n_input_column->dtype_info.time_unit);
    throw std::logic_error("BAD IMPLEMENTATION");
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_ColumnVector_approxQuantile(JNIEnv *env, jclass clazz,
                                                                          jlong input_column,
                                                                          jdouble quantile) {
  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  try {
    gdf_column *n_input_column = reinterpret_cast<gdf_column *>(input_column);
    gdf_context ctxt{0, GDF_SORT, 0, 0};
    gdf_scalar result{};
    JNI_GDF_TRY(env, NULL, cudf::quantile_approx(n_input_column, quantile, &result, &ctxt));
    //return cudf::jni::jscalar_from_scalar(env, result, n_input_column->dtype_info.time_unit);
    throw std::logic_error("BAD IMPLEMENTATION");
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_rollingWindow(
    JNIEnv *env, jclass clazz, jlong input_column, jint window, jint min_periods,
    jint forward_window, jint agg_type, jlong window_col, jlong min_periods_col,
    jlong forward_window_col) {

  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  try {
    gdf_column *n_input_column = reinterpret_cast<gdf_column *>(input_column);
    gdf_column *n_window_col = reinterpret_cast<gdf_column *>(window_col);
    gdf_column *n_min_periods_col = reinterpret_cast<gdf_column *>(min_periods_col);
    gdf_column *n_forward_window_col = reinterpret_cast<gdf_column *>(forward_window_col);

    gdf_column *result = cudf::rolling_window(
        *n_input_column, static_cast<cudf::size_type>(window),
        static_cast<cudf::size_type>(min_periods), static_cast<cudf::size_type>(forward_window),
        static_cast<gdf_agg_op>(agg_type),
        n_window_col == nullptr ? nullptr : reinterpret_cast<cudf::size_type *>(n_window_col->data),
        n_min_periods_col == nullptr ? nullptr :
                                       reinterpret_cast<cudf::size_type *>(n_min_periods_col->data),
        n_forward_window_col == nullptr ?
            nullptr :
            reinterpret_cast<cudf::size_type *>(n_forward_window_col->data));
    return reinterpret_cast<jlong>(result);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnVector_cudfSlice(JNIEnv *env, jclass clazz,
                                                                        jlong input_column,
                                                                        jlong slice_indices) {
  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  JNI_NULL_CHECK(env, slice_indices, "slice indices are null", 0);

  gdf_column *n_column = reinterpret_cast<gdf_column *>(input_column);
  gdf_column *n_slice_indices = reinterpret_cast<gdf_column *>(slice_indices);

  try {
    std::vector<gdf_column *> result = cudf::slice(
        *n_column, static_cast<cudf::size_type *>(n_slice_indices->data), n_slice_indices->size);
    cudf::jni::native_jlongArray n_result(env, reinterpret_cast<jlong *>(result.data()),
                                          result.size());
    return n_result.get_jArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ColumnVector_split(JNIEnv *env, jclass clazz,
                                                                    jlong input_column,
                                                                    jlong slice_indices) {
  JNI_NULL_CHECK(env, input_column, "native handle is null", 0);
  JNI_NULL_CHECK(env, slice_indices, "slice indices are null", 0);

  gdf_column *n_column = reinterpret_cast<gdf_column *>(input_column);
  gdf_column *n_slice_indices = reinterpret_cast<gdf_column *>(slice_indices);

  try {
    std::vector<gdf_column *> result = cudf::split(
        *n_column, static_cast<cudf::size_type *>(n_slice_indices->data), n_slice_indices->size);
    cudf::jni::native_jlongArray n_result(env, reinterpret_cast<jlong *>(result.data()),
                                          result.size());
    return n_result.get_jArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_lengths(JNIEnv *env, jclass clazz,
                                                                 jlong view_handle) {
  JNI_NULL_CHECK(env, view_handle, "input column is null", 0);
  try {
    cudf::column_view *n_column = reinterpret_cast<cudf::column_view *>(view_handle);
    std::unique_ptr<cudf::column> result = cudf::strings::count_characters(cudf::strings_column_view(*n_column));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_byteCount(JNIEnv *env, jclass clazz,
                                                                   jlong view_handle) {
  JNI_NULL_CHECK(env, view_handle, "input column is null", 0);
  try {
    cudf::column_view *n_column = reinterpret_cast<cudf::column_view *>(view_handle);
    std::unique_ptr<cudf::column> result = cudf::strings::count_bytes(cudf::strings_column_view(*n_column));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_getDeviceMemoryStringSize(JNIEnv *env, jobject j_object,
                                                                                   jlong handle) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    gdf_column *column = reinterpret_cast<gdf_column *>(handle);
    gdf_dtype dtype = column->dtype;
    if (dtype == GDF_STRING) {
      NVStrings *nvstr = static_cast<NVStrings *>(column->data);
      if (nvstr == nullptr) {
        // This can happen on an empty column.
        return 0;
      }
      return static_cast<jlong>(nvstr->memsize());
    } else if (dtype == GDF_STRING_CATEGORY) {
      NVCategory *cats = static_cast<NVCategory *>(column->dtype_info.category);
      if (cats == nullptr) {
        // This can happen on an empty column.
        return 0;
      }
      unsigned long dict_size = cats->keys_size();
      unsigned long dict_size_total = dict_size * GDF_INT32;
      // NOTE: Assumption being made that strings in each row is of 10 chars. So the result would be approximate.
      // custring_view structure is allocated 8B and 16B for 10 chars as it is aligned to 8 bytes.
      unsigned long category_size_total = dict_size * 24;
      return static_cast<jlong>(category_size_total + dict_size_total);
    } else {
      throw std::logic_error("ONLY STRING AND CATEGORY TYPES ARE SUPPORTED...");
    }
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_findAndReplaceAll(JNIEnv *env,
                                                                           jclass clazz,
                                                                           jlong old_values_handle,
                                                                           jlong new_values_handle,
                                                                           jlong input_handle) {
  JNI_NULL_CHECK(env, old_values_handle, "values column is null", 0);
  JNI_NULL_CHECK(env, new_values_handle, "replace column is null", 0);
  JNI_NULL_CHECK(env, input_handle, "input column is null", 0);

  using cudf::column_view;
  using cudf::column;

  try {
    column_view *input_column = reinterpret_cast<column_view *>(input_handle);
    column_view *old_values_column = reinterpret_cast<column_view *>(old_values_handle);
    column_view *new_values_column = reinterpret_cast<column_view *>(new_values_handle);

    std::unique_ptr<column> result =
        cudf::experimental::find_and_replace_all(*input_column, *old_values_column, *new_values_column);

    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_isNullNative(JNIEnv *env, jclass, jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    std::unique_ptr<cudf::column> ret = cudf::experimental::is_null(*input);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_isNotNullNative(JNIEnv *env, jclass, jlong handle) {
  JNI_NULL_CHECK(env, handle, "input column is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(handle);
    std::unique_ptr<cudf::column> ret = cudf::experimental::is_valid(*input);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_unaryOperation(JNIEnv *env, jclass,
        jlong input_ptr, jint int_op) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    cudf::experimental::unary_op op = static_cast<cudf::experimental::unary_op>(int_op);
    std::unique_ptr<cudf::column> ret = cudf::experimental::unary_operation(*input, op);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_year(JNIEnv *env, jclass,
                                                              jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_year(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_month(JNIEnv *env, jclass,
                                                               jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_month(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_day(JNIEnv *env, jclass,
                                                             jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_day(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_hour(JNIEnv *env, jclass,
                                                              jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_hour(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_minute(JNIEnv *env, jclass,
                                                                jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_minute(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_second(JNIEnv *env, jclass,
                                                                jlong input_ptr) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    const cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_ptr);
    std::unique_ptr<cudf::column> output = cudf::datetime::extract_second(*input);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_castTo(JNIEnv *env,
                                                                jobject j_object,
                                                                jlong handle, jint type) {
  JNI_NULL_CHECK(env, handle, "native handle is null", 0);
  try {
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::data_type n_data_type(static_cast<cudf::type_id>(type));
    std::unique_ptr<cudf::column> result = cudf::experimental::cast(*column, n_data_type);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_stringTimestampToTimestamp(
    JNIEnv *env, jobject j_object, jlong handle, jint time_unit, jstring formatObj) {
  JNI_NULL_CHECK(env, handle, "column is null", 0);
  JNI_NULL_CHECK(env, formatObj, "format is null", 0);

  try {
    cudf::jni::native_jstring format(env, formatObj);
    cudf::column_view *column = reinterpret_cast<cudf::column_view *>(handle);
    cudf::strings_column_view strings_column(*column);

    std::unique_ptr<cudf::column> result = cudf::strings::to_timestamps(strings_column, cudf::data_type(static_cast<cudf::type_id>(time_unit)), format.get());
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_ColumnVector_containsScalar(JNIEnv *env, jobject j_object,
                                                                 jlong j_view_handle, jlong j_scalar_handle) {
  JNI_NULL_CHECK(env, j_view_handle, "haystack vector is null", false);
  JNI_NULL_CHECK(env, j_scalar_handle, "scalar needle is null", false);
  try {
    cudf::column_view* column_view = reinterpret_cast<cudf::column_view*>(j_view_handle);
    cudf::scalar* scalar = reinterpret_cast<cudf::scalar*>(j_scalar_handle);

    return cudf::experimental::contains(*column_view, *scalar);
  }
  CATCH_STD(env, 0);

}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ColumnVector_containsVector(JNIEnv *env, jobject j_object,
                                                                 jlong j_haystack_handle, jlong j_needle_handle) {
  JNI_NULL_CHECK(env, j_haystack_handle, "haystack vector is null", false);
  JNI_NULL_CHECK(env, j_needle_handle, "needle vector is null", false);
  try {
    cudf::column_view* haystack = reinterpret_cast<cudf::column_view*>(j_haystack_handle);
    cudf::column_view* needle = reinterpret_cast<cudf::column_view*>(j_needle_handle);

    std::unique_ptr<cudf::column> result = std::move(cudf::experimental::contains(*haystack, *needle));
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);

}
} // extern "C"
