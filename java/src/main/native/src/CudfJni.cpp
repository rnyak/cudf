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

#include <memory>

#include "cudf/legacy/binaryop.hpp"
#include "cudf/legacy/reduction.hpp"
#include "cudf/legacy/stream_compaction.hpp"
#include "cudf/legacy/unary.hpp"

#include "jni_utils.hpp"

using unique_nvcat_ptr = std::unique_ptr<NVCategory, decltype(&NVCategory::destroy)>;
using unique_nvstr_ptr = std::unique_ptr<NVStrings, decltype(&NVStrings::destroy)>;

namespace cudf {
namespace jni {

static const jint MINIMUM_JNI_VERSION = JNI_VERSION_1_6;

static jni_rmm_unique_ptr<cudf::valid_type>
copy_validity(JNIEnv *env, cudf::size_type size, cudf::size_type null_count, cudf::valid_type *valid) {
  jni_rmm_unique_ptr<cudf::valid_type> ret{};
  if (null_count > 0) {
    cudf::size_type copy_size = ((size + 7) / 8);
    cudf::size_type alloc_size = gdf_valid_allocation_size(size);
    ret = jni_rmm_alloc<cudf::valid_type>(env, alloc_size);
    JNI_CUDA_TRY(env, 0, cudaMemcpy(ret.get(), valid, copy_size, cudaMemcpyDeviceToDevice));
  }
  return ret;
}

static jlong cast_string_cat_to(JNIEnv *env, NVCategory *cat, gdf_dtype target_type,
                                gdf_time_unit target_unit, cudf::size_type size,
                                cudf::size_type null_count, cudf::valid_type *valid) {
  switch (target_type) {
    case GDF_STRING: {
      if (size == 0) {
        gdf_column_wrapper output(size, target_type, null_count, nullptr,
                                nullptr);
        return reinterpret_cast<jlong>(output.release());
      }
      unique_nvstr_ptr str(cat->to_strings(), &NVStrings::destroy);

      jni_rmm_unique_ptr<cudf::valid_type> valid_copy = copy_validity(env, size, null_count, valid);

      gdf_column_wrapper output(size, target_type, null_count, str.release(), valid_copy.release());
      return reinterpret_cast<jlong>(output.release());
    }
    default: throw std::logic_error("Unsupported type to cast a string_cat to");
  }
}

static jlong cast_string_to(JNIEnv *env, NVStrings *str, gdf_dtype target_type,
                            gdf_time_unit target_unit, cudf::size_type size, cudf::size_type null_count,
                            cudf::valid_type *valid) {
  switch (target_type) {
    case GDF_STRING_CATEGORY: {
      if (size == 0) {
        gdf_column_wrapper output(size, target_type, null_count, nullptr,
                                nullptr, nullptr);
        return reinterpret_cast<jlong>(output.release());
      }
      unique_nvcat_ptr cat(NVCategory::create_from_strings(*str), &NVCategory::destroy);
      auto cat_data = jni_rmm_alloc<int>(env, sizeof(int) * size);
      if (size != cat->get_values(cat_data.get(), true)) {
        JNI_THROW_NEW(env, "java/lang/IllegalStateException", "Internal Error copying str cat data",
                      0);
      }

      jni_rmm_unique_ptr<cudf::valid_type> valid_copy = copy_validity(env, size, null_count, valid);

      gdf_column_wrapper output(size, target_type, null_count, cat_data.release(),
                                valid_copy.release(), cat.release());
      return reinterpret_cast<jlong>(output.release());
    }
    default: throw std::logic_error("Unsupported type to cast a string to");
  }
}

} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
    return JNI_ERR;
  }

  // cache any class objects and method IDs here

  return cudf::jni::MINIMUM_JNI_VERSION;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *) {
  JNIEnv *env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
    return;
  }

  // release cached class objects here.
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfBinaryOpVV(JNIEnv *env, jclass, jlong lhs_ptr,
                                                               jlong rhs_ptr, jint int_op,
                                                               jint out_dtype) {
  JNI_NULL_CHECK(env, lhs_ptr, "lhs is null", 0);
  JNI_NULL_CHECK(env, rhs_ptr, "rhs is null", 0);
  try {
    gdf_column *lhs = reinterpret_cast<gdf_column *>(lhs_ptr);
    gdf_column *rhs = reinterpret_cast<gdf_column *>(rhs_ptr);
    gdf_dtype out_type = static_cast<gdf_dtype>(out_dtype);
    gdf_binary_operator op = static_cast<gdf_binary_operator>(int_op);
    cudf::jni::gdf_column_wrapper ret(lhs->size, out_type,
                                      lhs->valid != nullptr || rhs->valid != nullptr);
    // Should be null count           lhs->null_count > 0 || rhs->null_count >
    // 0);

    cudf::binary_operation(ret.get(), lhs, rhs, op);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfBinaryOpSV(
    JNIEnv *env, jclass, jlong lhs_int_values, jfloat lhs_f_value, jdouble lhs_d_value,
    jboolean lhs_is_valid, int lhs_dtype, jlong rhs_ptr, jint int_op, jint out_dtype) {
  JNI_NULL_CHECK(env, rhs_ptr, "rhs is null", 0);
  try {
    gdf_scalar lhs{};
//    cudf::jni::gdf_scalar_init(&lhs, lhs_int_values, lhs_f_value, lhs_d_value, lhs_is_valid,
//                               lhs_dtype);
    throw std::logic_error("BAD IMPLEMENTATION");
    gdf_column *rhs = reinterpret_cast<gdf_column *>(rhs_ptr);
    gdf_dtype out_type = static_cast<gdf_dtype>(out_dtype);
    gdf_binary_operator op = static_cast<gdf_binary_operator>(int_op);
    cudf::jni::gdf_column_wrapper ret(rhs->size, out_type, !lhs.is_valid || rhs->valid != nullptr);
    // Should be null count           !lhs.is_valid || rhs->null_count > 0);

    cudf::binary_operation(ret.get(), &lhs, rhs, op);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfBinaryOpVS(
    JNIEnv *env, jclass, jlong lhs_ptr, jlong rhs_int_values, jfloat rhs_f_value,
    jdouble rhs_d_value, jboolean rhs_is_valid, int rhs_dtype, jint int_op, jint out_dtype) {
  JNI_NULL_CHECK(env, lhs_ptr, "lhs is null", 0);
  try {
    gdf_column *lhs = reinterpret_cast<gdf_column *>(lhs_ptr);
    gdf_scalar rhs{};
//    cudf::jni::gdf_scalar_init(&rhs, rhs_int_values, rhs_f_value, rhs_d_value, rhs_is_valid,
//                               rhs_dtype);
    throw std::logic_error("BAD IMPLEMENTATION");
    gdf_dtype out_type = static_cast<gdf_dtype>(out_dtype);
    gdf_binary_operator op = static_cast<gdf_binary_operator>(int_op);
    cudf::jni::gdf_column_wrapper ret(lhs->size, out_type, !rhs.is_valid || lhs->valid != nullptr);
    // Should be null count           !rhs.is_valid || lhs->null_count > 0);

    cudf::binary_operation(ret.get(), lhs, &rhs, op);
    return reinterpret_cast<jlong>(ret.release());
  }
  CATCH_STD(env, 0);
}


JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cudf_gdfCast(JNIEnv *env, jclass, jlong input_ptr,
                                                         jint dtype, jint time_unit) {
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  try {
    gdf_column *input = reinterpret_cast<gdf_column *>(input_ptr);
    gdf_dtype c_dtype = static_cast<gdf_dtype>(dtype);
    gdf_dtype_extra_info info{};
    gdf_time_unit c_time_unit = static_cast<gdf_time_unit>(time_unit);
    size_t size = input->size;
    if (input->dtype == GDF_STRING) {
      NVStrings *str = static_cast<NVStrings *>(input->data);
      return cudf::jni::cast_string_to(env, str, c_dtype, c_time_unit, size, input->null_count,
                                       input->valid);
    } else if (input->dtype == GDF_STRING_CATEGORY && c_dtype == GDF_STRING) {
      NVCategory *cat = static_cast<NVCategory *>(input->dtype_info.category);
      return cudf::jni::cast_string_cat_to(env, cat, c_dtype, c_time_unit, size, input->null_count,
                                           input->valid);
    } else {
      std::unique_ptr<gdf_column, decltype(free) *> ret(
          static_cast<gdf_column *>(malloc(sizeof(gdf_column))), free);
      info.time_unit = c_time_unit;
      *ret.get() = cudf::cast(*input, c_dtype, info);
      return reinterpret_cast<jlong>(ret.release());
    }
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_Cudf_reduce(JNIEnv *env, jclass, jlong jcol, jint jop,
                                                          jint jdtype) {
  JNI_NULL_CHECK(env, jcol, "input column is null", 0);
  try {
    gdf_column *col = reinterpret_cast<gdf_column *>(jcol);
    cudf::reduction::operators op = static_cast<cudf::reduction::operators>(jop);
    gdf_dtype dtype = static_cast<gdf_dtype>(jdtype);
    gdf_scalar scalar = cudf::reduce(col, op, dtype);
//    return cudf::jni::jscalar_from_scalar(env, scalar, col->dtype_info.time_unit);
    throw std::logic_error("BAD IMPLEMENTATION");
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Cudf_getCategoryIndex(JNIEnv *env, jclass, jlong jcol,
                                                                 jbyteArray jstr) {
  JNI_NULL_CHECK(env, jcol, "input column is null", -1);
  JNI_NULL_CHECK(env, jstr, "string data is null", -1);
  try {
    gdf_column *col = reinterpret_cast<gdf_column *>(jcol);
    if (col->size <= 0) {
      // it is empty so nothing is in there.
      return -1;
    }
    NVCategory *cat = static_cast<NVCategory *>(col->dtype_info.category);
    JNI_NULL_CHECK(env, cat, "category is null", -1);

    int len = env->GetArrayLength(jstr);
    cudf::jni::check_java_exception(env);
    std::unique_ptr<char[]> str(new char[len + 1]);
    env->GetByteArrayRegion(jstr, 0, len, reinterpret_cast<jbyte *>(str.get()));
    cudf::jni::check_java_exception(env);
    str[len] = '\0'; // NUL-terminate UTF-8 string

    return cat->get_value(str.get());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jintArray JNICALL Java_ai_rapids_cudf_Cudf_getCategoryBounds(JNIEnv *env, jclass,
     jlong jcol, jbyteArray jstr) {
  JNI_NULL_CHECK(env, jcol, "input column is null", nullptr);
  JNI_NULL_CHECK(env, jstr, "string data is null", nullptr);
  try {
    gdf_column *col = reinterpret_cast<gdf_column *>(jcol);
    std::pair<int, int> bounds(-1, -1);
    if (col->size > 0) {
      NVCategory *cat = static_cast<NVCategory *>(col->dtype_info.category);
      JNI_NULL_CHECK(env, cat, "category is null", nullptr);

      int len = env->GetArrayLength(jstr);
      cudf::jni::check_java_exception(env);
      std::unique_ptr<char[]> str(new char[len + 1]);
      env->GetByteArrayRegion(jstr, 0, len, reinterpret_cast<jbyte *>(str.get()));
      cudf::jni::check_java_exception(env);
      str[len] = '\0'; // NUL-terminate UTF-8 string

      bounds = cat->get_value_bounds(str.get());
    }

    cudf::jni::native_jintArray jbounds(env, 2);
    jbounds[0] = bounds.first;
    jbounds[1] = bounds.second;
    return jbounds.get_jArray();
  }
  CATCH_STD(env, 0);
}
} // extern "C"
