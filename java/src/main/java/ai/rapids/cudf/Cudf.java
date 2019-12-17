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
package ai.rapids.cudf;

/**
 * This is the binding class for cudf lib.
 */
class Cudf {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /* arith */

  static long gdfBinaryOp(ColumnVector lhs, ColumnVector rhs, BinaryOp op, DType outputType) {
    return gdfBinaryOpVV(lhs.getNativeView(), rhs.getNativeView(),
        op.nativeId, outputType.nativeId);
  }

  private static native long gdfBinaryOpVV(long lhs, long rhs, int op, int dtype);

  static long gdfBinaryOp(Scalar lhs, ColumnVector rhs, BinaryOp op, DType outputType) {
    throw new UnsupportedOperationException(ColumnVector.STANDARD_CUDF_PORTING_MSG);
/*
    if (rhs.getType() == TypeId.STRING_CATEGORY && lhs.getType() == TypeId.STRING
        && BinaryOp.COMPARISON.contains(op)) {
      // Currenty cudf cannot handle string scalars, so convert the string scalar to a
      // category index and compare with that instead.
      lhs = rhs.getCategoryIndex(lhs);
    }
    return gdfBinaryOpSV(lhs.intTypeStorage, lhs.floatTypeStorage, lhs.doubleTypeStorage,
        lhs.isValid, lhs.type.nativeId,
        rhs.getNativeCudfColumnAddress(), op.nativeId, outputType.nativeId);
*/
  }

  private static native long gdfBinaryOpSV(long lhsIntValues, float lhsFValue, double lhsDValue,
                                           boolean lhsIsValid, int lhsDtype,
                                           long rhs,
                                           int op, int dtype);

  static long gdfBinaryOp(ColumnVector lhs, Scalar rhs, BinaryOp op, DType outputType) {
    throw new UnsupportedOperationException(ColumnVector.STANDARD_CUDF_PORTING_MSG);
/*
    if (lhs.getType() == TypeId.STRING_CATEGORY && rhs.getType() == TypeId.STRING
        && BinaryOp.COMPARISON.contains(op)) {
      // Currenty cudf cannot handle string scalars, so convert the string scalar to a
      // category index and compare with that instead.
      if (BinaryOp.INEQUALITY_COMPARISON.contains(op)) {
        // Need to compute value bounds and potentially adjust the operation being performed
        // since scalar might not be present in the category.
        int[] bounds = lhs.getCategoryBounds(rhs);
        if (bounds[0] == bounds[1]) {
          // scalar is present in the category
          rhs = Scalar.fromInt(bounds[0]);
        } else {
          // The scalar is not present in the category so either the lower bound or upper bound
          // needs to be used as a proxy. The upper bound is chosen to avoid any potential issues
          // with negative indices if the value were to be the first key in an updated category.
          rhs = Scalar.fromInt(bounds[1]);
          if (op == BinaryOp.LESS_EQUAL) {
            // Avoid matching the upper bound since it is strictly greater than the scalar.
            op = BinaryOp.LESS;
          } else if (op == BinaryOp.GREATER) {
            // Include the upper bound since it is strictly greater than the scalar.
            op = BinaryOp.GREATER_EQUAL;
          }
        }
      } else {
        rhs = lhs.getCategoryIndex(rhs);
      }
    }
    return gdfBinaryOpVS(lhs.getNativeCudfColumnAddress(),
        rhs.intTypeStorage, rhs.floatTypeStorage, rhs.doubleTypeStorage, rhs.isValid,
        rhs.type.nativeId,
        op.nativeId, outputType.nativeId);
*/
  }

  private static native long gdfBinaryOpVS(long lhs,
                                           long rhsIntValues, float rhsFValue, double rhsDValue,
                                           boolean rhsIsValid, int rhsDtype,
                                           int op, int dtype);


  static Scalar reduce(ColumnVector v, ReductionOp op, DType outType) {
    return reduce(v.getNativeView(), op.nativeId, outType.nativeId);
  }

  private static native Scalar reduce(long v, int op, int dtype);
}
