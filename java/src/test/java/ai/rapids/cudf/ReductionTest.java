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

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ReductionTest extends CudfTestBase {

  public static final double DELTAD = 0.00001;
  public static final float DELTAF = 0.00001f;

  private static Scalar buildExpectedScalar(ReductionOp op, DType baseType, Object expectedObject) {
    if (expectedObject == null) {
      return Scalar.fromNull(baseType);
    }
    if (op == ReductionOp.STD) {
      return Scalar.fromDouble((Double) expectedObject);
    }
    switch (baseType) {
    case BOOL8:
      return Scalar.fromBool((Boolean) expectedObject);
    case INT8:
      return Scalar.fromByte((Byte) expectedObject);
    case INT16:
      return Scalar.fromShort((Short) expectedObject);
    case INT32:
      return Scalar.fromInt((Integer) expectedObject);
    case INT64:
      return Scalar.fromLong((Long) expectedObject);
    case FLOAT32:
      return Scalar.fromFloat((Float) expectedObject);
    case FLOAT64:
      return Scalar.fromDouble((Double) expectedObject);
    case TIMESTAMP_DAYS:
      return Scalar.timestampDaysFromInt((Integer) expectedObject);
    case TIMESTAMP_SECONDS:
    case TIMESTAMP_MILLISECONDS:
    case TIMESTAMP_MICROSECONDS:
    case TIMESTAMP_NANOSECONDS:
      return Scalar.timestampFromLong(baseType, (Long) expectedObject);
    case STRING:
      return Scalar.fromString((String) expectedObject);
    default:
      throw new IllegalArgumentException("Unexpected type: " + baseType);
    }
  }

  private static Stream<Arguments> createBooleanParams() {
    Boolean[] vals = new Boolean[]{true, true, null, false, true, false, null};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Boolean[0], null, 0.),
        Arguments.of(ReductionOp.SUM, new Boolean[]{null, null, null}, null, 0.),
        Arguments.of(ReductionOp.SUM, vals, true, 0.),
        Arguments.of(ReductionOp.MIN, vals, false, 0.),
        Arguments.of(ReductionOp.MAX, vals, true, 0.),
        Arguments.of(ReductionOp.PRODUCT, vals, false, 0.),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, true, 0.),
        Arguments.of(ReductionOp.STD, vals, 0.5477225575051662, DELTAD)
    );
  }

  private static Stream<Arguments> createByteParams() {
    Byte[] vals = new Byte[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Byte[0], null, 0.),
        Arguments.of(ReductionOp.SUM, new Byte[]{null, null, null}, null, 0.),
        Arguments.of(ReductionOp.SUM, vals, (byte) 83, 0.),
        Arguments.of(ReductionOp.MIN, vals, (byte) -1, 0.),
        Arguments.of(ReductionOp.MAX, vals, (byte) 123, 0.),
        Arguments.of(ReductionOp.PRODUCT, vals, (byte) 160, 0.),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, (byte) 47, 0.),
        Arguments.of(ReductionOp.STD, vals, 49.24530434467839, DELTAD)
    );
  }

  private static Stream<Arguments> createShortParams() {
    Short[] vals = new Short[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Short[0], null, 0.),
        Arguments.of(ReductionOp.SUM, new Short[]{null, null, null}, null, 0.),
        Arguments.of(ReductionOp.SUM, vals, (short) 339, 0.),
        Arguments.of(ReductionOp.MIN, vals, (short) -1, 0.),
        Arguments.of(ReductionOp.MAX, vals, (short) 123, 0.),
        Arguments.of(ReductionOp.PRODUCT, vals, (short) -22624, 0.),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, (short) 31279, 0.),
        Arguments.of(ReductionOp.STD, vals, 49.24530434467839, DELTAD)
    );
  }

  private static Stream<Arguments> createIntParams() {
    Integer[] vals = new Integer[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Integer[0], null, 0.),
        Arguments.of(ReductionOp.SUM, new Integer[]{null, null, null}, null, 0.),
        Arguments.of(ReductionOp.SUM, vals, 339, 0.),
        Arguments.of(ReductionOp.MIN, vals, -1, 0.),
        Arguments.of(ReductionOp.MAX, vals, 123, 0.),
        Arguments.of(ReductionOp.PRODUCT, vals, -258300000, 0.),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, 31279, 0.),
        Arguments.of(ReductionOp.STD, vals, 49.24530434467839, DELTAD)
    );
  }

  private static Stream<Arguments> createLongParams() {
    Long[] vals = new Long[]{-1L, 7L, 123L, null, 50L, 60L, 100L};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Long[0], null, 0.),
        Arguments.of(ReductionOp.SUM, new Long[]{null, null, null}, null, 0.),
        Arguments.of(ReductionOp.SUM, vals, 339L, 0.),
        Arguments.of(ReductionOp.MIN, vals, -1L, 0.),
        Arguments.of(ReductionOp.MAX, vals, 123L, 0.),
        Arguments.of(ReductionOp.PRODUCT, vals, -258300000L, 0.),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, 31279L, 0.),
        Arguments.of(ReductionOp.STD, vals, 49.24530434467839, DELTAD)
    );
  }

  private static Stream<Arguments> createFloatParams() {
    Float[] vals = new Float[]{-1f, 7f, 123f, null, 50f, 60f, 100f};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Float[0], null, 0f),
        Arguments.of(ReductionOp.SUM, new Float[]{null, null, null}, null, 0f),
        Arguments.of(ReductionOp.SUM, vals, 339f, 0f),
        Arguments.of(ReductionOp.MIN, vals, -1f, 0f),
        Arguments.of(ReductionOp.MAX, vals, 123f, 0f),
        Arguments.of(ReductionOp.PRODUCT, vals, -258300000f, 0f),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, 31279f, 0f),
        Arguments.of(ReductionOp.STD, vals, 49.24530434467839f, DELTAF)
    );
  }

  private static Stream<Arguments> createDoubleParams() {
    Double[] vals = new Double[]{-1., 7., 123., null, 50., 60., 100.};
    return Stream.of(
        Arguments.of(ReductionOp.SUM, new Double[0], null, 0.),
        Arguments.of(ReductionOp.SUM, new Double[]{null, null, null}, null, 0.),
        Arguments.of(ReductionOp.SUM, vals, 339., 0.),
        Arguments.of(ReductionOp.MIN, vals, -1., 0.),
        Arguments.of(ReductionOp.MAX, vals, 123., 0.),
        Arguments.of(ReductionOp.PRODUCT, vals, -258300000., 0.),
        Arguments.of(ReductionOp.SUMOFSQUARES, vals, 31279., 0.),
        Arguments.of(ReductionOp.STD, vals, 49.24530434467839, DELTAD)
    );
  }

  private static Stream<Arguments> createTimestampDaysParams() {
    Integer[] vals = new Integer[]{-1, 7, 123, null, 50, 60, 100};
    return Stream.of(
        Arguments.of(ReductionOp.MAX, new Integer[0], null),
        Arguments.of(ReductionOp.MAX, new Integer[]{null, null, null}, null),
        Arguments.of(ReductionOp.MAX, vals, 123),
        Arguments.of(ReductionOp.MIN, vals, -1)
    );
  }

  private static Stream<Arguments> createTimestampResolutionParams() {
    Long[] vals = new Long[]{-1L, 7L, 123L, null, 50L, 60L, 100L};
    return Stream.of(
        Arguments.of(ReductionOp.MAX, new Long[0], null),
        Arguments.of(ReductionOp.MAX, new Long[]{null, null, null}, null),
        Arguments.of(ReductionOp.MIN, vals, -1L),
        Arguments.of(ReductionOp.MAX, vals, 123L)
    );
  }

  private static void assertEqualsDelta(ReductionOp op, Scalar expected, Scalar result,
      Double delta) {
    if (op == ReductionOp.STD) {
      assertEquals(expected.getDouble(), result.getDouble(), delta);
    } else {
      assertEquals(expected, result);
    }
  }

  private static void assertEqualsDelta(ReductionOp op, Scalar expected, Scalar result,
      Float delta) {
    if (op == ReductionOp.STD) {
      assertEquals(expected.getFloat(), result.getFloat(), delta);
    } else {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createBooleanParams")
  void testBoolean(ReductionOp op, Boolean[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.BOOL8, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedBooleans(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createByteParams")
  void testByte(ReductionOp op, Byte[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.INT8, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedBytes(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createShortParams")
  void testShort(ReductionOp op, Short[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.INT16, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedShorts(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createIntParams")
  void testInt(ReductionOp op, Integer[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.INT32, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedInts(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongParams")
  void testLong(ReductionOp op, Long[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.INT64, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createFloatParams")
  void testFloat(ReductionOp op, Float[] values, Object expectedObject, Float delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.FLOAT32, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedFloats(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createDoubleParams")
  void testDouble(ReductionOp op, Double[] values, Object expectedObject, Double delta) {
    try (Scalar expected = buildExpectedScalar(op, DType.FLOAT64, expectedObject);
         ColumnVector v = ColumnVector.fromBoxedDoubles(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEqualsDelta(op, expected, result, delta);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampDaysParams")
  void testTimestampDays(ReductionOp op, Integer[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_DAYS, expectedObject);
         ColumnVector v = ColumnVector.timestampDaysFromBoxedInts(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampResolutionParams")
  void testTimestampSeconds(ReductionOp op, Long[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_SECONDS, expectedObject);
         ColumnVector v = ColumnVector.timestampSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampResolutionParams")
  void testTimestampMilliseconds(ReductionOp op, Long[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_MILLISECONDS, expectedObject);
         ColumnVector v = ColumnVector.timestampMilliSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampResolutionParams")
  void testTimestampMicroseconds(ReductionOp op, Long[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_MICROSECONDS, expectedObject);
         ColumnVector v = ColumnVector.timestampMicroSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createTimestampResolutionParams")
  void testTimestampNanoseconds(ReductionOp op, Long[] values, Object expectedObject) {
    try (Scalar expected = buildExpectedScalar(op, DType.TIMESTAMP_NANOSECONDS, expectedObject);
         ColumnVector v = ColumnVector.timestampNanoSecondsFromBoxedLongs(values);
         Scalar result = v.reduce(op, expected.getType())) {
      assertEquals(expected, result);
    }
  }
}
