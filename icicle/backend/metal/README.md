# ICICLE Metal Backend Libraries

The ICICLE framework provides dedicated backend Metal libraries for fields, curves, and hashes. Each library embeds precompiled Metal shaders and provides runtime support for dispatching GPU computations efficiently. This README outlines the structure, shader organization, and naming conventions used across the ICICLE Metal backend libraries.

## **Library Structure**

Each ICICLE frontend (field, curve, and hash) has a corresponding Metal backend library:

1. **Field Libraries**:
   - Represent scalar fields (e.g., BN254 scalar field or M31 field).
   - Contain shaders for operations like vector-addition, ntt and others
   - Shader naming convention: `field_{FIELD_NAME}_{KERNEL_NAME}`.

2. **Curve Libraries**:
   - Represent elliptic curves (e.g., BN254 curve).
   - Include the base field, extension fields, G1, G2 curves arithmetic and operations like msm, ecntt.
   - Shader naming convention: `curve_{CURVE_NAME}_{KERNEL_NAME}`.

3. **Hash Library**:
   - Supports cryptographic hashing operations.
   - Includes shaders for hash computations (e.g., Keccak or Blake2).
   - Shader naming convention: `hash_{KERNEL_NAME}`.

## **Shader Organization**

The Metal shaders are organized as follows:

- **Shaders Directory**:
  - `shaders/fields/{FIELD}`: Contains field-specific shaders.
  - `shaders/curves/{CURVE}`: Contains curve-specific shaders.
  - `shaders/hash`: Contains hash-related shaders.

- **Include Templates**:
  - `shaders/include`: Contains reusable shader templates shared across libraries.
  - These templates are referenced by field, curve, or hash shaders during compilation.

## **Compilation and Embedding**

1. **Compilation**:
   - All shaders (per library) are compiled into `.air` intermediate files and then linked into a `.metallib` object.
   - Each `.metallib` object is specific to the library it belongs to.

2. **Embedding**:
   - The `.metallib` object is embedded into the corresponding library binary.
   - At runtime, the library loads the `.metallib` to access the compiled Metal kernels.

## **Runtime Dispatch**

At runtime, shaders are accessed through the `METAL_GET_PIPELINE(KERNEL_NAME)` macro. The naming convention ensures that the correct kernel is loaded based on the prefix associated with the library.

### **Naming Convention**
- **Field Libraries**:
  - Kernel names are prefixed as `field_{FIELD_NAME}_{KERNEL_NAME}`.
  - Example: `field_babybear_vector_add`.

- **Curve Libraries**:
  - Kernel names are prefixed as `curve_{CURVE_NAME}_{KERNEL_NAME}`.
  - Example: `curve_bn254_msm`.

- **Hash Library**:
  - Kernel names are prefixed as `hash_{KERNEL_NAME}`.
  - Example: `hash_keccak`.

### **Usage Example**

For the scalar field **BabyBear**, a kernel performing vector addition is named:

- **Shader Name**: `field_babybear_vector_add`.
- **Runtime Call**: `METAL_GET_PIPELINE("vector_add")`.

This ensures the correct kernel is loaded from the `BabyBear` field library at runtime.

## **Key Notes**

1. **Prefixes**:
   - Always use the correct prefix (`field_`, `curve_`, or `hash_`) when naming kernels in shaders.
   - Only the `KERNEL_NAME` is specified during runtime calls.

2. **Consistency**:
   - Ensure kernel names match the naming convention to avoid runtime errors.

3. **Modular Design**:
   - Each ICICLE Metal backend library is self-contained, allowing seamless integration with ICICLE frontends.

