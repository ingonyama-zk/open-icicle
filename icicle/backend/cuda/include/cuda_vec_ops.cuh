#pragma once
#include <cuda.h>
#include "icicle/program/program.h"

template <typename S>
__device__ inline void execute_instructions(
  S** data,
  const InstructionType* instructions,
  const int nof_instructions,
  const int nof_variables,
  const int nof_constants,
  const int elm_idx)
{
  for (int inst_idx = 0; inst_idx < nof_instructions; ++inst_idx) {
    // decode instruction
    InstructionType instruction = instructions[inst_idx];
    const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
    int func_opcode = (int)inst_arr[Program<S>::INST_OPCODE];
    int operand_1 = (int)inst_arr[Program<S>::INST_OPERAND1];
    int operand_2 = (int)inst_arr[Program<S>::INST_OPERAND2];
    int target = (int)inst_arr[Program<S>::INST_RESULT];

    // check if operands are constants
    int is_operand_1_constant = (int)((operand_1 >= nof_variables) & (operand_1 < nof_variables + nof_constants));
    int is_operand_2_constant = (int)((operand_2 >= nof_variables) & (operand_2 < nof_variables + nof_constants));

    // execute instruction
    switch (func_opcode) {
    case OP_COPY:
      data[target][elm_idx] = data[operand_1][elm_idx * (1 - is_operand_1_constant)];
      break;
    case OP_ADD:
      data[target][elm_idx] =
        data[operand_1][elm_idx * (1 - is_operand_1_constant)] + data[operand_2][elm_idx * (1 - is_operand_2_constant)];
      break;
    case OP_MULT:
      data[target][elm_idx] =
        data[operand_1][elm_idx * (1 - is_operand_1_constant)] * data[operand_2][elm_idx * (1 - is_operand_2_constant)];
      break;
    case OP_SUB:
      data[target][elm_idx] =
        data[operand_1][elm_idx * (1 - is_operand_1_constant)] - data[operand_2][elm_idx * (1 - is_operand_2_constant)];
      break;
    case OP_INV:
      data[target][elm_idx] = data[operand_1][elm_idx * (1 - is_operand_1_constant)].inverse();
      break;
    case NOF_OPERATIONS + AB_MINUS_C:
      data[3][elm_idx] = data[0][elm_idx] * data[1][elm_idx] - data[2][elm_idx];
      break;
    case NOF_OPERATIONS + EQ_X_AB_MINUS_C:
      data[4][elm_idx] = data[3][elm_idx] * (data[0][elm_idx] * data[1][elm_idx] - data[2][elm_idx]);
      break;
    }
  }
}