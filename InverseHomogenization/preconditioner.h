/**
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2012-2023 by the DOpElib authors
 * Copyright (C) 2021-2023 by Manaswinee Bezbaruah, Matthias Maier, Winifried Wollner
 **/

#pragma once

#include <deal.II/base/config.h>
#include <deal.II/lac/sparse_direct.h>

template <typename MATRIX>
class Preconditioner
{
public:
  void initialize(const MATRIX& matrix)
  {
    solver_ = std::make_unique<SparseDirectUMFPACK>();
    solver_->initialize(matrix.block(0,0));
  }

  template <typename VECTOR>
  void vmult(VECTOR &dst, const VECTOR &src) const
  {
    for (unsigned int i = 0; i < src.n_blocks(); ++i)
      solver_->vmult(dst.block(i), src.block(i));
  }

private:
  std::unique_ptr<SparseDirectUMFPACK> solver_;
};
