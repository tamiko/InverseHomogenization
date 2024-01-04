/**
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2012-2023 by the DOpElib authors
 * Copyright (C) 2021-2023 by Manaswinee Bezbaruah, Matthias Maier, Winifried Wollner
 **/

#pragma once

#include <deal.II/base/config.h>
#include <deal.II/base/parameter_acceptor.h>

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <interfaces/functionalinterface.h>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

using namespace std;
using namespace dealii;
using namespace DOpE;

template <int dealdim>
class ProblemDescription : public dealii::ParameterAcceptor
{
public:
  constexpr static unsigned int control_size = dealdim;
  constexpr static unsigned int state_size = 2 * dealdim;

  using rank0_type = std::complex<double>;
  using rank1_type = Tensor<1, dealdim, rank0_type>;
  using rank2_type = Tensor<2, dealdim, rank0_type>;


  ProblemDescription()
    : ParameterAcceptor("material parameters")
  {
    ParameterAcceptor::parse_parameters_call_back.connect(
        std::bind(&ProblemDescription::parse_parameters_callback, this));

    omega_ = 0.5;
    this->add_parameter("omega", omega_, "Angular frequency");

    tau_ = 243.;
    this->add_parameter("tau", tau_, "Relaxation time");

    epsilon_[0][0] = 1.0;
    epsilon_[1][1] = 1.0;
    this->add_parameter("epsilon", epsilon_, "Permittivity tensor");

    target_epsilon_[0][0] = 1.4;
    target_epsilon_[1][1] = 1.2;
    this->add_parameter(
        "target epsilon", target_epsilon_, "Target permittivity tensor for optimization procedure");

    weights_[0][0] = 1.0;
    weights_[0][1] = 1.0;
    weights_[1][0] = 1.0;
    weights_[1][1] = 1.0;
    this->add_parameter("weights", weights_, "Weights for the weighted Frobenius norm");
  }

  double omega() const
  {
    return omega_;
  }

  rank0_type tau() const
  {
    return tau_;
  }

  const rank2_type epsilon(const dealii::Point<dealdim> &/*position*/) const
  {
    return epsilon_;
  }

  const rank2_type target_epsilon() const
  {
    return target_epsilon_;
  }

  const Tensor<2, dealdim> weights() const
  {
    return weights_;
  }

  rank2_type sigma(const dealii::Point<dealdim> &/*position*/) const
  {
    return sigma_;
  }

  static constexpr dealii::Tensor<2, dealdim> identity()
  {
    return identity_;
  }

  void parse_parameters_callback()
  {
    /* A simple rescaled Drude model: */

    constexpr double omega_p = 4. / 137.;
    for (unsigned int d = 0; d < dealdim; ++d) {
      sigma_[d][d] = 1.0i * omega_p / (omega_ + 1.0i / tau_);
    }
  }

private:
  double omega_;
  rank0_type tau_;

  rank2_type epsilon_;
  rank2_type target_epsilon_;
  Tensor<2, dealdim> weights_;
  rank2_type sigma_;

  static constexpr dealii::Tensor<2, dealdim> identity_{{{1.0, 0.0}, {0.0, 1.0}}};
};
