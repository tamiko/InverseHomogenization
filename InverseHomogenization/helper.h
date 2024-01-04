/**
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2012-2023 by the DOpElib authors
 * Copyright (C) 2021-2023 by Manaswinee Bezbaruah, Matthias Maier, Winifried Wollner
 **/

#pragma once

#include "problem_description.h"

using namespace std;
using namespace dealii;
using namespace DOpE;


//
// Given a complex-valued rank-2 tensor, extract the real-valued rank-2
// tensor consisting of the real part of all entries.
//
template <int dim>
DEAL_II_ALWAYS_INLINE inline Tensor<2, dim> real(const Tensor<2, dim, std::complex<double>> tensor)
{
  Tensor<2, dim> ret;
  for (unsigned int i = 0; i < dim; i++) {
    for (unsigned int j = 0; j < dim; j++) {
      ret[i][j] = tensor[i][j].real();
    }
  }
  return ret;
}


//
// Given a complex-valued rank-2 tensor, extract the real-valued rank-2
// tensor consisting of the imaginary part of all entries.
//
template <int dim>
DEAL_II_ALWAYS_INLINE inline Tensor<2, dim> imag(const Tensor<2, dim, std::complex<double>> tensor)
{
  Tensor<2, dim> ret;
  for (unsigned int i = 0; i < dim; i++) {
    for (unsigned int j = 0; j < dim; j++) {
      ret[i][j] = tensor[i][j].imag();
    }
  }
  return ret;
}


//
// Compute projection matrix: Id - n \otimes n
//
template <int dealdim>
DEAL_II_ALWAYS_INLINE inline Tensor<2, dealdim>
projector_matrix(const Tensor<1, dealdim> &normal)
{
  static constexpr dealii::Tensor<2, dealdim> identity{{{1.0, 0.0}, {0.0, 1.0}}};

  return identity - outer_product(normal / normal.norm(), normal / normal.norm());
}


//
// Functions to select real and imaginary components from the state variable
//


//
// By convention we set
//
//   (jacobian)_{ij} = \partial_j u_i,  i,j = 1, ..., dim
//
// Thus, the first index i is the component and the second index j the partial
// derivative \partial_j of the i-th component. We store the real part of
// the Jacobian first and the imaginary part second.
//
template <int dealdim>
DEAL_II_ALWAYS_INLINE inline Tensor<2, dealdim> ToRe(const std::vector<Tensor<1, dealdim>> &ug)
{
  Tensor<2, dealdim> ret;
  for (unsigned int i = 0; i < dealdim; i++) {
    for (unsigned int j = 0; j < dealdim; j++) {
      ret[i][j] = ug[i][j];
    }
  }
  return ret;
}


//
// By convention we set
//
//   (jacobian)_{ij} = \partial_j u_i,  i,j = 1, ..., dim
//
// Thus, the first index i is the component and the second index j the partial
// derivative \partial_j of the i-th component. We store the real part of
// the Jacobian first and the imaginary part second.
//
template <int dealdim>
DEAL_II_ALWAYS_INLINE inline Tensor<2, dealdim> ToIm(const std::vector<Tensor<1, dealdim>> &ug)
{
  Tensor<2, dealdim> ret;
  for (unsigned int c = 0; c < dealdim; c++) {
    for (unsigned int i = 0; i < dealdim; i++) {
      ret[c][i] = ug[c + dealdim][i];
    }
  }
  return ret;
}


//
// By convention we set
//
//   (jacobian)_{ij} = \partial_j u_i,  i,j = 1, ..., dim
//
// Thus, the first index i is the component and the second index j the partial
// derivative \partial_j of the i-th component.
//
// Compound function that returns the compound complex-valued state
//
template <int dealdim>
DEAL_II_ALWAYS_INLINE inline Tensor<2, dealdim, std::complex<double>>
ToJacobian(const std::vector<Tensor<1, dealdim>> &ug, const double sign = 1.)
{
  return ToRe(ug) + std::complex<double>(0., sign) * ToIm(ug);
}


//
// Functions for computing deformation tensors and determinants from the
// control value:
//

template <int dealdim>
DEAL_II_ALWAYS_INLINE inline Tensor<2, dealdim, double>
FInvT(const std::vector<Tensor<1, dealdim>> &qg)
{
  /* \nabla q */
  auto F = ToRe(qg);
  /* + Id */
  for (unsigned int i = 0; i < dealdim; ++i)
    F[i][i] += 1.;
  F = transpose(F);
  F = invert(F);
  return F;
}

template <int dealdim>
DEAL_II_ALWAYS_INLINE inline Tensor<2, dealdim, std::complex<double>>
deltaFInvT(const Tensor<2, dealdim> &FInvT, const Tensor<2, dealdim> &gradqg)
{
  auto delta_FT = transpose(gradqg);
  return -FInvT * delta_FT * FInvT;
}

template <int dealdim>
DEAL_II_ALWAYS_INLINE inline double deltaDetF(const double DetF,
                                              const std::vector<Tensor<1, dealdim>> &qg,
                                              const Tensor<2, dealdim> &gradqg)
{
  /*delta(det(F)) = det(F)*(trace(inverse(F)*delta(F))) --> Jacobi's Formula for invertible F */
  /* \nabla q */
  auto F = ToRe(qg);
  /* + Id */
  for (unsigned int i = 0; i < dealdim; ++i)
    F[i][i] += 1.;
  auto delta_F = gradqg;
  auto delta_Det_F = DetF * trace(invert(F) * delta_F);
  return delta_Det_F;
}

template <int dealdim>
DEAL_II_ALWAYS_INLINE inline double DetF(const std::vector<Tensor<1, dealdim>> &qg)
{
  /* \nabla q */
  auto F = ToRe(qg);
  /* + Id */
  for (unsigned int i = 0; i < dealdim; ++i)
    F[i][i] += 1.;
  return determinant(F);
}


namespace
{
  template <int dim, typename Callable>
  class ToFunction : public dealii::Function<dim>
  {
  public:
    ToFunction(const Callable &callable)
        : dealii::Function<dim>(dim)
        , callable_(callable)
    {
    }

    virtual double value(const dealii::Point<dim> &point, unsigned int component) const
    {
      return callable_(point)[component];
    }

  private:
    const Callable callable_;
  };
} // namespace


template <int dim, typename Callable>
ToFunction<dim, Callable> to_function(const Callable &callable)
{
  return {callable};
}


namespace
{
  template <typename T>
  constexpr bool is_edc = false;

#if DEAL_II_VERSION_GTE(9, 3, 0)
  template <bool DH, typename VECTOR, int dim>
#else
  template <template <int, int> class DH, typename VECTOR, int dim>
#endif
  constexpr bool is_edc<ElementDataContainer<DH, VECTOR, dim>> = true;
} // namespace


const auto get_grads_control = [](const auto &edc, const std::string &name) {
  using edc_type = std::remove_cv_t<std::remove_reference_t<decltype(edc)>>;
  constexpr auto dealdim = [&]() {
    if constexpr (is_edc<edc_type>)
      return std::remove_reference_t<decltype(edc.GetFEValuesState())>::space_dimension;
    else
      return std::remove_reference_t<decltype(edc.GetFEFaceValuesState())>::space_dimension;
  }();
  static_assert(dealdim >= 0);
  const unsigned int n_q_points = edc.GetNQPoints();
  static constexpr unsigned int control_size = dealdim;

  vector<std::vector<Tensor<1, dealdim>>> qgrads(n_q_points,
                                                 std::vector<Tensor<1, dealdim>>(control_size));

  if constexpr (is_edc<edc_type>)
    edc.GetGradsControl(name, qgrads);
  else
    edc.GetFaceGradsControl(name, qgrads);

  return qgrads;
};


const auto get_grads_state = [](const auto &edc, const std::string &name) {
  using edc_type = std::remove_cv_t<std::remove_reference_t<decltype(edc)>>;
  constexpr auto dealdim = [&]() {
    if constexpr (is_edc<edc_type>)
      return std::remove_reference_t<decltype(edc.GetFEValuesState())>::space_dimension;
    else
      return std::remove_reference_t<decltype(edc.GetFEFaceValuesState())>::space_dimension;
  }();
  static_assert(dealdim >= 0);
  const unsigned int n_q_points = edc.GetNQPoints();
  static constexpr unsigned int state_size = 2 * dealdim;

  vector<std::vector<Tensor<1, dealdim>>> ugrads(n_q_points,
                                                 std::vector<Tensor<1, dealdim>>(state_size));

  if constexpr (is_edc<edc_type>)
    edc.GetGradsState(name, ugrads);
  else
    edc.GetFaceGradsState(name, ugrads);

  return ugrads;
};


const auto get_transformed_epsilon_JxW = [](const auto &PD, const auto &edc, const auto qgrads) {
  constexpr auto dealdim =
      std::remove_reference_t<decltype(edc.GetFEValuesState())>::space_dimension;
  const auto &state_fe_values = edc.GetFEValuesState();
  const unsigned int n_q_points = edc.GetNQPoints();

  std::vector<typename ProblemDescription<dealdim>::rank2_type> weighted_epsilon(n_q_points);
  std::vector<typename ProblemDescription<dealdim>::rank2_type> transformation(n_q_points);

  for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {

    const auto &quadrature_points = state_fe_values.get_quadrature_points();
    const auto position = quadrature_points[q_point];
    const auto epsilon = PD->epsilon(position);

    const auto grad_q = qgrads[q_point];
    const auto F_inv_T = FInvT(grad_q);
    transformation[q_point] = F_inv_T;

    const auto J = DetF(grad_q);
    weighted_epsilon[q_point] = epsilon * J * state_fe_values.JxW(q_point);
  }

  return std::make_tuple(weighted_epsilon, transformation);
};


const auto get_delta_transformed_epsilon_JxW =
    [](const auto &PD, const auto &edc, const auto qgrads, const auto q_point) {
      constexpr auto dealdim =
          std::remove_reference_t<decltype(edc.GetFEValuesState())>::space_dimension;
      const auto &control_fe_values = edc.GetFEValuesControl();
      FEValuesViews::Vector<dealdim> fe_values_real(control_fe_values, 0);
#ifdef DEBUG
      const unsigned int n_q_points = edc.GetNQPoints();
      assert(q_point < n_q_points);
#endif
      const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();

      std::vector<typename ProblemDescription<dealdim>::rank2_type> weighted_epsilon(n_dofs_per_element);
      std::vector<typename ProblemDescription<dealdim>::rank2_type> transformation(n_dofs_per_element);

      const auto &quadrature_points = control_fe_values.get_quadrature_points();
      const auto position = quadrature_points[q_point];
      const auto epsilon = PD->epsilon(position);

      const auto grad_q = qgrads[q_point];
      const auto F_inv_T = FInvT(grad_q);
      const auto J = DetF(grad_q);

      for (unsigned int k = 0; k < n_dofs_per_element; k++) {
        const auto delta_grad_q = fe_values_real.gradient(k, q_point);
        const auto delta_F_inv_T = deltaFInvT(F_inv_T, delta_grad_q);
        const auto delta_J = deltaDetF(J, grad_q, delta_grad_q);

        transformation[k] = delta_F_inv_T;
        weighted_epsilon[k] = epsilon * delta_J * control_fe_values.JxW(q_point);
      }

      return std::make_tuple(weighted_epsilon, transformation);
    };


const auto get_transformed_sigma_JxW = [](const auto &PD, const auto &fdc, const auto qgrads) {
  constexpr auto dealdim =
      std::remove_reference_t<decltype(fdc.GetFEFaceValuesState())>::space_dimension;
  const auto &state_fe_values = fdc.GetFEFaceValuesState();
  const unsigned int n_q_points = fdc.GetNQPoints();

  std::vector<typename ProblemDescription<dealdim>::rank2_type> weighted_sigma(n_q_points);
  std::vector<typename ProblemDescription<dealdim>::rank2_type> transformation(n_q_points);

  for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {

    const auto &quadrature_points = state_fe_values.get_quadrature_points();
    const auto position = quadrature_points[q_point];
    const auto normal = state_fe_values.normal_vector(q_point);
    const auto sigma = PD->sigma(position);

    const auto grad_q = qgrads[q_point];
    const auto F_inv_T = FInvT(grad_q);
    transformation[q_point] = F_inv_T;

    auto transformed_normal = F_inv_T * normal;
    transformed_normal /= transformed_normal.norm();
    const auto projector = projector_matrix(transformed_normal);

    const auto J = DetF(grad_q);
    const auto F_inv_T_n_J = (F_inv_T * normal).norm() * J;
    weighted_sigma[q_point] = projector * sigma * projector;
    weighted_sigma[q_point] *= F_inv_T_n_J * state_fe_values.JxW(q_point);
  }

  return std::make_tuple(weighted_sigma, transformation);
};


const auto get_delta_transformed_sigma_JxW =
    [](const auto &PD, const auto &fdc, const auto qgrads, const auto q_point) {
      constexpr auto dealdim =
          std::remove_reference_t<decltype(fdc.GetFEFaceValuesState())>::space_dimension;
      const auto &control_fe_values = fdc.GetFEFaceValuesControl();
      FEValuesViews::Vector<dealdim> fe_values_real(control_fe_values, 0);
#ifdef DEBUG
      const unsigned int n_q_points = fdc.GetNQPoints();
      assert(q_point < n_q_points);
#endif
      const unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();

      std::vector<typename ProblemDescription<dealdim>::rank2_type> weighted_sigma(n_dofs_per_element);
      std::vector<typename ProblemDescription<dealdim>::rank2_type> transformation(n_dofs_per_element);

      const auto &quadrature_points = control_fe_values.get_quadrature_points();
      const auto position = quadrature_points[q_point];
      const auto normal = control_fe_values.normal_vector(q_point);
      const auto sigma = PD->sigma(position);

      const auto grad_q = qgrads[q_point];
      const auto F_inv_T = FInvT(grad_q);

      const auto n = F_inv_T * normal;
      const auto n_hat = n / n.norm();
      const auto projector = projector_matrix(n_hat);

      const auto J = DetF(grad_q);
      const auto F_inv_T_n_J = (F_inv_T * normal).norm() * J;

      for (unsigned int k = 0; k < n_dofs_per_element; k++) {
        const auto delta_grad_q = fe_values_real.gradient(k, q_point);
        const auto delta_F_inv_T = deltaFInvT(F_inv_T, delta_grad_q);
        const auto delta_J = deltaDetF(J, grad_q, delta_grad_q);

        transformation[k] = delta_F_inv_T;

        /*
         * With  \tilde n = n / |n| we have
         *   \delta\tilde n =  \delta n / |n| - n * (n, \delta n) / |n|^3
         */
        const auto delta_n = delta_F_inv_T * normal;
        const auto delta_n_hat = delta_n / n.norm() - n * (n * delta_n) / std::pow(n.norm(), 3);
        const auto delta_projector = -outer_product(delta_n_hat, n_hat) - outer_product(n_hat, delta_n_hat);

        const auto delta_sigma =
            delta_projector * sigma * projector + projector * sigma * delta_projector;

        const auto F_inv_T_n = F_inv_T * normal;
        const auto delta_F_inv_T_n = delta_F_inv_T * normal;
        const auto delta_F_inv_T_n_J =
            F_inv_T_n.norm() * delta_J + (F_inv_T_n * delta_F_inv_T_n) / F_inv_T_n.norm() * J;

        weighted_sigma[k] =
            delta_sigma * F_inv_T_n_J + projector * sigma * projector * delta_F_inv_T_n_J;
        weighted_sigma[k] *= control_fe_values.JxW(q_point);
      }

      return std::make_tuple(weighted_sigma, transformation);
    };
