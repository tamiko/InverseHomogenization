/**
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2012-2023 by the DOpElib authors
 * Copyright (C) 2021-2023 by Manaswinee Bezbaruah, Matthias Maier, Winifried Wollner
 **/

#pragma once

#include "helper.h"
#include "problem_description.h"

#include <deal.II/base/parameter_acceptor.h>

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <interfaces/functionalinterface.h>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

using namespace std;
using namespace dealii;
using namespace DOpE;

#if DEAL_II_VERSION_GTE(9, 3, 0)
template <template <bool DH, typename VECTOR, int dealdim> class EDC,
          template <bool DH, typename VECTOR, int dealdim>
          class FDC,
          bool DH,
          typename VECTOR,
          int dopedim,
          int dealdim = dopedim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>,
                        public dealii::ParameterAcceptor
#else
template <template <template <int, int> class DH, typename VECTOR, int dealdim> class EDC,
          template <template <int, int> class DH, typename VECTOR, int dealdim>
          class FDC,
          template <int, int>
          class DH,
          typename VECTOR,
          int dopedim,
          int dealdim = dopedim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>,
                        public dealii::ParameterAcceptor
#endif
{
public:
  constexpr static unsigned int control_size = dealdim;
  constexpr static unsigned int state_size = 2 * dealdim;


  LocalFunctional(const ProblemDescription<dealdim> &PD)
      : ParameterAcceptor("algorithm setup")
      , PD_(&PD)
  {
    with_complex_part_ = true;
    this->add_parameter(
        "with complex part",
        with_complex_part_,
        "If set to true the full Frobenius norm of the effective tensor and the target tensor is "
        "computed. If set to false only the real parts are considered.");

    interface_penalty_.push_back(0.1);
    this->add_parameter(
        "interface penalty",
        interface_penalty_,
        "Additional penalty for Thikhonov regularization close to the interface, scaled by 1/h");

    alpha_.push_back(0.1);
    this->add_parameter("alpha", alpha_, "Tikhonov regularization parameter");

    beta_1_.push_back(0.1);
    this->add_parameter("beta 1", beta_1_, "Jacobian penalization parameter");

    beta_2_.push_back(0.1);
    this->add_parameter("beta 2", beta_2_, "Jacobian penalization parameter");

    current_iteration_ = 0;

    /* We store a complex-valued tensor of rank 2: */
    n_values_ = 2 * dealdim * dealdim;

    ParameterAcceptor::parse_parameters_call_back.connect(
        std::bind(&LocalFunctional::parse_parameters_callback, this));
  }

  void parse_parameters_callback()
  {
    // FIXME use OutputHandler
    std::cout << "(New) regularization parameters:" << std::endl;
    std::cout << "\talpha  = " << alpha_[current_iteration_]
              << " (interface penalty = " << interface_penalty_[current_iteration_] << ")"
              << std::endl;
    std::cout << "\tbeta_1 = " << beta_1_[current_iteration_] << std::endl;
    std::cout << "\tbeta_2 = " << beta_2_[current_iteration_] << std::endl;
  }

  double AlgebraicValue(const std::map<std::string, const dealii::Vector<double> *> &param_values,
                        const std::map<std::string, const VECTOR *> & /*domain_values*/)
  {
    assert(this->GetProblemType() == "cost_functional");

    auto it = param_values.find("cost_functional_pre");
    assert(it != param_values.end());
    const auto epsilon_eff = compute_epsilon_eff(*(it->second));

    // FIXME use OutputHandler
    std::cout << "\tepsilon eff: " << epsilon_eff << std::endl;

    const auto d = epsilon_eff - PD_->target_epsilon();
    const auto weighted_d = schur_product(d, PD_->weights());

    auto error_fr = (::real(d)).norm_square();
    if (with_complex_part_)
      error_fr += (::imag(d)).norm_square();

    // FIXME use OutputHandler
    std::cout << "\tdeviation (Frobenius norm): " << std::sqrt(error_fr) << std::endl;

    double ret = 0.5 * scalar_product(::real(weighted_d), ::real(d));
    if (with_complex_part_)
      ret += 0.5 * scalar_product(::imag(weighted_d), ::imag(d));

    std::cout << "\tdeviation (weighted norm):  " << std::sqrt(2.0 * ret) << std::endl;

    return ret;
  }

  double ElementValue(const EDC<DH, VECTOR, dealdim> &edc)
  {
    double ret = 0.;

    if (this->GetProblemType() == "cost_functional_pre") {
      /* Compute effective tensor: */
      const auto &[i, j, is_imaginary_part] = from_linear_range(this->GetProblemNum());

      unsigned int n_q_points = edc.GetNQPoints();

      const auto ugrads = get_grads_state(edc, "state");
      const auto qgrads = get_grads_control(edc, "control");
      const auto &[epsilon_JxW, F_inv_T] = get_transformed_epsilon_JxW(PD_, edc, qgrads);

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
        constexpr auto identity = ProblemDescription<dealdim>::identity();

        const auto id_grad_chi_j = // select j-th component:
            identity[j] + F_inv_T[q_point] * ToJacobian(ugrads[q_point])[j];
        const auto id_grad_chi_i_conj = // select i-th component:
            identity[i] + F_inv_T[q_point] * ToJacobian(ugrads[q_point], -1.)[i];

        const auto temp = scalar_product(epsilon_JxW[q_point] * id_grad_chi_j, id_grad_chi_i_conj);

        if (is_imaginary_part) {
          ret += temp.imag();
        } else {
          ret += temp.real();
        }
      }

    } else {
      assert(this->GetProblemType() == "cost_functional");

      /* Stabilization term: */
      const auto &control_fe_values = edc.GetFEValuesControl();
      unsigned int n_q_points = edc.GetNQPoints();

      const auto qgrads = get_grads_control(edc, "control");

      const auto factor =
          AtSigma(edc) ? interface_penalty_[current_iteration_] / edc.GetElementDiameter() : 0.;

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
        const auto JxW = control_fe_values.JxW(q_point);
        const auto grad_q = qgrads[q_point];
        const auto J = DetF(grad_q);

        // Compute the regularization term here H^1_0 norm
        ret += 0.5 * alpha_[current_iteration_] * (1. + factor) *
               scalar_product(ToRe(grad_q), ToRe(grad_q)) * JxW;

        if (J < 1.)
          ret += beta_1_[current_iteration_] * (J - 1.) * (J - 1.) / (std::abs(J) + J) * JxW;
        else
          ret += beta_2_[current_iteration_] * (J - 1.) * (J - 1.) * JxW;
      }
    }
    return ret;
  }


  void ElementValue_U(const EDC<DH, VECTOR, dealdim> &edc,
                      dealii::Vector<double> &local_vector,
                      double scale)
  {
    assert(this->GetProblemType() == "adjoint");

    dealii::Vector<double> pre_values(n_values_);
    edc.GetParamValues("cost_functional_pre", pre_values);
    const auto epsilon_eff = compute_epsilon_eff(pre_values);
    /* we have to take the derivative of g = d_ij conj(d_ij), where: */
    const auto d = epsilon_eff - PD_->target_epsilon();
    const auto weighted_d = schur_product(d, PD_->weights());

    const auto &state_fe_values = edc.GetFEValuesState();
    FEValuesViews::Vector<dealdim> fe_values_real(state_fe_values, 0);
    FEValuesViews::Vector<dealdim> fe_values_imag(state_fe_values, dealdim);
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const auto ugrads = get_grads_state(edc, "state");
    const auto qgrads = get_grads_control(edc, "control");
    const auto &[epsilon_JxW, F_inv_T] = get_transformed_epsilon_JxW(PD_, edc, qgrads);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {

      constexpr auto identity = ProblemDescription<dealdim>::identity();

      const auto id_grad_chi_T =
          identity + F_inv_T[q_point] * transpose(ToJacobian(ugrads[q_point]));
      const auto id_grad_chi_conj =
          identity + ToJacobian(ugrads[q_point], -1.) * transpose(F_inv_T[q_point]);

      for (unsigned int k = 0; k < n_dofs_per_element; k++) {
        const auto grad_phi_k_T =
            F_inv_T[q_point] * transpose(fe_values_real.gradient(k, q_point) +
                                         +1.0i * fe_values_imag.gradient(k, q_point));
        const auto grad_phi_k_conj =
            (fe_values_real.gradient(k, q_point) - 1.0i * fe_values_imag.gradient(k, q_point)) *
            transpose(F_inv_T[q_point]);

        const auto delta_epsilon_eff = id_grad_chi_conj * (epsilon_JxW[q_point] * grad_phi_k_T) +
                                       grad_phi_k_conj * (epsilon_JxW[q_point] * id_grad_chi_T);

        /*
         * Trick: We have to compute 0.5 * (|z|^2)', which reduces to
         *   0.5 * (|z|^2)' = Re(z) * Re(\delta z) + Imag(z) * Imag(\delta z)
         */

        double temp = scalar_product(::real(weighted_d), ::real(delta_epsilon_eff));
        if (with_complex_part_)
          temp += scalar_product(::imag(weighted_d), ::imag(delta_epsilon_eff));

        local_vector(k) += scale * temp;
      }
    }
  }


  void ElementValue_Q(const EDC<DH, VECTOR, dealdim> &edc,
                      dealii::Vector<double> &local_vector,
                      double scale)
  {
    assert(this->GetProblemType() == "gradient");

    dealii::Vector<double> pre_values(n_values_);
    edc.GetParamValues("cost_functional_pre", pre_values);
    const auto epsilon_eff = compute_epsilon_eff(pre_values);
    /* we have to take the derivative of g = \sum_ij d_ij conj(d_ij), where: */
    const auto d = epsilon_eff - PD_->target_epsilon();
    const auto weighted_d = schur_product(d, PD_->weights());

    const auto &control_fe_values = edc.GetFEValuesControl();
    FEValuesViews::Vector<dealdim> fe_values_real(control_fe_values, 0);
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const auto ugrads = get_grads_state(edc, "state");
    const auto qgrads = get_grads_control(edc, "control");
    const auto &[epsilon_JxW, F_inv_T] = get_transformed_epsilon_JxW(PD_, edc, qgrads);

    const auto factor =
        AtSigma(edc) ? interface_penalty_[current_iteration_] / edc.GetElementDiameter() : 0.;

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
      const auto JxW = control_fe_values.JxW(q_point);

      constexpr auto identity = ProblemDescription<dealdim>::identity();
      const auto id_grad_chi_T =
          identity + F_inv_T[q_point] * transpose(ToJacobian(ugrads[q_point]));
      const auto id_grad_chi_conj =
          identity + ToJacobian(ugrads[q_point], -1.) * transpose(F_inv_T[q_point]);

      const auto grad_q = qgrads[q_point];
      const auto J = DetF(grad_q);
      const auto &[delta_epsilon_JxW, delta_F_inv_T] =
          get_delta_transformed_epsilon_JxW(PD_, edc, qgrads, q_point);

      for (unsigned int k = 0; k < n_dofs_per_element; k++) {

        const auto delta_grad_chi_T = //
            delta_F_inv_T[k] * transpose(ToJacobian(ugrads[q_point]));
        const auto delta_grad_chi_conj =
            ToJacobian(ugrads[q_point], -1.) * transpose(delta_F_inv_T[k]);

        const auto delta_epsilon_eff =
            delta_grad_chi_conj * (epsilon_JxW[q_point] * id_grad_chi_T) +
            id_grad_chi_conj * (delta_epsilon_JxW[k] * id_grad_chi_T) +
            id_grad_chi_conj * (epsilon_JxW[q_point] * delta_grad_chi_T);

        const auto delta_grad_q = fe_values_real.gradient(k, q_point);

        double temp = scalar_product(::real(weighted_d), ::real(delta_epsilon_eff)) +
                      alpha_[current_iteration_] * (1. + factor) *
                          scalar_product(ToRe(grad_q), delta_grad_q) * JxW;
        if (with_complex_part_)
          temp += scalar_product(::imag(weighted_d), ::imag(delta_epsilon_eff));

        const auto delta_J = deltaDetF(J, grad_q, delta_grad_q);
        if (J < 1.) {
          temp += beta_1_[current_iteration_] * 2. * (J - 1.) / (std::abs(J) + J) * delta_J * JxW;
          temp += beta_1_[current_iteration_] * (J - 1.) * (J - 1.) / //
                  (std::abs(J) + J) / (std::abs(J) + J) * (-2.) * delta_J * JxW;
        } else {
          temp += beta_2_[current_iteration_] * 2. * (J - 1.) * delta_J * JxW;
        }

        local_vector(k) += scale * temp;
      }
    }
  }


  double FaceValue(const FDC<DH, VECTOR, dealdim> &fdc)
  {
    double ret = 0.;

    if (!AtSigma(fdc))
      return ret;

    if (this->GetProblemType() == "cost_functional_pre") {
      /* Compute effective tensor: */
      const auto &[i, j, is_imaginary_part] = from_linear_range(this->GetProblemNum());

      unsigned int n_q_points = fdc.GetNQPoints();

      const auto ugrads = get_grads_state(fdc, "state");
      const auto qgrads = get_grads_control(fdc, "control");
      const auto &[sigma_JxW, F_inv_T] = get_transformed_sigma_JxW(PD_, fdc, qgrads);

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
        constexpr auto identity = ProblemDescription<dealdim>::identity();
        const auto id_grad_chi_j = // select j-th component:
            identity[j] + F_inv_T[q_point] * ToJacobian(ugrads[q_point])[j];
        const auto id_grad_chi_i_conj = // select i-th component:
            identity[i] + F_inv_T[q_point] * ToJacobian(ugrads[q_point], -1.)[i];

        const auto temp = scalar_product(sigma_JxW[q_point] * id_grad_chi_j, id_grad_chi_i_conj);

        if (is_imaginary_part) {
          ret += (0.5 * 1.0i / PD_->omega() * temp).imag();
        } else {
          ret += (0.5 * 1.0i / PD_->omega() * temp).real();
        }
      }
    }
    return ret;
  }


  void FaceValue_U(const FDC<DH, VECTOR, dealdim> &fdc,
                   dealii::Vector<double> &local_vector,
                   double scale)
  {
    assert(this->GetProblemType() == "adjoint");

    if (!AtSigma(fdc))
      return;

    dealii::Vector<double> pre_values(n_values_);
    fdc.GetParamValues("cost_functional_pre", pre_values);
    const auto epsilon_eff = compute_epsilon_eff(pre_values);
    /* we have to take the derivative of g = d_ij conj(d_ij), where: */
    const auto d = epsilon_eff - PD_->target_epsilon();
    const auto weighted_d = schur_product(d, PD_->weights());

    const auto &state_fe_values = fdc.GetFEFaceValuesState();
    FEValuesViews::Vector<dealdim> fe_values_real(state_fe_values, 0);
    FEValuesViews::Vector<dealdim> fe_values_imag(state_fe_values, dealdim);
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();

    const auto ugrads = get_grads_state(fdc, "state");
    const auto qgrads = get_grads_control(fdc, "control");
    const auto &[sigma_JxW, F_inv_T] = get_transformed_sigma_JxW(PD_, fdc, qgrads);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
      constexpr auto identity = ProblemDescription<dealdim>::identity();
      const auto id_grad_chi_T =
          identity + F_inv_T[q_point] * transpose(ToJacobian(ugrads[q_point]));
      const auto id_grad_chi_conj =
          identity + ToJacobian(ugrads[q_point], -1.) * transpose(F_inv_T[q_point]);

      for (unsigned int k = 0; k < n_dofs_per_element; k++) {
        const auto grad_phi_k_T =
            F_inv_T[q_point] * transpose(fe_values_real.gradient(k, q_point) +
                                         1.0i * fe_values_imag.gradient(k, q_point));
        const auto grad_phi_k_conj =
            (fe_values_real.gradient(k, q_point) - 1.0i * fe_values_imag.gradient(k, q_point)) *
            transpose(F_inv_T[q_point]);

        auto delta_sigma_eff = id_grad_chi_conj * (sigma_JxW[q_point] * grad_phi_k_T) +
                               grad_phi_k_conj * (sigma_JxW[q_point] * id_grad_chi_T);

        delta_sigma_eff *= 1.0i / PD_->omega();

        /*
         * Trick: We have to compute 0.5 * (|z|^2)', which reduces to
         *   0.5 * (|z|^2)' = Re(z) * Re(\delta z) + Imag(z) * Imag(\delta z)
         */

        double temp = scalar_product(::real(weighted_d), ::real(delta_sigma_eff)) +
                      scalar_product(::imag(weighted_d), ::imag(delta_sigma_eff));

        local_vector(k) += 0.5 * scale * temp;
      }
    }
  }


  void FaceValue_Q(const FDC<DH, VECTOR, dealdim> &fdc,
                   dealii::Vector<double> &local_vector,
                   double scale)
  {
    assert(this->GetProblemType() == "gradient");

    if (!AtSigma(fdc))
      return;

    dealii::Vector<double> pre_values(n_values_);
    fdc.GetParamValues("cost_functional_pre", pre_values);
    const auto epsilon_eff = compute_epsilon_eff(pre_values);
    /* we have to take the derivative of g = \sum_ij d_ij conj(d_ij), where: */
    const auto d = epsilon_eff - PD_->target_epsilon();
    const auto weighted_d = schur_product(d, PD_->weights());

    const auto &control_fe_values = fdc.GetFEFaceValuesControl();
    FEValuesViews::Vector<dealdim> fe_values_real(control_fe_values, 0);
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();

    const auto ugrads = get_grads_state(fdc, "state");
    const auto qgrads = get_grads_control(fdc, "control");
    const auto &[sigma_JxW, F_inv_T] = get_transformed_sigma_JxW(PD_, fdc, qgrads);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
      constexpr auto identity = ProblemDescription<dealdim>::identity();
      const auto id_grad_chi_T =
          identity + F_inv_T[q_point] * transpose(ToJacobian(ugrads[q_point]));
      const auto id_grad_chi_conj =
          identity + ToJacobian(ugrads[q_point], -1.) * transpose(F_inv_T[q_point]);

      const auto &[delta_sigma_JxW, delta_F_inv_T] =
          get_delta_transformed_sigma_JxW(PD_, fdc, qgrads, q_point);

      for (unsigned int k = 0; k < n_dofs_per_element; k++) {
        const auto delta_grad_chi_T = //
            delta_F_inv_T[k] * transpose(ToJacobian(ugrads[q_point]));
        const auto delta_grad_chi_conj =
            ToJacobian(ugrads[q_point], -1.) * transpose(delta_F_inv_T[k]);

        auto delta_sigma_eff = delta_grad_chi_conj * (sigma_JxW[q_point] * id_grad_chi_T) +
                               id_grad_chi_conj * (delta_sigma_JxW[k] * id_grad_chi_T) +
                               id_grad_chi_conj * (sigma_JxW[q_point] * delta_grad_chi_T);
        delta_sigma_eff *= 1.0i / PD_->omega();

        double temp = scalar_product(::real(weighted_d), ::real(delta_sigma_eff)) +
                      scalar_product(::imag(weighted_d), ::imag(delta_sigma_eff));

        local_vector(k) += 0.5 * scale * temp;
      }
    }
  }


  UpdateFlags GetUpdateFlags() const
  {
    return update_values | update_gradients | update_quadrature_points;
  }


  UpdateFlags GetFaceUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients | update_normal_vectors;
  }


  string GetType() const
  {
    // More complicated selection to avoid implementing unneeded
    // derivatives
    // clang-format off
    if (this->GetProblemType() == "cost_functional_pre"            // Only calculates integral for effective tensor
        || this->GetProblemType() == "cost_functional_pre_tangent" // Only calculates effective tensor_derivative
        || this->GetProblemType() == "adjoint"                     // J'_u is calculated as a domain integral!
        || this->GetProblemType() == "gradient"                    // J'_q is calculated as a domain integral!
        || this->GetProblemType() == "adjoint_hessian"             // J'_{uu} is calculated as a domain integral!
        || this->GetProblemType() == "hessian"                     // J'_{qq} is calculated as a domain integral!
    )
      // clang-format on
      return "domain face";

    else {

      if (this->GetProblemType() == "cost_functional") {
        return "domain algebraic face";

      } else {

        std::cout << "Unknown type ,," << this->GetProblemType() << "'' in LocalFunctional::GetType"
                  << std::endl;
        abort();
      }
    }
  }


  string GetName() const
  {
    return "cost functional";
  }


  bool HasInterfaces() const
  {
    return false;
  }


  unsigned int NeedPrecomputations() const
  {
    // With the given interface we have to compute 2 * dim * dim integrals
    // separately for the real and imaginary part of every component.
    return n_values_;
  }

  void signal_next_iteration_by_incrementing_current_iteration_variable()
  {
    ++current_iteration_;
    parse_parameters_callback();
  }

protected:
  bool AtSigma(const FaceDataContainer<DH, VECTOR, dealdim> &fdc) const
  {
    /* Return false at boundary: */
    if (fdc.GetIsAtBoundary())
      return false;

    /*
     * Return true if the face is located at a material interface, i.e., if
     * the material id changes:
     */
    return fdc.GetMaterialId() != fdc.GetNbrMaterialId();
  }


  bool AtSigma(const ElementDataContainer<DH, VECTOR, dealdim> &edc) const
  {
    /* Return false at boundary: */
    if (edc.GetIsAtBoundary())
      return false;

    /*
     * Return true if the face is located at a material interface, i.e., if
     * the material id changes:
     */
    static_assert(dealdim == 2, "not implemented");
    bool result = false;
    for (unsigned int f = 0; f < 4; ++f)
      if (edc.GetMaterialId() != edc.GetNbrMaterialId(f)) {
        result = true;
        break;
      }

    return result;
  }


  template <typename T>
  inline Tensor<2, dealdim, std::complex<double>> compute_epsilon_eff(const T &values)
  {
    Tensor<2, dealdim, std::complex<double>> epsilon_eff;
    for (unsigned int i = 0; i < dealdim; ++i)
      for (unsigned int j = 0; j < dealdim; ++j)
        epsilon_eff[i][j] =
            values[to_linear_range(std::make_tuple(i, j, /*is_imaginary_part*/ false))] +
            1.0i * values[to_linear_range(std::make_tuple(i, j, /*is_imaginary_part*/ true))];

    return epsilon_eff;
  }


  inline unsigned int to_linear_range(const std::tuple<unsigned int, unsigned int, bool> indices)
  {
    const auto i = std::get<0>(indices);
    const auto j = std::get<1>(indices);
    const auto is_imaginary_part = std::get<2>(indices);
    return i + dealdim * j + dealdim * dealdim * static_cast<unsigned int>(is_imaginary_part);
  }


  inline std::tuple<unsigned int, unsigned int, bool> from_linear_range(const unsigned int index)
  {
    unsigned int i = index % dealdim;
    unsigned int j = (index / dealdim) % dealdim;
    unsigned int is_imaginary_part = index / (dealdim * dealdim);
    return std::make_tuple(i, j, static_cast<bool>(is_imaginary_part));
  }

private:
  bool with_complex_part_;

  std::vector<double> interface_penalty_;
  std::vector<double> alpha_;
  std::vector<double> beta_1_;
  std::vector<double> beta_2_;

  unsigned int current_iteration_;

  unsigned int n_values_;

  dealii::SmartPointer<const ProblemDescription<dealdim>> PD_;
};
