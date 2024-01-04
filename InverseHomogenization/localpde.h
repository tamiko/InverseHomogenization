/**
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2012-2023 by the DOpElib authors
 * Copyright (C) 2021-2023 by Manaswinee Bezbaruah, Matthias Maier, Winifried Wollner
 **/

#pragma once

#include "helper.h"
#include "problem_description.h"

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <interfaces/pdeinterface.h>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

using namespace std;
using namespace dealii;
using namespace DOpE;

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
  class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
#else
template <template <template <int, int> class DH, typename VECTOR, int dealdim> class EDC,
          template <template <int, int> class DH, typename VECTOR, int dealdim>
          class FDC,
          template <int, int>
          class DH,
          typename VECTOR,
          int dealdim>
  class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
#endif
{
public:
  static constexpr unsigned int control_size = ProblemDescription<dealdim>::control_size;
  static constexpr unsigned int state_size = ProblemDescription<dealdim>::state_size;

  LocalPDE(const ProblemDescription<dealdim> &PD)
      : control_block_component_(control_size, 0)
      , state_block_component_(state_size, 0)
      , PD_(&PD)
  {
    for (unsigned int c = 0; c < dealdim; ++c) {
      state_block_component_[c] = c;
      state_block_component_[dealdim + c] = c;
    }
  }


  /****************************************************************************/


  void ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                       dealii::Vector<double> &local_vector,
                       double scale,
                       double /*scale_ico*/)
  {
    assert(this->problem_type_ == "state");

    const auto &state_fe_values = edc.GetFEValuesState();
    FEValuesViews::Vector<dealdim> fe_values_real(state_fe_values, 0);
    FEValuesViews::Vector<dealdim> fe_values_imag(state_fe_values, dealdim);
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const auto ugrads = get_grads_state(edc, "last_newton_solution"); /*SIC!*/
    const auto qgrads = get_grads_control(edc, "control");
    const auto &[epsilon_JxW, F_inv_T] = get_transformed_epsilon_JxW(PD_, edc, qgrads);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
      constexpr auto identity = ProblemDescription<dealdim>::identity();
      const auto grad_chi = ToJacobian(ugrads[q_point]);

      for (unsigned int i = 0; i < n_dofs_per_element; i++) {
        const auto grad_phi_i =
            fe_values_real.gradient(i, q_point) - 1.0i * fe_values_imag.gradient(i, q_point);

        const auto temp = scalar_product(epsilon_JxW[q_point] *
                                             (identity + F_inv_T[q_point] * transpose(grad_chi)),
                                         F_inv_T[q_point] * transpose(grad_phi_i));
        local_vector(i) += scale * temp.real();
      }
    }
  }

  void ElementEquation_U(const EDC<DH, VECTOR, dealdim> &edc,
                         dealii::Vector<double> &local_vector,
                         double scale,
                         double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint");

    const auto &state_fe_values = edc.GetFEValuesState();
    FEValuesViews::Vector<dealdim> fe_values_real(state_fe_values, 0);
    FEValuesViews::Vector<dealdim> fe_values_imag(state_fe_values, dealdim);
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const auto zgrads = get_grads_state(edc, "last_newton_solution"); /*SIC!*/
    const auto qgrads = get_grads_control(edc, "control");
    const auto &[epsilon_JxW, F_inv_T] = get_transformed_epsilon_JxW(PD_, edc, qgrads);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
      const auto grad_z = ToJacobian(zgrads[q_point], -1.0);

      for (unsigned int i = 0; i < n_dofs_per_element; i++) {
        const auto grad_phi_i =
            fe_values_real.gradient(i, q_point) + 1.0i * fe_values_imag.gradient(i, q_point);

        const auto temp =
            scalar_product(epsilon_JxW[q_point] * (F_inv_T[q_point] * transpose(grad_phi_i)),
                           F_inv_T[q_point] * transpose(grad_z));
        local_vector(i) += scale * temp.real();
      }
    }
  }

  void ElementEquation_Q(const EDC<DH, VECTOR, dealdim> &edc,
                         dealii::Vector<double> &local_vector,
                         double scale,
                         double /*scale_ico*/)
  {
    assert(this->problem_type_ == "gradient");

    const auto &control_fe_values = edc.GetFEValuesControl();
    FEValuesViews::Vector<dealdim> fe_values_real(control_fe_values, 0);
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const auto ugrads = get_grads_state(edc, "state"); /*SIC!*/
    const auto zgrads = get_grads_state(edc, "adjoint"); /*SIC!*/
    const auto qgrads = get_grads_control(edc, "control");
    const auto &[epsilon_JxW, F_inv_T] = get_transformed_epsilon_JxW(PD_, edc, qgrads);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
      constexpr auto identity = ProblemDescription<dealdim>::identity();
      const auto id_grad_chi_T = identity + F_inv_T[q_point] * transpose(ToJacobian(ugrads[q_point]));
      const auto grad_z_T = F_inv_T[q_point] * transpose(ToJacobian(zgrads[q_point], -1.));

      const auto &[delta_epsilon_JxW, delta_F_inv_T] = get_delta_transformed_epsilon_JxW(PD_, edc, qgrads, q_point);

      for (unsigned int k = 0; k < n_dofs_per_element; k++) {
        const auto delta_grad_chi_T = delta_F_inv_T[k] * transpose(ToJacobian(ugrads[q_point]));
        const auto delta_grad_z_T = delta_F_inv_T[k] * transpose(ToJacobian(zgrads[q_point], -1.));
        const auto temp = scalar_product(delta_epsilon_JxW[k] * id_grad_chi_T, grad_z_T) +
                          scalar_product(epsilon_JxW[q_point] * delta_grad_chi_T, grad_z_T) +
                          scalar_product(epsilon_JxW[q_point] * id_grad_chi_T, delta_grad_z_T);

        local_vector(k) += scale * temp.real();
      }
    }
  }

  void ElementRightHandSide(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                            dealii::Vector<double> & /*local_vector*/,
                            double /*scale*/)
  {
    // not needed
  }

  void ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                     FullMatrix<double> &local_matrix,
                     double scale,
                     double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values = edc.GetFEValuesState();
    FEValuesViews::Vector<dealdim> fe_values_real(state_fe_values, 0);
    FEValuesViews::Vector<dealdim> fe_values_imag(state_fe_values, dealdim);
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const auto qgrads = get_grads_control(edc, "control");
    const auto &[epsilon_JxW, F_inv_T] = get_transformed_epsilon_JxW(PD_, edc, qgrads);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
      for (unsigned int i = 0; i < n_dofs_per_element; i++) {

        const auto grad_phi_i =
            fe_values_real.gradient(i, q_point) - 1.0i * fe_values_imag.gradient(i, q_point);

        for (unsigned int j = 0; j < n_dofs_per_element; j++) {

          const auto grad_phi_j =
              fe_values_real.gradient(j, q_point) + 1.0i * fe_values_imag.gradient(j, q_point);

          const auto temp =
              scalar_product(epsilon_JxW[q_point] * F_inv_T[q_point] * transpose(grad_phi_j),
                             F_inv_T[q_point] * transpose(grad_phi_i));

          local_matrix(i, j) += scale * temp.real();
        }
      }
    }
  }


  /****************************************************************************/


  bool HasFaces() const
  {
    return true;
  }

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

  void FaceEquation(const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
                    dealii::Vector<double> &local_vector,
                    double scale,
                    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "state");

    /* Only integrate over interface Sigma */
    if(!AtSigma(fdc))
      return;

    const auto &state_fe_values = fdc.GetFEFaceValuesState();
    FEValuesViews::Vector<dealdim> fe_values_real(state_fe_values, 0);
    FEValuesViews::Vector<dealdim> fe_values_imag(state_fe_values, dealdim);
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();

    const auto ugrads = get_grads_state(fdc, "last_newton_solution"); /*SIC!*/
    const auto qgrads = get_grads_control(fdc, "control");
    const auto &[sigma_JxW, F_inv_T] = get_transformed_sigma_JxW(PD_, fdc, qgrads);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
      constexpr auto identity = ProblemDescription<dealdim>::identity();
      const auto grad_chi = ToJacobian(ugrads[q_point]);

      for (unsigned int i = 0; i < n_dofs_per_element; i++) {
        const auto grad_phi_i =
            fe_values_real.gradient(i, q_point) - 1.0i * fe_values_imag.gradient(i, q_point);

        const auto temp = scalar_product(sigma_JxW[q_point] *
                                             (identity + F_inv_T[q_point] * transpose(grad_chi)),
                                         F_inv_T[q_point] * transpose(grad_phi_i));
        local_vector(i) += 0.5 * scale * (1.0i / PD_->omega() * temp).real();
      }
    }
  }


  void FaceEquation_U(const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
                      dealii::Vector<double> &local_vector,
                      double scale,
                      double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint");

    /* Only integrate over interface Sigma */
    if(!AtSigma(fdc))
      return;

    const auto &state_fe_values = fdc.GetFEFaceValuesState();
    FEValuesViews::Vector<dealdim> fe_values_real(state_fe_values, 0);
    FEValuesViews::Vector<dealdim> fe_values_imag(state_fe_values, dealdim);
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();

    const auto zgrads = get_grads_state(fdc, "last_newton_solution"); /*SIC!*/
    const auto qgrads = get_grads_control(fdc, "control");
    const auto &[sigma_JxW, F_inv_T] = get_transformed_sigma_JxW(PD_, fdc, qgrads);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
      const auto grad_z = ToJacobian(zgrads[q_point], -1.);

      for (unsigned int i = 0; i < n_dofs_per_element; i++) {

        const auto grad_phi_i =
            (fe_values_real.gradient(i, q_point) + 1.0i * fe_values_imag.gradient(i, q_point));

        const auto temp =
            scalar_product(sigma_JxW[q_point] * (F_inv_T[q_point] * transpose(grad_phi_i)),
                           F_inv_T[q_point] * transpose(grad_z));

        local_vector(i) += 0.5 * scale * (1.0i / PD_->omega() * temp).real();
      }
    }
  }


  void FaceEquation_Q(const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
                      dealii::Vector<double> &local_vector,
                      double scale,
                      double /*scale_ico*/)
  {
    assert(this->problem_type_ == "gradient");

    /* Only integrate over interface Sigma */
    if(!AtSigma(fdc))
      return;

    const auto &control_fe_values = fdc.GetFEFaceValuesControl();
    FEValuesViews::Vector<dealdim> fe_values_real(control_fe_values, 0);
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();

    const auto ugrads = get_grads_state(fdc, "state"); /*SIC!*/
    const auto zgrads = get_grads_state(fdc, "adjoint"); /*SIC!*/
    const auto qgrads = get_grads_control(fdc, "control");
    const auto &[sigma_JxW, F_inv_T] = get_transformed_sigma_JxW(PD_, fdc, qgrads);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
      constexpr auto identity = ProblemDescription<dealdim>::identity();
      const auto id_grad_chi_T = identity + F_inv_T[q_point] * transpose(ToJacobian(ugrads[q_point]));
      const auto grad_z_T = F_inv_T[q_point] * transpose(ToJacobian(zgrads[q_point], -1.));

      const auto &[delta_sigma_JxW, delta_F_inv_T] = get_delta_transformed_sigma_JxW(PD_, fdc, qgrads, q_point);

      for (unsigned int k = 0; k < n_dofs_per_element; k++) {
        const auto delta_grad_chi_T = delta_F_inv_T[k] * transpose(ToJacobian(ugrads[q_point]));
        const auto delta_grad_z_T = delta_F_inv_T[k] * transpose(ToJacobian(zgrads[q_point], -1.));
        const auto temp = scalar_product(delta_sigma_JxW[k] * id_grad_chi_T, grad_z_T) +
                          scalar_product(sigma_JxW[q_point] * delta_grad_chi_T, grad_z_T) +
                          scalar_product(sigma_JxW[q_point] * id_grad_chi_T, delta_grad_z_T);

        local_vector(k) += 0.5 * scale * (1.0i / PD_->omega() * temp).real();
      }
    }
  }

  void FaceRightHandSide(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                         dealii::Vector<double> & /*local_vector*/,
                         double /*scale*/)
  {
    // not needed
  }

  void FaceMatrix(const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
                  FullMatrix<double> &local_matrix,
                  double scale,
                  double /*scale_ico*/)
  {
    /* Only integrate over interface Sigma */
    if(!AtSigma(fdc))
      return;

    const auto &state_fe_values = fdc.GetFEFaceValuesState();
    FEValuesViews::Vector<dealdim> fe_values_real(state_fe_values, 0);
    FEValuesViews::Vector<dealdim> fe_values_imag(state_fe_values, dealdim);
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();

    const auto qgrads = get_grads_control(fdc, "control");
    const auto &[sigma_JxW, F_inv_T] = get_transformed_sigma_JxW(PD_, fdc, qgrads);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
      for (unsigned int i = 0; i < n_dofs_per_element; i++) {
        const auto grad_phi_i =
            fe_values_real.gradient(i, q_point) - 1.0i * fe_values_imag.gradient(i, q_point);

        for (unsigned int j = 0; j < n_dofs_per_element; j++) {
          const auto grad_phi_j =
              fe_values_real.gradient(j, q_point) + 1.0i * fe_values_imag.gradient(j, q_point);

          const auto temp =
              scalar_product(sigma_JxW[q_point] * F_inv_T[q_point] * transpose(grad_phi_j),
                             F_inv_T[q_point] * transpose(grad_phi_i));

          local_matrix(i, j) += 0.5 * scale * (1.0i / PD_->omega() * temp).real();
        }
      }
    }
  }


  /****************************************************************************/


  bool HasInterfaces() const
  {
    return false;
  }

  template <typename ELEMENTITERATOR>
  bool AtInterface(ELEMENTITERATOR & /*element*/, unsigned int /*face*/) const
  {
    return false;
  }

  void InterfaceEquation(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                         dealii::Vector<double> & /*local_vector*/,
                         double /*scale*/,
                         double /*scale_ico*/)
  {
    // not needed
  }

  void InterfaceEquation_U(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                           dealii::Vector<double> & /*local_vector*/,
                           double /*scale*/,
                           double /*scale_ico*/)
  {
    // not needed
  }

  void InterfaceMatrix(const FaceDataContainer<DH, VECTOR, dealdim> &/*fdc*/,
                       FullMatrix<double> &/*local_matrix*/,
                       double /*scale*/,
                       double /*scale_ico*/)
  {
    // not needed
  }


  /****************************************************************************/


  void ControlElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                              dealii::Vector<double> &local_vector,
                              double scale)
  {
    // This equation assumes the H^1_0 inner product on the control space
    const DOpEWrapper::FEValues<dealdim> &control_fe_values = edc.GetFEValuesControl();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    assert((this->problem_type_ == "gradient") || (this->problem_type_ == "hessian"));

    const auto funcgradgrads = get_grads_control(edc, "last_newton_solution");

    const FEValuesExtractors::Vector Ctrl(0);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
      for (unsigned int i = 0; i < n_dofs_per_element; i++) {
        local_vector(i) += scale *
                           (scalar_product(ToRe(funcgradgrads[q_point]),
                                           control_fe_values[Ctrl].gradient(i, q_point))) *
                           control_fe_values.JxW(q_point);
      }
    }
  }

  void ControlElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                            FullMatrix<double> &local_matrix,
                            double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &control_fe_values = edc.GetFEValuesControl();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const FEValuesExtractors::Vector Ctrl(0);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
      for (unsigned int i = 0; i < n_dofs_per_element; i++) {
        for (unsigned int j = 0; j < n_dofs_per_element; j++) {
          local_matrix(i, j) += scale *
                                (scalar_product(control_fe_values[Ctrl].gradient(j, q_point),
                                                control_fe_values[Ctrl].gradient(i, q_point))) *
                                control_fe_values.JxW(q_point);
        }
      }
    }
  }


  /****************************************************************************/


  UpdateFlags GetUpdateFlags() const
  {
    return update_values | update_gradients | update_quadrature_points;
  }

  UpdateFlags GetFaceUpdateFlags() const
  {
    return update_values | update_gradients | update_normal_vectors | update_quadrature_points;
  }

  unsigned int GetControlNBlocks() const
  {
    return 1;
  }

  unsigned int GetStateNBlocks() const
  {
    return dealdim;
  }

  std::vector<unsigned int> &GetControlBlockComponent()
  {
    return control_block_component_;
  }

  const std::vector<unsigned int> &GetControlBlockComponent() const
  {
    return control_block_component_;
  }

  std::vector<unsigned int> &GetStateBlockComponent()
  {
    return state_block_component_;
  }

  const std::vector<unsigned int> &GetStateBlockComponent() const
  {
    return state_block_component_;
  }

protected:
private:
  vector<unsigned int> control_block_component_;
  vector<unsigned int> state_block_component_;

  dealii::SmartPointer<const ProblemDescription<dealdim>> PD_;
};
