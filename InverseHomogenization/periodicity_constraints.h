/**
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2012-2023 by the DOpElib authors
 * Copyright (C) 2021-2023 by Manaswinee Bezbaruah, Matthias Maier, Winifried Wollner
 **/

#pragma once

#include <deal.II/dofs/dof_tools.h>

#include <include/userdefineddofconstraints.h>

template <class DH>
void pin_a_dof(const DH &dof_handler,
               dealii::AffineConstraints<double> &affine_constraints,
               unsigned int component = 0)
{
  Assert(dof_handler.get_triangulation().n_levels() >= 2, dealii::ExcNotImplemented());

  for (auto it = dof_handler.begin_active(); it != dof_handler.end(); ++it) {

    if (it->at_boundary() == false) {

      std::vector<unsigned int> local_dof_indices(it->get_fe().dofs_per_cell);
      it->get_dof_indices(local_dof_indices);
      Assert(local_dof_indices.size() != 0, dealii::ExcInternalError());

      const unsigned int dof = it->get_fe().component_to_system_index(component, 0);
      affine_constraints.add_line(local_dof_indices[dof]);
      affine_constraints.set_inhomogeneity(local_dof_indices[dof], 0.);

      return;
    }
  }
  AssertThrow(false, dealii::ExcInternalError());

  return;
}

namespace DOpE
{
#if DEAL_II_VERSION_GTE(9,3,0)
  template <bool DH, int dim>
  class PeriodicityConstraints : public UserDefinedDoFConstraints<DH, dim>
#else
  template <template <int, int> class DH, int dim>
  class PeriodicityConstraints : public UserDefinedDoFConstraints<DH, dim>
#endif
  {
  public:
    PeriodicityConstraints() = default;

    static void declare_params(ParameterReader &/*param_reader*/)
    {
      /* We currently do not have parameters */
    }

    virtual void MakeStateDoFConstraints(
#if DEAL_II_VERSION_GTE(9,3,0)
        const DOpEWrapper::DoFHandler<dim> & dof_handler,
#else
        const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
#endif
        dealii::AffineConstraints<double> &constraint_matrix) const final override
    {
      DoFTools::make_periodicity_constraints(
          static_cast<const dealii::DoFHandler<dim> &>(dof_handler),
          /*id left */ 0,
          /*id right */ 2,
          /*direction*/ 0,
          constraint_matrix);

      DoFTools::make_periodicity_constraints(
          static_cast<const dealii::DoFHandler<dim> &>(dof_handler),
          /*id left */ 1,
          /*id right */ 3,
          /*direction*/ 1,
          constraint_matrix);

      /* Pin a degree of freedom for every component of the state: */
      for (unsigned int component = 0; component < 2 * dim; ++component)
        pin_a_dof(dof_handler, constraint_matrix, component);
    }

    virtual void MakeControlDoFConstraints(
#if DEAL_II_VERSION_GTE(9,3,0)
        const DOpEWrapper::DoFHandler<dim> & /*dof_handler*/,
#else
        const DOpEWrapper::DoFHandler<dim, DH> & /*dof_handler*/,
#endif
        dealii::AffineConstraints<double> & /*constraint_matrix*/) const final override
    {
      /*
       * Do nothing. We already set Dirichlet boundary conditions in
       * main.cc via
       *   SetControlDirichletBoundaryColors(0, ccomp_mask, &czf);
       */
    }
  };
} // namespace DOpE
