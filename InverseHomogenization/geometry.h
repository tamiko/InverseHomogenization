/**
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2012-2023 by the DOpElib authors
 * Copyright (C) 2021-2023 by Manaswinee Bezbaruah, Matthias Maier, Winifried Wollner
 **/

#pragma once

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

/*
 * Create a unit cell [0,1]^2 with a mesh-aligned circular interface Sigma
 * centered at (0.5,0.5) and radius @ref radius.
 *
 * We set boundary ids 0 on the left, 2 at the right, 1 at the bottom and 3
 * at the top. We attach a spherical manifold ID to everything except the
 * center cell and boundary faces.
 *
 * Cells "inside" the interface, i.e., with radius less than @ref radius to
 * the center have material ID 2, cells outside ID 1.
 */
template <int dim>
void create_circle(dealii::Triangulation<dim> &triangulation, const double radius)
{
  static_assert(dim == 2, "Not implemented");

  const auto center = Point<dim>(0.5, 0.5);

  AssertThrow(radius < sqrt(0.5), ExcMessage("Oops, radius too large"));

  /* Create a triangulation with circular boundary: */

  Triangulation<dim> tria1;
  GridGenerator::hyper_ball(tria1, center, radius, true);

  /*
   * And create a second triangulation that shares the same boundary points
   * of the disc and has an outer boundary coinciding with the boundary of
   * the unit square:
   */

  const std::vector<Point<dim>> vertices{
      {0.0, 0.0},                      // 0
      {0.0, 0.5 - sqrt(0.5) * radius}, // 1
      {0.0, 0.5 + sqrt(0.5) * radius}, // 2
      {0.0, 1.0},                      // 3

      {0.5 - sqrt(0.5) * radius, 0.0},                      // 4
      {0.5 - sqrt(0.5) * radius, 0.5 - sqrt(0.5) * radius}, // 5
      {0.5 - sqrt(0.5) * radius, 0.5 + sqrt(0.5) * radius}, // 6
      {0.5 - sqrt(0.5) * radius, 1.0},                      // 7

      {0.5 + sqrt(0.5) * radius, 0.0},                      // 8
      {0.5 + sqrt(0.5) * radius, 0.5 - sqrt(0.5) * radius}, // 9
      {0.5 + sqrt(0.5) * radius, 0.5 + sqrt(0.5) * radius}, // 10
      {0.5 + sqrt(0.5) * radius, 1.0},                      // 11

      {1.0, 0.0},                      // 12
      {1.0, 0.5 - sqrt(0.5) * radius}, // 13
      {1.0, 0.5 + sqrt(0.5) * radius}, // 14
      {1.0, 1.0},                      // 15
  };

  std::vector<CellData<dim>> cells(8);
  {
#if DEAL_II_VERSION_GTE(9, 3, 0)
    cells[0].vertices = {0, 4, 1, 5};
    cells[1].vertices = {1, 5, 2, 6};
    cells[2].vertices = {2, 6, 3, 7};
    cells[3].vertices = {4, 8, 5, 9};
    cells[4].vertices = {6, 10, 7, 11};
    cells[5].vertices = {8, 12, 9, 13};
    cells[6].vertices = {9, 13, 10, 14};
    cells[7].vertices = {10, 14, 11, 15};
#else
    const auto assign = [](auto b, std::array<unsigned int, 4> a) {
      std::copy(a.begin(), a.end(), b);
    };
    assign(cells[0].vertices, {0, 4, 1, 5});
    assign(cells[1].vertices, {1, 5, 2, 6});
    assign(cells[2].vertices, {2, 6, 3, 7});
    assign(cells[3].vertices, {4, 8, 5, 9});
    assign(cells[4].vertices, {6, 10, 7, 11});
    assign(cells[5].vertices, {8, 12, 9, 13});
    assign(cells[6].vertices, {9, 13, 10, 14});
    assign(cells[7].vertices, {10, 14, 11, 15});
#endif
  }

  Triangulation<dim> tria2;
  tria2.create_triangulation(vertices, cells, SubCellData());

  GridGenerator::merge_triangulations(tria1, tria2, triangulation);

  /* Colorize boundaries: */

  for (auto face : triangulation.active_face_iterators()) {
    const auto center = face->center();
    constexpr double eps = 1.0e-6;
    if (center[0] < eps) {
      face->set_boundary_id(0);
    } else if (center[0] > 1.0 - eps) {
      face->set_boundary_id(2);
    } else if (center[1] < eps) {
      face->set_boundary_id(1);
    } else if (center[1] > 1.0 - eps) {
      face->set_boundary_id(3);
    }
  }

  /* Material ids: */

  for (auto cell : triangulation.active_cell_iterators()) {
    const auto distance = (cell->center() - center).norm();
    if (distance < radius / sqrt(2.))
      cell->set_material_id(2);
    else
      cell->set_material_id(1);
  }

  /*
   * Attach
   *  - a flat manifold on the outer boundary and the central cell,
   *  - a spherical manifold to faces on the interface,
   *  - a transfinite interpolation manifold to the rest.
   */

  triangulation.set_all_manifold_ids(2);

  triangulation.set_all_manifold_ids_on_boundary(numbers::flat_manifold_id);

  for (auto cell : triangulation.active_cell_iterators()) {
    const auto distance = (cell->center() - center).norm();

    /* central cell: */
    if (distance < 1.0e-6)
      cell->set_all_manifold_ids(numbers::flat_manifold_id);

    /* interface */
    for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f) {
      const auto face = cell->face(f);

      bool on_interface = true;
      for (unsigned int v = 0; v < GeometryInfo<1>::vertices_per_cell; ++v) {
        const auto vertex = face->vertex(v);
        const auto distance = (vertex - center).norm();
        if (std::abs(distance - radius) > 1.0e-10) {
          on_interface = false;
          break;
        }
      }
      if (on_interface)
        face->set_manifold_id(1);
    }
  }

  triangulation.set_manifold(1, SphericalManifold<dim>(center));

  TransfiniteInterpolationManifold<2> transfinite;
  transfinite.initialize(triangulation);
  triangulation.set_manifold(2, transfinite);
}
