#include <time.h>

#include <Eigen/Sparse>
#include <cmath>

//#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include <pxr/usd/usdGeom/mesh.h>

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <cfloat>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"
#include "geom_node_base.h"

/*
** @brief HW4_TutteParameterization
**
** This file presents the basic framework of a "node", which processes inputs
** received from the left and outputs specific variables for downstream nodes to
** use.
** - In the first function, node_declare, you can set up the node's input and
** output variables.
** - The second function, node_exec is the execution part of the node, where we
** need to implement the node's functionality.
** Your task is to fill in the required logic at the specified locations
** within this template, especially in node_exec.
*/

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(hw5_param)
{
    b.add_input<Geometry>("Original Mesh");
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");

    /*
    ** NOTE: You can add more inputs or outputs if necessary. For example, in
    *some cases,
    ** additional information (e.g. other mesh geometry, other parameters) is
    *required to perform
    ** the computation.
    **
    ** Be sure that the input/outputs do not share the same name. You can add
    *one geometry as
    **
    **                b.add_input<Geometry>("Input");
    **
    ** Or maybe you need a value buffer like:
    **
    **                b.add_input<float1Buffer>("Weights");
    */

    // Output-1: Minimal surface with fixed boundary
    b.add_output<Geometry>("Uniform");
    b.add_output<Geometry>("Cotangent");
    b.add_output<Geometry>("Floater");
}

NODE_EXECUTION_FUNCTION(hw5_param)
{
    // Get the input from params
    auto original_input = params.get_input<Geometry>("Original Mesh");
    auto input = params.get_input<Geometry>("Input");

    // (TO BE UPDATED) Avoid processing the node when there is no input
    if (!original_input.get_component<MeshComponent>() || !input.get_component<MeshComponent>()) {
        throw std::runtime_error("Both Original Mesh and Input must contain a mesh component.");
        return false;
    }

    /* ----------------------------- Preprocess -------------------------------
    ** Create a halfedge structure (using OpenMesh) for the input mesh. The
    ** half-edge data structure is a widely used data structure in geometric
    ** processing, offering convenient operations for traversing and modifying
    ** mesh elements.
    */
    auto original_mesh = operand_to_openmesh(&original_input);
    auto target_mesh = operand_to_openmesh(&input);

    if (original_mesh->n_vertices() != target_mesh->n_vertices()) {
        throw std::runtime_error("Original Mesh and Input must have the same number of vertices.");
        return false;
    }

    /* ---------------- [HW4_TODO] TASK 1: Minimal Surface --------------------
    ** In this task, you are required to generate a 'minimal surface' mesh with
    ** the boundary of the input mesh as its boundary.
    **
    ** Specifically, the positions of the boundary vertices of the input mesh
    ** should be fixed. By solving a global Laplace equation on the mesh,
    ** recalculate the coordinates of the vertices inside the mesh to achieve
    ** the minimal surface configuration.
    **
    ** (Recall the Poisson equation with Dirichlet Boundary Condition in HW3)
    */

    /*
    ** Algorithm Pseudocode for Minimal Surface Calculation
    ** ------------------------------------------------------------------------
    ** 1. Initialize mesh with input boundary conditions.
    **    - For each boundary vertex, fix its position.
    **    - For internal vertices, initialize with initial guess if necessary.
    **
    ** 2. Construct Laplacian matrix for the mesh.
    **    - Compute weights for each edge based on the chosen weighting scheme
    **      (e.g., uniform weights for simplicity).
    **    - Assemble the global Laplacian matrix.
    **
    ** 3. Solve the Laplace equation for interior vertices.
    **    - Apply Dirichlet boundary conditions for boundary vertices.
    **    - Solve the linear system (Laplacian * X = 0) to find new positions
    **      for internal vertices.
    **
    ** 4. Update mesh geometry with new vertex positions.
    **    - Ensure the mesh respects the minimal surface configuration.
    **
    ** Note: This pseudocode outlines the general steps for calculating a
    ** minimal surface mesh given fixed boundary conditions using the Laplace
    ** equation. The specific implementation details may vary based on the mesh
    ** representation and numerical methods used.
    **
    */
    int n_vertices = original_mesh->n_vertices();

    std::vector<bool> is_boundary(n_vertices, false);
    for (const auto& vertex_handle : target_mesh->vertices()) {
        is_boundary[vertex_handle.idx()] =
            target_mesh->is_boundary(vertex_handle);
    }

    std::vector<int> internal_idx(n_vertices, -1);
    int internal_count = 0;
    for (int i = 0; i < n_vertices; ++i) {
        if (!is_boundary[i]) {
            internal_idx[i] = internal_count++;
        }
    }

    int n_internal = internal_count;

    auto cotangent = [](const OpenMesh::Vec3f& v0, const OpenMesh::Vec3f& v1, const OpenMesh::Vec3f& v2) {
        OpenMesh::Vec3f e0 = v1 - v0;
        OpenMesh::Vec3f e1 = v2 - v0;
        double cos_angle = (e0 | e1) / (e0.norm() * e1.norm());
        double sin_angle = (e0 % e1).norm() / (e0.norm() * e1.norm());
        return cos_angle / sin_angle;
    };

    Eigen::SparseMatrix<double> A_uniform(n_internal, n_internal);
    Eigen::SparseMatrix<double> A_cotangent(n_internal, n_internal);
    Eigen::SparseMatrix<double> A_floater(n_internal, n_internal);

    std::vector<Eigen::Triplet<double>> triplets_uniform, triplets_cotangent, triplets_floater;

    Eigen::MatrixXd B_uniform(n_internal, 3);
    Eigen::MatrixXd B_cotangent(n_internal, 3);
    Eigen::MatrixXd B_floater(n_internal, 3);

    A_uniform.setZero();
    A_cotangent.setZero();
    A_floater.setZero();
    B_uniform.setZero();
    B_cotangent.setZero();
    B_floater.setZero();

    for (const auto& vertex_handle : original_mesh->vertices()) {
        int idx = vertex_handle.idx();
        if (is_boundary[idx]) continue;

        int internal_i = internal_idx[idx];

        struct EdgeInfo {
            int neighbor_idx;
            double w_uniform, w_cotangent, w_floater;
        };
        std::vector<EdgeInfo> edge_infos;

        double degree_uniform = 0.0, degree_cotangent = 0.0, degree_floater = 0.0;

        for (const auto& neighbor_handle : original_mesh->vv_range(vertex_handle)) {
            int neighbor_idx = neighbor_handle.idx();

            OpenMesh::HalfedgeHandle halfedge_handle = original_mesh->find_halfedge(vertex_handle, neighbor_handle);
            OpenMesh::HalfedgeHandle next_a = original_mesh->next_halfedge_handle(halfedge_handle);
            OpenMesh::VertexHandle opposite_a = original_mesh->to_vertex_handle(next_a);

            OpenMesh::HalfedgeHandle opposite_halfedge_handle = original_mesh->opposite_halfedge_handle(halfedge_handle);
            OpenMesh::HalfedgeHandle next_b = original_mesh->next_halfedge_handle(opposite_halfedge_handle);
            OpenMesh::VertexHandle opposite_b = original_mesh->to_vertex_handle(next_b);

            double cotangent_a = cotangent(
                original_mesh->point(opposite_a),
                original_mesh->point(vertex_handle),
                original_mesh->point(neighbor_handle));
            double cotangent_b = cotangent(
                original_mesh->point(opposite_b),
                original_mesh->point(vertex_handle),
                original_mesh->point(neighbor_handle));

            double cot_ci1 = cotangent(
                original_mesh->point(vertex_handle),
                original_mesh->point(neighbor_handle),
                original_mesh->point(opposite_a));
            double cot_ci2 = cotangent(
                original_mesh->point(vertex_handle),
                original_mesh->point(neighbor_handle),
                original_mesh->point(opposite_b));
            double cot_cj1 = cotangent(
                original_mesh->point(neighbor_handle),
                original_mesh->point(vertex_handle),
                original_mesh->point(opposite_a));
            double cot_cj2 = cotangent(
                original_mesh->point(neighbor_handle),
                original_mesh->point(vertex_handle),
                original_mesh->point(opposite_b));
            double length_ij = (original_mesh->point(vertex_handle) - original_mesh->point(neighbor_handle)).norm();

            double w_uniform = 1.0;
            double w_cotangent = cotangent_a + cotangent_b;
            double w_floater = (cot_ci1 + cot_ci2 + cot_cj1 + cot_cj2) / (length_ij * length_ij);

            degree_uniform += w_uniform;
            degree_cotangent += w_cotangent;
            degree_floater += w_floater;

            edge_infos.push_back({neighbor_idx, w_uniform, w_cotangent, w_floater});
        }

        for (const auto& info : edge_infos) {
            if (degree_uniform == 0.0) {
                throw std::runtime_error("Degree for uniform weights is zero, which may indicate an issue with the mesh connectivity.");
            }
            if (degree_cotangent == 0.0) {
                throw std::runtime_error("Degree for cotangent weights is zero, which may indicate an issue with the mesh connectivity.");
            }
            if (degree_floater == 0.0) {
                throw std::runtime_error("Degree for Floater weights is zero, which may indicate an issue with the mesh connectivity.");
            }
            double norm_uniform = info.w_uniform / degree_uniform;
            double norm_cotangent = info.w_cotangent / degree_cotangent;
            double norm_floater = info.w_floater / degree_floater;

            if (is_boundary[info.neighbor_idx]) {
                auto neighbor_position = target_mesh->point(target_mesh->vertex_handle(info.neighbor_idx));

                B_uniform(internal_i, 0) += norm_uniform * neighbor_position[0];
                B_uniform(internal_i, 1) += norm_uniform * neighbor_position[1];
                B_uniform(internal_i, 2) += norm_uniform * neighbor_position[2];

                B_cotangent(internal_i, 0) += norm_cotangent * neighbor_position[0];
                B_cotangent(internal_i, 1) += norm_cotangent * neighbor_position[1];
                B_cotangent(internal_i, 2) += norm_cotangent * neighbor_position[2];

                B_floater(internal_i, 0) += norm_floater * neighbor_position[0];
                B_floater(internal_i, 1) += norm_floater * neighbor_position[1];
                B_floater(internal_i, 2) += norm_floater * neighbor_position[2];
            } else {
                int internal_j = internal_idx[info.neighbor_idx];
                triplets_uniform.emplace_back(internal_i, internal_j, -norm_uniform);
                triplets_cotangent.emplace_back(internal_i, internal_j, -norm_cotangent);
                triplets_floater.emplace_back(internal_i, internal_j, -norm_floater);
            }
        }

        triplets_uniform.emplace_back(internal_i, internal_i, 1.0);
        triplets_cotangent.emplace_back(internal_i, internal_i, 1.0);
        triplets_floater.emplace_back(internal_i, internal_i, 1.0);
    }
    A_uniform.setFromTriplets(triplets_uniform.begin(), triplets_uniform.end());
    A_cotangent.setFromTriplets(triplets_cotangent.begin(), triplets_cotangent.end());
    A_floater.setFromTriplets(triplets_floater.begin(), triplets_floater.end());

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_uniform, solver_cotangent, solver_floater;
    solver_uniform.compute(A_uniform);
    if (solver_uniform.info() != Eigen::Success) {
        throw std::runtime_error("Decomposition of A_uniform failed");
    }

    solver_cotangent.compute(A_cotangent);
    if (solver_cotangent.info() != Eigen::Success) {
        throw std::runtime_error("Decomposition of A_cotangent failed");
    }

    solver_floater.compute(A_floater);
    if (solver_floater.info() != Eigen::Success) {
        throw std::runtime_error("Decomposition of A_floater failed");
    }

    Eigen::MatrixXd X_uniform = solver_uniform.solve(B_uniform);
    if (solver_uniform.info() != Eigen::Success) {
        throw std::runtime_error("Solving of A_uniform failed");
    }

    Eigen::MatrixXd X_cotangent = solver_cotangent.solve(B_cotangent);
    if (solver_cotangent.info() != Eigen::Success) {
        throw std::runtime_error("Solving of A_cotangent failed");
    }

    Eigen::MatrixXd X_floater = solver_floater.solve(B_floater);
    if (solver_floater.info() != Eigen::Success) {
        throw std::runtime_error("Solving of A_floater failed");
    }

    // Update vertex positions
    auto uniform_mesh = std::make_shared<OpenMesh::PolyMesh_ArrayKernelT<>>(*original_mesh);
    auto cotangent_mesh = std::make_shared<OpenMesh::PolyMesh_ArrayKernelT<>>(*original_mesh);
    auto floater_mesh = std::make_shared<OpenMesh::PolyMesh_ArrayKernelT<>>(*original_mesh);
    for (const auto& vertex_handle : uniform_mesh->vertices()) {
        int idx = vertex_handle.idx();
        if (!is_boundary[idx]) {
            int internal_i = internal_idx[idx];
            float x = X_uniform(internal_i, 0);
            float y = X_uniform(internal_i, 1);
            float z = X_uniform(internal_i, 2);
            uniform_mesh->set_point(vertex_handle, OpenMesh::Vec3f(x, y, z));
        }
        else {
            auto boundary_position = target_mesh->point(target_mesh->vertex_handle(idx));
            uniform_mesh->set_point(vertex_handle, boundary_position);
        }
    }

    for (const auto& vertex_handle : cotangent_mesh->vertices()) {
        int idx = vertex_handle.idx();
        if (!is_boundary[idx]) {
            int internal_i = internal_idx[idx];
            float x = X_cotangent(internal_i, 0);
            float y = X_cotangent(internal_i, 1);
            float z = X_cotangent(internal_i, 2);
            cotangent_mesh->set_point(vertex_handle, OpenMesh::Vec3f(x, y, z));
        }
        else {
            auto boundary_position = target_mesh->point(target_mesh->vertex_handle(idx));
            cotangent_mesh->set_point(vertex_handle, boundary_position);
        }
    }

    for (const auto& vertex_handle : floater_mesh->vertices()) {
        int idx = vertex_handle.idx();
        if (!is_boundary[idx]) {
            int internal_i = internal_idx[idx];
            float x = X_floater(internal_i, 0);
            float y = X_floater(internal_i, 1);
            float z = X_floater(internal_i, 2);
            floater_mesh->set_point(vertex_handle, OpenMesh::Vec3f(x, y, z));
        }
        else {
            auto boundary_position = target_mesh->point(target_mesh->vertex_handle(idx));
            floater_mesh->set_point(vertex_handle, boundary_position);
        }
    }

    uniform_mesh->update_normals();
    cotangent_mesh->update_normals();
    floater_mesh->update_normals();

    /* ----------------------------- Postprocess ------------------------------
    ** Convert the minimal surface mesh from the halfedge structure back to
    ** Geometry format as the node's output.
    */
    auto geometry_uniform = openmesh_to_operand(uniform_mesh.get());
    auto geometry_cotangent = openmesh_to_operand(cotangent_mesh.get());
    auto geometry_floater = openmesh_to_operand(floater_mesh.get());

    // Set the output of the nodes
    params.set_output("Uniform", std::move(*geometry_uniform));
    params.set_output("Cotangent", std::move(*geometry_cotangent));
    params.set_output("Floater", std::move(*geometry_floater));
    return true;
}

NODE_DECLARATION_UI(hw5_param);
NODE_DEF_CLOSE_SCOPE