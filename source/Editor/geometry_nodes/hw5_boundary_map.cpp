#include <time.h>

#include <Eigen/Sparse>
#include <cmath>

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
** This file contains two nodes whose primary function is to map the boundary of
** a mesh to a plain convex closed curve (circle or square), setting the stage
** for subsequent Laplacian equation solution and mesh parameterization tasks.
**
** Key to this node's implementation is the adept manipulation of half-edge data
** structures to identify and modify the boundary of the mesh.
**
** Task Overview:
** - The two execution functions (node_map_boundary_to_square_exec,
** node_map_boundary_to_circle_exec) require an update to accurately map the
** mesh boundary to a circle / square. This entails identifying the boundary
** edges, evenly distributing boundary vertices along the target curve's
** perimeter, and ensuring the internal vertices' positions remain unchanged.
** - A focus on half-edge data structures to efficiently traverse and modify
** mesh boundaries.
*/

NODE_DEF_OPEN_SCOPE

/*
** HW4_TODO: Node to map the mesh boundary to a circle.
*/

NODE_DECLARATION_FUNCTION(hw5_circle_boundary_mapping)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");
    // Output-1: Processed 3D mesh whose boundary is mapped to a circle and the
    // interior vertices remains the same
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_circle_boundary_mapping)
{
    // Get the input from params
    auto input = params.get_input<Geometry>("Input");

    // Avoid processing the node when there is no input
    if (!input.get_component<MeshComponent>()) {
        throw std::runtime_error("Boundary Mapping (circle): Need Geometry Input.");
        return false;
    }

    /* ----------------------------- Preprocess -------------------------------
    ** Create a halfedge structure (using OpenMesh) for the input mesh. The
    ** half-edge data structure is a widely used data structure in geometric
    ** processing, offering convenient operations for traversing and modifying
    ** mesh elements.
    */
    auto halfedge_mesh = operand_to_openmesh(&input);

    /* ----------- [HW4_TODO] TASK 2.1: Boundary Mapping (to circle) -----------
    ** In this task, you are required to map the boundary of the mesh to a
    ** circle shape while ensuring the internal vertices remain unaffected.
    ** This step is crucial for setting up the mesh for subsequent
    ** parameterization tasks.
    **
    ** Algorithm Pseudocode for Boundary Mapping to Circle
    ** ------------------------------------------------------------------------
    ** 1. Identify the boundary loop(s) of the mesh using the half-edge
    **    structure.
    **
    ** 2. Calculate the total length of the boundary loop to determine the
    **    spacing between vertices when mapped to a circle.
    **
    ** 3. Sequentially assign each boundary vertex a new position along the
    **    circle's perimeter, maintaining the calculated spacing to ensure
    **    proper distribution.
    **
    ** 4. Keep the interior vertices' positions unchanged during this process.
    **
    ** Note: How to distribute the points on the circle?
    **   Use arc-length parameterization: the fraction of the total boundary
    **   length up to a vertex determines its angle on the circle.
    **
    ** Note: It would be better to normalize the boundary to a unit circle in
    **   [0,1]x[0,1] for texture mapping. Here we map the boundary to a circle
    **   of radius 0.5 centered at (0.5, 0.5), which fits inside the unit square.
    */

    // --- 1. Collect boundary vertices in order ---------------------------------
    // Find a boundary halfedge to start the traversal
    OpenMesh::HalfedgeHandle start_halfedge;
    bool found_boundary = false;
    for (auto halfedge_handle : halfedge_mesh->halfedges()) {
        if (halfedge_mesh->is_boundary(halfedge_handle)) {
            start_halfedge = halfedge_handle;
            found_boundary = true;
            break;
        }
    }
    if (!found_boundary) {
        throw std::runtime_error("Circle mapping: No boundary found in the mesh.");
    }

    // Traverse the boundary loop to obtain an ordered list of boundary vertices
    std::vector<OpenMesh::VertexHandle> boundary_vertices;
    OpenMesh::HalfedgeHandle halfedge_handle = start_halfedge;
    do {
        // The to-vertex of a boundary halfedge is a boundary vertex
        OpenMesh::VertexHandle vertex_to = halfedge_mesh->to_vertex_handle(halfedge_handle);
        boundary_vertices.push_back(vertex_to);
        // Move to the next boundary halfedge
        halfedge_handle = halfedge_mesh->next_halfedge_handle(halfedge_handle);
        // Ensure we stay on the boundary (next boundary halfedge)
        while (halfedge_handle.is_valid() && !halfedge_mesh->is_boundary(halfedge_handle)) {
            halfedge_handle = halfedge_mesh->next_halfedge_handle(
                halfedge_mesh->opposite_halfedge_handle(halfedge_handle));
        }
        if (!halfedge_handle.is_valid()) break;
    } while (halfedge_handle != start_halfedge);

    size_t boundary_count = boundary_vertices.size();
    if (boundary_count < 3) {
        throw std::runtime_error("Circle mapping: Boundary has fewer than 3 vertices.");
    }

    // --- 2. Compute cumulative arc lengths along the boundary ------------------
    std::vector<double> arc_lengths(boundary_count, 0.0);
    double total_length = 0.0;
    for (size_t i = 0; i < boundary_count; ++i) {
        size_t next_idx = (i + 1) % boundary_count;
        auto p0 = halfedge_mesh->point(boundary_vertices[i]);
        auto p1 = halfedge_mesh->point(boundary_vertices[next_idx]);
        double len = (p1 - p0).length();
        total_length += len;
        arc_lengths[next_idx] = total_length; // cumulative length at vertex i+1
    }
    // The first vertex has cumulative length 0 (arc_lengths[0] remains 0)

    // --- 3. Map each boundary vertex to the unit circle ------------------------
    for (size_t i = 0; i < boundary_count; ++i) {
        double t = arc_lengths[i] / total_length; // normalized arc length [0,1)
        double angle = 2.0 * M_PI * t;
        // Map to a circle of radius 0.5, centered at (0.5, 0.5)
        double x = 0.5 + 0.5 * std::cos(angle);
        double y = 0.5 + 0.5 * std::sin(angle);
        halfedge_mesh->set_point(boundary_vertices[i], OpenMesh::Vec3f(x, y, 0.0));
    }

    // Internal vertices remain unchanged (already the case)

    halfedge_mesh->update_normals();

    /* ----------------------------- Postprocess ------------------------------
    ** Convert the result mesh from the halfedge structure back to Geometry
    ** format as the node's output.
    */
    auto geometry = openmesh_to_operand(halfedge_mesh.get());

    // Set the output of the nodes
    params.set_output("Output", std::move(*geometry));
    return true;
}


/*
** HW4_TODO: Node to map the mesh boundary to a square.
*/

NODE_DECLARATION_FUNCTION(hw5_square_boundary_mapping)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");

    // Output-1: Processed 3D mesh whose boundary is mapped to a square and the
    // interior vertices remains the same
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_square_boundary_mapping)
{
    // Get the input from params
    auto input = params.get_input<Geometry>("Input");

    // Avoid processing the node when there is no input
    if (!input.get_component<MeshComponent>()) {
        throw std::runtime_error("Boundary Mapping (square): Need Geometry Input.");
    }

    /* ----------------------------- Preprocess -------------------------------
    ** Create a halfedge structure (using OpenMesh) for the input mesh.
    */
    auto halfedge_mesh = operand_to_openmesh(&input);

    /* ----------- [HW4_TODO] TASK 2.2: Boundary Mapping (to square) -----------
    ** In this task, you are required to map the boundary of the mesh to a
    ** square shape while ensuring the internal vertices remain unaffected.
    **
    ** Algorithm Pseudocode for Boundary Mapping to Square
    ** ------------------------------------------------------------------------
    ** 1. Identify the boundary loop of the mesh (same as circle mapping).
    **
    ** 2. Compute the total length of the boundary.
    **
    ** 3. Distribute boundary vertices along the unit square's perimeter
    **    proportionally to their arc length. The unit square has sides
    **    [0,1]x[0,1].  The perimeter is traversed starting from (0,0)
    **    clockwise: right along bottom edge, up right edge, left along top
    **    edge, down left edge.
    **
    ** 4. Keep the interior vertices' positions unchanged.
    **
    ** Note: Can you preserve the 4 corners of the square after boundary
    ** mapping?  The straightforward proportional mapping will map vertices to
    ** the perimeter; if a vertex lies exactly at a square corner's arc length
    ** fraction, it will become a corner.  This implementation does not force
    ** corners – it simply distributes vertices evenly along the perimeter.
    **
    ** Note: It would be better to normalize the boundary to a unit square in
    ** [0,1]x[0,1] for texture mapping.
    */

    // --- 1. Collect boundary vertices in order ---------------------------------
    OpenMesh::HalfedgeHandle start_halfedge;
    bool found_boundary = false;
    for (auto halfedge_handle : halfedge_mesh->halfedges()) {
        if (halfedge_mesh->is_boundary(halfedge_handle)) {
            start_halfedge = halfedge_handle;
            found_boundary = true;
            break;
        }
    }
    if (!found_boundary) {
        throw std::runtime_error("Square mapping: No boundary found in the mesh.");
    }

    std::vector<OpenMesh::VertexHandle> boundary_vertices;
    OpenMesh::HalfedgeHandle halfedge_handle = start_halfedge;
    do {
        OpenMesh::VertexHandle v_to = halfedge_mesh->to_vertex_handle(halfedge_handle);
        boundary_vertices.push_back(v_to);
        halfedge_handle = halfedge_mesh->next_halfedge_handle(halfedge_handle);
        while (halfedge_handle.is_valid() && !halfedge_mesh->is_boundary(halfedge_handle)) {
            halfedge_handle = halfedge_mesh->next_halfedge_handle(
                halfedge_mesh->opposite_halfedge_handle(halfedge_handle));
        }
        if (!halfedge_handle.is_valid()) break;
    } while (halfedge_handle != start_halfedge);

    size_t boundary_count = boundary_vertices.size();
    if (boundary_count < 4) {
        throw std::runtime_error("Square mapping: Boundary has fewer than 4 vertices.");
    }

    // --- 2. Compute cumulative arc lengths --------------------------------------
    std::vector<double> arc_lengths(boundary_count, 0.0);
    double total_length = 0.0;
    for (size_t i = 0; i < boundary_count; ++i) {
        size_t next_idx = (i + 1) % boundary_count;
        auto p0 = halfedge_mesh->point(boundary_vertices[i]);
        auto p1 = halfedge_mesh->point(boundary_vertices[next_idx]);
        double len = (p1 - p0).length();
        total_length += len;
        arc_lengths[next_idx] = total_length;
    }

    // --- 3. Map each boundary vertex to the unit square perimeter --------------
    // Perimeter of unit square is 4.  Each vertex is placed proportionally.
    for (size_t i = 0; i < boundary_count; ++i) {
        double t = arc_lengths[i] / total_length; // [0, 1)
        double perim = t * 4.0;                  // position along the square perimeter
        double x, y;
        if (perim < 1.0) {
            // Bottom edge: (0,0) -> (1,0)
            x = perim;
            y = 0.0;
        }
        else if (perim < 2.0) {
            // Right edge: (1,0) -> (1,1)
            x = 1.0;
            y = perim - 1.0;
        }
        else if (perim < 3.0) {
            // Top edge: (1,1) -> (0,1) (moving left)
            x = 3.0 - perim;
            y = 1.0;
        }
        else {
            // Left edge: (0,1) -> (0,0) (moving down)
            x = 0.0;
            y = 4.0 - perim;
        }
        halfedge_mesh->set_point(boundary_vertices[i], OpenMesh::Vec3f(x, y, 0.0));
    }

    halfedge_mesh->update_normals();

    /* ----------------------------- Postprocess ------------------------------
    ** Convert the result mesh from the halfedge structure back to Geometry
    ** format as the node's output.
    */
    auto geometry = openmesh_to_operand(halfedge_mesh.get());

    // Set the output of the nodes
    params.set_output("Output", std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(boundary_mapping);
NODE_DEF_CLOSE_SCOPE